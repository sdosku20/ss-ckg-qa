"""
ikgqa.pcst.gw
=============

The Goemans-Williamson primal-dual algorithm for the Prize-Collecting Steiner
Tree problem, implemented here from the papers rather than called from a
library.

Sources, and what each one supplies:

  * Bienstock, Goemans, Simchi-Levi & Williamson, *A note on the prize
    collecting traveling salesman problem*, Math. Prog. 59 (1993) 413-420.
    Section 4 defines the problem this module solves: find a subtree T
    minimising ``c(T) + pi(complement of T)`` -- the cost of the edges bought
    plus the penalties of the nodes left out.

  * Goemans & Williamson, *A general approximation technique for constrained
    forest problems*, SIAM J. Comput. 24(2) (1995) 296-317, and their
    primal-dual survey chapter (Hochbaum, ed., 1997). The growth stage below is
    their moat-growing process, and the two invariants it maintains are the two
    dual constraints.

  * Johnson, Minkoff & Phillips, *The prize collecting Steiner tree problem:
    theory and practice*, SODA 2000, 760-769. Strong pruning: solve PCST
    exactly on the growth-stage tree by dynamic programming. Their result is
    that this is never worse than GW's own pruning rule, which we also
    implement so the published behaviour can be reproduced exactly.

  * Hegde, Indyk & Schmidt, *A fast, adaptive variant of the GW scheme for
    PCST*, DIMACS 11th Implementation Challenge (2014). Their Section 2 states
    the growth stage as pseudo-code and names the two invariants; their Algorithm
    1 is what `_growth` follows. Their speed-ups (dynamic edge splitting with
    sentinel nodes) are deliberately NOT implemented -- see "Complexity" below.

The public entry point is signature-compatible with `pcst_fast.pcst_fast`, so it
is a drop-in replacement:

    vertices, edges = solve(edges, prizes, costs, root, num_clusters, pruning, verbosity)

Only the unrooted, single-component case is supported, because that is the only
one G-Retriever uses. Anything else raises rather than quietly doing something
else.

Complexity. Each iteration of the growth loop is O(m) with numpy, and there are
at most 2n iterations (every iteration either merges two clusters or deactivates
one), so the growth stage is O(n*m). Hegde et al. get O(m log n) by splitting
edges at sentinel nodes so that all live slacks drain at one rate; that is a
data-structure optimisation which changes no output, and it is left out here in
favour of code that can be read against the papers. On the subgraph sizes this
thesis feeds it (two-stage retrieval caps the region at 2,000 nodes) the
difference does not matter.

Correctness is checked two ways in `tests/test_gw.py`: the objective value must
never be worse than `pcst_fast`'s, and on `pruning="strong"` the node and edge
sets must match `pcst_fast` exactly on randomly generated instances.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

PRUNING_MODES = ("none", "gw", "strong")

# Slack and prize comparisons are float; a tolerance keeps an exactly-tight
# constraint from being missed to a rounding error, which would deadlock the
# growth loop.
EPS = 1e-9


# ---------------------------------------------------------------------------
# Growth stage
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Growth:
    """Everything the growth stage produced, for the pruning stage to use.

    Attributes:
        forest: edge ids, in the order they were bought.
        dead: for each cluster deactivated *because its own cluster constraint
            went tight*, the frozenset of its members. A cluster that stopped
            being active only because it was merged into a bigger one is not
            listed: it did not run out of prize, it was absorbed. GW pruning
            depends on exactly this distinction.
        events: (time, kind, detail) triples, for teaching and debugging.
    """

    forest: List[int]
    dead: List[frozenset]
    events: List[tuple]
    final: List[frozenset] = dataclasses.field(default_factory=list)
    """The maximal clusters at the end of growth, one per connected component.

    These are never prunable. Every cluster eventually runs out of prize,
    including the last one, so testing "did it run out?" alone would mark a whole
    component prunable and delete the answer. On a connected graph the last
    cluster spans everything and a size test would have caught it; on a
    disconnected graph it does not, which is exactly the case that broke.
    """


def _growth(
    edge_ends: np.ndarray,
    prizes: np.ndarray,
    costs: np.ndarray,
    verbosity: int = 0,
) -> Growth:
    """Grow moats until no cluster is active. GW95; Hegde et al. Algorithm 1.

    Two invariants hold throughout, and they are the dual constraints of the LP
    in section 4 of Bienstock et al.:

        edge constraint     sum of moats crossing e  <=  cost(e)
        cluster constraint  sum of moats inside C    <=  prize(C)

    Growing every active moat at unit rate, an edge's slack drains at one unit
    per *active endpoint*: two active endpoints close the gap twice as fast as
    one. When an edge constraint goes tight its two clusters merge, and the new
    cluster is active with a fresh moat of zero but carries forward everything
    its parts already spent. When a cluster constraint goes tight that cluster
    deactivates for good.
    """
    n = int(prizes.shape[0])
    m = int(edge_ends.shape[0])
    src = edge_ends[:, 0].astype(np.int64)
    dst = edge_ends[:, 1].astype(np.int64)

    # comp[v] is a stable component label. On a merge only the smaller side is
    # relabelled, which keeps the total relabelling work to O(n log n).
    comp = np.arange(n, dtype=np.int64)
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}

    # rec[c] is the cluster *record* currently representing component c. A merge
    # creates a new record (moat back to zero) while reusing the surviving
    # component label, which is why these are two separate arrays.
    rec = np.arange(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    spent = np.zeros(n, dtype=np.float64)  # sum of moats in this record and below
    worth = prizes.astype(np.float64).copy()  # total prize of the record's members
    record_members: Dict[int, frozenset] = {i: frozenset((i,)) for i in range(n)}

    slack = costs.astype(np.float64).copy()
    alive = np.ones(m, dtype=bool)  # False once both ends are in one cluster

    forest: List[int] = []
    dead: List[frozenset] = []
    events: List[tuple] = []
    clock = 0.0

    def log(kind: str, detail: str) -> None:
        events.append((clock, kind, detail))
        if verbosity:
            print(f"t={clock:.6g}  {kind}: {detail}")

    # A prize-zero node has a tight cluster constraint before the clock starts,
    # so it is never active and can only enter the answer by being swallowed in
    # a merge. That is precisely what a Steiner node is.
    for i in range(n):
        if active[i] and worth[i] - spent[i] <= EPS:
            active[i] = False
            dead.append(record_members[i])
            log("deactivate", f"{sorted(record_members[i])} (no prize to spend)")

    while active.any():
        r_src = rec[comp[src]]
        r_dst = rec[comp[dst]]
        internal = r_src == r_dst
        if internal.any():
            alive &= ~internal

        rate = (active[r_src].astype(np.int8) + active[r_dst].astype(np.int8)) * alive
        can = alive & (rate > 0)

        best_time = np.inf
        best_edge = -1
        if can.any():
            times = np.where(can, slack / np.maximum(rate, 1), np.inf)
            best_edge = int(np.argmin(times))
            best_time = float(times[best_edge])

        room = np.where(active, worth - spent, np.inf)
        best_cluster = int(np.argmin(room))
        cluster_time = float(room[best_cluster])

        # A cluster going tight at the same moment as an edge is resolved in the
        # cluster's favour: it has no prize left, so it cannot pay for the edge.
        if cluster_time <= best_time + EPS:
            step = max(cluster_time, 0.0)
            kind = "cluster"
        else:
            step = max(best_time, 0.0)
            kind = "edge"

        clock += step
        if step > 0:
            spent[active] += step
            slack[can] -= step * rate[can]

        if kind == "cluster":
            active[best_cluster] = False
            dead.append(record_members[best_cluster])
            log("deactivate", f"{sorted(record_members[best_cluster])} (prize exhausted)")
            continue

        u, v = int(src[best_edge]), int(dst[best_edge])
        cu, cv = int(comp[u]), int(comp[v])
        ru, rv = int(rec[cu]), int(rec[cv])

        # Relabel the smaller component into the larger one.
        keep, gone = (cu, cv) if len(members[cu]) >= len(members[cv]) else (cv, cu)
        moved = members.pop(gone)
        comp[np.asarray(moved, dtype=np.int64)] = keep
        members[keep].extend(moved)

        # The two merged records stop being active, but they are NOT added to
        # `dead`: they were absorbed into something larger, they did not run out
        # of prize. Only the second kind may be pruned, so conflating them would
        # let GW pruning delete the whole answer. Without this the old records
        # also keep accruing `spent` and deactivate spuriously later on.
        active[ru] = False
        active[rv] = False

        new_rec = int(rec.shape[0])
        rec = np.append(rec, 0)  # placeholder, overwritten by rec[keep] below
        rec[keep] = new_rec
        active = np.append(active, True)
        spent = np.append(spent, spent[ru] + spent[rv])
        worth = np.append(worth, worth[ru] + worth[rv])
        record_members[new_rec] = record_members[ru] | record_members[rv]

        forest.append(int(best_edge))
        alive[best_edge] = False
        log(
            "merge",
            f"edge {u}-{v} -> {sorted(record_members[new_rec])} (moat reset to 0)",
        )

    # The maximal clusters left standing when growth ended, one per component.
    final = sorted(
        {record_members[int(rec[int(comp[v])])] for v in range(n)}, key=lambda f: -len(f)
    )
    return Growth(forest=forest, dead=dead, events=events, final=final)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def _adjacency(edge_ends: np.ndarray, kept: Sequence[int]) -> Dict[int, List[tuple]]:
    adj: Dict[int, List[tuple]] = {}
    for e in kept:
        u, v = int(edge_ends[e][0]), int(edge_ends[e][1])
        adj.setdefault(u, []).append((v, int(e)))
        adj.setdefault(v, []).append((u, int(e)))
    return adj


def _components(adj: Dict[int, List[tuple]], nodes: Sequence[int]) -> List[List[int]]:
    seen: Set[int] = set()
    out: List[List[int]] = []
    for start in nodes:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt, _ in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(group)
    return out


def _postorder(adj: Dict[int, List[tuple]], root: int, parent: int = -1) -> List[tuple]:
    """(node, parent, edge to parent) in post-order. Iterative, so a chain-shaped
    graph cannot blow the recursion limit.

    `parent` matters: passing it walks only the subtree hanging below `root`,
    which is what collecting a branch to delete needs. Leaving it at -1 walks
    the whole component.
    """
    order: List[tuple] = []
    stack = [(root, parent, -1, False)]
    while stack:
        node, par, pedge, done = stack.pop()
        if done:
            order.append((node, par, pedge))
            continue
        stack.append((node, par, pedge, True))
        for nxt, e in adj.get(node, ()):
            if nxt != par:
                stack.append((nxt, node, e, False))
    return order


def _prune_gw(
    edge_ends: np.ndarray,
    prizes: np.ndarray,
    costs: np.ndarray,
    growth: Growth,
    all_nodes: int,
) -> Tuple[List[int], List[int]]:
    """GW's own pruning rule (GW95).

    Delete a branch of the growth forest when everything hanging off it lies
    inside a cluster that ran out of prize during growth. Those are the clusters
    the growth stage gave up on, so the edges reaching into them were bought
    only to satisfy connectivity and are not paid for by what they reach.

    The final cluster is excluded: by the end it contains everything, so
    allowing it would prune the whole answer away.
    """
    adj = _adjacency(edge_ends, growth.forest)
    nodes = sorted(adj) or list(range(min(all_nodes, 1)))

    # Prunable only if the cluster was later absorbed into a bigger one. A
    # cluster that was still maximal when growth ended is a component root: it
    # ran out of prize like every other cluster does, but there is nothing left
    # to prune it in favour of.
    dead_sets = [
        d for d in growth.dead if any(d < f for f in growth.final)
    ]
    # For each node, which dead clusters contain it. Laminar, so this is small.
    holds: Dict[int, Set[int]] = {}
    for idx, group in enumerate(dead_sets):
        for v in group:
            holds.setdefault(v, set()).add(idx)

    dropped_edges: Set[int] = set()
    dropped_nodes: Set[int] = set()

    for group in _components(adj, nodes):
        root = group[0]
        cand: Dict[int, Set[int]] = {}
        children: Dict[int, List[tuple]] = {}
        for node, parent, pedge in _postorder(adj, root):
            surviving = [
                (c, e) for (c, e) in children.get(node, []) if e not in dropped_edges
            ]
            here = set(holds.get(node, ()))
            for c, _ in surviving:
                here &= cand[c]
            cand[node] = here
            if parent != -1:
                children.setdefault(parent, []).append((node, pedge))
                # A branch is prunable exactly when one dead cluster contains
                # all of it. Decided here, before the parent is processed, so a
                # cascade of prunings resolves in this single pass.
                if here:
                    dropped_edges.add(pedge)
                    # Only the branch below `node`; passing the parent stops the
                    # walk from climbing back into the part we are keeping.
                    for n2, _, _ in _postorder(adj, node, parent):
                        dropped_nodes.add(n2)

    kept_edges = [e for e in growth.forest if e not in dropped_edges]
    kept_adj = _adjacency(edge_ends, kept_edges)
    survivors = [v for v in sorted(kept_adj) if v not in dropped_nodes]
    return _best_component(edge_ends, prizes, costs, kept_edges, survivors)


def _prune_strong(
    edge_ends: np.ndarray,
    prizes: np.ndarray,
    costs: np.ndarray,
    growth: Growth,
    all_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Strong pruning (Johnson, Minkoff & Phillips 2000): exact PCST on the tree.

    On a tree the best connected subgraph can be found by one dynamic program.
    For each node,

        value(v) = prize(v) + sum over children c of max(0, value(c) - cost(v,c))

    is the best a connected subtree rooted at v and lying inside v's subtree can
    do. Every connected subtree has a unique node closest to the root, so the
    global best is the largest value(v), and the tree achieving it is recovered
    by descending from that node and keeping a child only when it pays for its
    edge. Linear time, and exact on the tree the growth stage produced.
    """
    adj = _adjacency(edge_ends, growth.forest)
    if not adj:
        return _best_component(edge_ends, prizes, costs, [], [])

    best_value = -np.inf
    best_root = -1
    best_keep: Dict[int, List[tuple]] = {}

    for group in _components(adj, sorted(adj)):
        order = _postorder(adj, group[0])
        value: Dict[int, float] = {}
        keep: Dict[int, List[tuple]] = {}
        for node, parent, _ in order:
            total = float(prizes[node])
            kept: List[tuple] = []
            for nxt, e in adj.get(node, ()):
                if nxt == parent:
                    continue
                gain = value[nxt] - float(costs[e])
                if gain > EPS:
                    total += gain
                    kept.append((nxt, e))
            value[node] = total
            keep[node] = kept
            if total > best_value:
                best_value, best_root, best_keep = total, node, keep

    # An isolated node may beat every subtree.
    lone = int(np.argmax(prizes)) if prizes.size else 0
    if prizes.size and float(prizes[lone]) > best_value:
        return [lone], []

    chosen_nodes: List[int] = []
    chosen_edges: List[int] = []
    stack = [best_root]
    while stack:
        node = stack.pop()
        chosen_nodes.append(node)
        for nxt, e in best_keep.get(node, ()):
            chosen_edges.append(e)
            stack.append(nxt)
    return sorted(chosen_nodes), sorted(chosen_edges)


def _best_component(
    edge_ends: np.ndarray,
    prizes: np.ndarray,
    costs: np.ndarray,
    kept_edges: Sequence[int],
    survivors: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Return the single highest-value connected piece, as num_clusters=1 asks.

    "Value" is prizes collected minus edges paid for, not prizes alone: a piece
    that collects 20 through 25 of edges is worse than a lone node worth 10.
    """
    adj = _adjacency(edge_ends, kept_edges)
    groups = _components(adj, [v for v in survivors if v in adj])
    lone = int(np.argmax(prizes)) if prizes.size else 0
    best_nodes: List[int] = [lone]
    best_edges: List[int] = []
    best = float(prizes[lone]) if prizes.size else 0.0

    for group in groups:
        members = set(group)
        inside = [
            e
            for e in kept_edges
            if int(edge_ends[e][0]) in members and int(edge_ends[e][1]) in members
        ]
        value = float(prizes[list(group)].sum())
        if inside:
            value -= float(costs[np.asarray(inside, dtype=np.int64)].sum())
        if value > best:
            best, best_nodes, best_edges = value, sorted(group), sorted(inside)
    return best_nodes, best_edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def solve(
    edges,
    prizes,
    costs,
    root: int = -1,
    num_clusters: int = 1,
    pruning: str = "gw",
    verbosity: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve PCST. Signature-compatible with `pcst_fast.pcst_fast`.

    Args:
        edges: [m, 2] endpoint pairs.
        prizes: [n] non-negative node prizes.
        costs: [m] non-negative edge costs.
        root: must be -1. The rooted variant assumes a node the tree has to
            contain, which is exactly the thing retrieval does not know.
        num_clusters: must be 1.
        pruning: "none", "gw" (the published behaviour) or "strong"
            (Johnson-Minkoff-Phillips, provably never worse).
        verbosity: >0 prints the growth stage event by event.

    Returns:
        (vertices, edges) as index arrays into the inputs.
    """
    edge_ends = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    prize_vec = np.asarray(prizes, dtype=np.float64).reshape(-1)
    cost_vec = np.asarray(costs, dtype=np.float64).reshape(-1)

    if root != -1:
        raise ValueError(f"only the unrooted problem is implemented, got root={root}")
    if num_clusters != 1:
        raise ValueError(f"only num_clusters=1 is implemented, got {num_clusters}")
    if pruning not in PRUNING_MODES:
        raise ValueError(f"pruning must be one of {PRUNING_MODES}, got {pruning!r}")
    if edge_ends.shape[0] != cost_vec.shape[0]:
        raise ValueError(
            f"{edge_ends.shape[0]} edges but {cost_vec.shape[0]} costs"
        )
    if prize_vec.size and prize_vec.min() < -EPS:
        raise ValueError("prizes must be non-negative")
    if cost_vec.size and cost_vec.min() < -EPS:
        raise ValueError("costs must be non-negative")

    n = int(prize_vec.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if edge_ends.shape[0] == 0:
        return np.array([int(np.argmax(prize_vec))], dtype=np.int64), np.zeros(0, dtype=np.int64)

    growth = _growth(edge_ends, prize_vec, cost_vec, verbosity=verbosity)

    if pruning == "none":
        nodes = sorted({int(v) for e in growth.forest for v in edge_ends[e]})
        return np.asarray(nodes, dtype=np.int64), np.asarray(sorted(growth.forest), dtype=np.int64)
    if pruning == "strong":
        keep_n, keep_e = _prune_strong(edge_ends, prize_vec, cost_vec, growth, n)
    else:
        keep_n, keep_e = _prune_gw(edge_ends, prize_vec, cost_vec, growth, n)
    return np.asarray(keep_n, dtype=np.int64), np.asarray(keep_e, dtype=np.int64)


def objective(
    edge_ends,
    prizes,
    costs,
    nodes: Sequence[int],
    edges_used: Sequence[int],
) -> float:
    """c(T) + pi(complement of T): the quantity section 4 of Bienstock et al.
    minimises, and the only fair way to compare two solvers' answers."""
    prize_vec = np.asarray(prizes, dtype=np.float64).reshape(-1)
    cost_vec = np.asarray(costs, dtype=np.float64).reshape(-1)
    chosen = set(int(v) for v in nodes)
    bought = float(sum(cost_vec[int(e)] for e in edges_used))
    forfeited = float(sum(p for i, p in enumerate(prize_vec) if i not in chosen))
    return bought + forfeited
