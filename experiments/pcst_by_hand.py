"""
Trace the Goemans-Williamson algorithm on a four-node graph, by hand and by solver.

Run this while reading `docs/pcst_from_the_papers.md`. It prints the event
sequence of the moat-growing stage, then calls the real `pcst_fast` on the same
instance so you can check that the trace you followed and the solver the thesis
actually uses land on the same tree.

The instance is chosen so that the two phenomena that matter both appear:

      a(10) --4-- b(0) --4-- c(10) --5-- d(1)

  * `b` has prize 0 and is still in the answer, because it lies between two
    nodes worth 10 each. This is a Steiner node, and it is the thing no top-k
    scorer can ever return.
  * `d` has prize 1 and is not in the answer, because reaching it costs 5.

The optimum is {a, b, c}: cost 8, forgone prize 1, objective 9. Enumerate the
alternatives yourself before running this, then check.

    python experiments/pcst_by_hand.py

Everything here is a teaching aid. The retriever uses `ikgqa.pcst.core`.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np

# --- the instance -----------------------------------------------------------

NAMES = ("a", "b", "c", "d")
PRIZES = np.array([10.0, 0.0, 10.0, 1.0])
EDGES = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
COSTS = np.array([4.0, 4.0, 5.0])

# A six-node version with a cycle, for the whiteboard. Same two lessons as the
# four-node one (a prize-0 bridge kept, a prized leaf pruned) plus two more: a
# cycle the solver must choose an edge out of, and a well-spaced event schedule
# so no two events collide except harmlessly at t=6.
#
#              A(10) --5-- B(0) --3-- C(8) --8-- D(2)
#                |                      |
#                +--------12------------+--2-- E(6) --6-- F(0)
#                                     (A-E is the shortcut)
EXAMPLES = {
    "four": dict(
        names=("a", "b", "c", "d"),
        prizes=[10.0, 0.0, 10.0, 1.0],
        edges=[[0, 1], [1, 2], [2, 3]],
        costs=[4.0, 4.0, 5.0],
    ),
    "six": dict(
        names=("A", "B", "C", "D", "E", "F"),
        prizes=[10.0, 0.0, 8.0, 2.0, 6.0, 0.0],
        edges=[[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4]],
        costs=[5.0, 3.0, 8.0, 2.0, 6.0, 12.0],
    ),
    # "six" plus two cheap ways into D: E-D at 2 and F-D at 1. D's prize is 2,
    # so reaching it via E-D is exactly break-even and the objective cannot
    # tell the two answers apart. That tie is the point of this variant.
    "six-b": dict(
        names=("A", "B", "C", "D", "E", "F"),
        prizes=[10.0, 0.0, 8.0, 2.0, 6.0, 0.0],
        edges=[[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4], [5, 3], [4, 3]],
        costs=[5.0, 3.0, 8.0, 2.0, 6.0, 12.0, 1.0, 2.0],
    ),
    # "six-b" with A-E dropped from 12 to 5 and E-D raised from 2 to 4.
    # Two things flip. D stops paying for itself (prize 2, now costs 4 to
    # reach), and A gets a direct route in, so B stops being the only bridge
    # and leaves the answer. A Steiner node is only a Steiner node while it is
    # the cheapest way through.
    "six-c": dict(
        names=("A", "B", "C", "D", "E", "F"),
        prizes=[10.0, 0.0, 8.0, 2.0, 6.0, 0.0],
        edges=[[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4], [5, 3], [4, 3]],
        costs=[5.0, 3.0, 8.0, 2.0, 6.0, 5.0, 1.0, 4.0],
    ),
}


def select_example(name: str) -> None:
    """Rebind the module-level instance. Everything below reads these globals."""
    global NAMES, PRIZES, EDGES, COSTS
    spec = EXAMPLES[name]
    NAMES = spec["names"]
    PRIZES = np.array(spec["prizes"])
    EDGES = np.array(spec["edges"], dtype=np.int64)
    COSTS = np.array(spec["costs"])


def objective(nodes: frozenset, edges: tuple) -> float:
    """The PCST objective as the literature states it: c(T) + pi(complement of T).

    Cost of the edges you buy, plus the prizes of every node you left behind.
    Lower is better. This is the quantity every one of the five papers
    minimises, and `pcst_fast` minimises it too.
    """
    return COSTS[list(edges)].sum() + PRIZES[[i for i in range(len(PRIZES)) if i not in nodes]].sum()


def brute_force() -> list:
    """Every connected subtree, scored. Small enough to enumerate exhaustively."""
    results = []
    for r in range(len(EDGES) + 1):
        for chosen in itertools.combinations(range(len(EDGES)), r):
            nodes = frozenset(int(v) for e in chosen for v in EDGES[e])
            if r == 0:
                for single in [frozenset()] + [frozenset({v}) for v in range(len(PRIZES))]:
                    results.append((objective(single, ()), single, ()))
                continue
            if not _connected(nodes, chosen):
                continue
            results.append((objective(nodes, chosen), nodes, chosen))
    return sorted(results, key=lambda row: row[0])


def _connected(nodes: frozenset, chosen: tuple) -> bool:
    if not nodes:
        return True
    adj: dict = {v: set() for v in nodes}
    for e in chosen:
        u, v = (int(x) for x in EDGES[e])
        adj[u].add(v)
        adj[v].add(u)
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == set(nodes)


# --- the growth stage, simulated event by event -----------------------------


@dataclasses.dataclass
class Cluster:
    """One element of the laminar family (Hegde et al., Definition 2)."""

    members: frozenset
    active: bool = True
    moat: float = 0.0  # y_C, this cluster's own dual variable
    spent: float = 0.0  # sum of y over this cluster and everything merged into it


def growth_stage(verbose: bool = True) -> list:
    """The GW growth stage: grow every active moat at unit rate, stop at events.

    Two invariants hold throughout, and they are the two dual constraints:

      edge constraint     sum of moats separating the endpoints of e <= c(e)
      cluster constraint  sum of moats inside C                     <= prize(C)

    Two events can make one tight. A tight edge merges its two clusters; a tight
    cluster deactivates. The loop ends when nothing is active.
    """
    clusters = [Cluster(frozenset({i})) for i in range(len(PRIZES))]
    owner = {i: i for i in range(len(PRIZES))}  # node -> index of its maximal cluster
    slack = {e: float(COSTS[e]) for e in range(len(EDGES))}
    forest: list = []
    time = 0.0
    log: list = []

    def note(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    # A zero-prize node has a tight cluster constraint before the clock starts,
    # so it is never active on its own. This is why `b` never grows a moat.
    for c in clusters:
        if c.active and c.spent >= PRIZES[list(c.members)].sum():
            c.active = False
            note(f"t=0.000  deactivate {_show(c)} -- prize 0, cluster constraint tight already")

    while any(c.active for c in clusters):
        # How long until each possible event? Active endpoints per edge set the
        # rate: two active endpoints burn slack twice as fast as one.
        best, kind, target = float("inf"), "", None
        for e, s in slack.items():
            u, v = (int(x) for x in EDGES[e])
            cu, cv = owner[u], owner[v]
            if cu == cv:
                continue
            rate = int(clusters[cu].active) + int(clusters[cv].active)
            if rate and s / rate < best:
                best, kind, target = s / rate, "edge", e
        for i, c in enumerate(clusters):
            if not c.active:
                continue
            room = PRIZES[list(c.members)].sum() - c.spent
            if room < best:
                best, kind, target = room, "cluster", i

        if kind == "":
            break

        # Advance the clock and charge every active moat for the elapsed time.
        time += best
        for c in clusters:
            if c.active:
                c.moat += best
                c.spent += best
        for e in list(slack):
            u, v = (int(x) for x in EDGES[e])
            rate = int(clusters[owner[u]].active) + int(clusters[owner[v]].active)
            if owner[u] != owner[v]:
                slack[e] -= best * rate

        if kind == "cluster":
            clusters[target].active = False
            note(f"t={time:.3f}  deactivate {_show(clusters[target])} -- cluster constraint tight")
        else:
            u, v = (int(x) for x in EDGES[target])
            cu, cv = owner[u], owner[v]
            merged = Cluster(
                members=clusters[cu].members | clusters[cv].members,
                active=True,
                moat=0.0,
                spent=clusters[cu].spent + clusters[cv].spent,
            )
            clusters[cu].active = clusters[cv].active = False
            clusters.append(merged)
            for node in merged.members:
                owner[node] = len(clusters) - 1
            forest.append(target)
            del slack[target]
            for e in list(slack):  # drop edges now inside the merged cluster
                a, b = (int(x) for x in EDGES[e])
                if owner[a] == owner[b]:
                    del slack[e]
            note(
                f"t={time:.3f}  edge {NAMES[u]}-{NAMES[v]} tight -> merge into "
                f"{_show(merged)}, new moat 0"
            )

    note(f"growth ends: forest F = {[_edge_name(e) for e in forest]}")
    note("  note that F spans every node, including d. Pruning is what removes d.")
    return forest


def _show(c: Cluster) -> str:
    return "{" + ",".join(NAMES[i] for i in sorted(c.members)) + "}"


def _edge_name(e: int) -> str:
    u, v = (int(x) for x in EDGES[e])
    return f"{NAMES[u]}-{NAMES[v]}"


# --- what the real solver says ----------------------------------------------


def solver() -> tuple:
    from pcst_fast import pcst_fast

    return pcst_fast(EDGES, PRIZES, COSTS, -1, 1, "gw", 0)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="pcst_by_hand.py")
    ap.add_argument(
        "--example", choices=sorted(EXAMPLES), default="four",
        help="'four' is the minimal teaching case; 'six' adds a cycle (whiteboard version)",
    )
    args = ap.parse_args()
    select_example(args.example)

    print(f"=== example: {args.example} nodes ===")
    print("prizes  " + "  ".join(f"{n}={p:g}" for n, p in zip(NAMES, PRIZES)))
    print("edges   " + "  ".join(
        f"{NAMES[int(a)]}-{NAMES[int(b)]}:{c:g}" for (a, b), c in zip(EDGES, COSTS)
    ))
    print(f"total prize PI = {PRIZES.sum():g}")
    print("\n--- every connected subtree, scored (lower is better) ---")
    for value, nodes, chosen in brute_force()[:5]:
        label = "{" + ",".join(NAMES[i] for i in sorted(nodes)) + "}"
        print(f"  {value:6.2f}  nodes {label:12s} edges {[_edge_name(e) for e in chosen]}")

    print("\n--- growth stage, event by event ---")
    growth_stage()

    print("\n--- pcst_fast, unrooted, one cluster, gw pruning ---")
    vertices, edges = solver()
    got = frozenset(int(v) for v in vertices)
    print(f"  nodes  {'{' + ','.join(NAMES[i] for i in sorted(got)) + '}'}")
    print(f"  edges  {[_edge_name(int(e)) for e in edges]}")
    print(f"  objective {objective(got, tuple(int(e) for e in edges)):.2f}")

    all_subtrees = brute_force()
    best = all_subtrees[0]
    print(f"  F(S) = PI - objective = {PRIZES.sum():g} - {best[0]:g} = {PRIZES.sum() - best[0]:g}")

    # Compare objective VALUES, not node sets. Several different subtrees can
    # share the optimal objective, and calling one of them "the" optimum would
    # report a tie as a failure.
    tied = sorted(
        {tuple(sorted(n)) for value, n, _ in all_subtrees if abs(value - best[0]) < 1e-9}
    )
    got_value = objective(got, tuple(int(e) for e in solver()[1]))
    if abs(got_value - best[0]) < 1e-9:
        print(f"\n  solver reached the optimal objective {best[0]:g}.")
    else:
        print(f"\n  solver scored {got_value:g} against the optimum {best[0]:g}.")
    if len(tied) > 1:
        print(f"  {len(tied)} different subtrees share that objective:")
        for nodes in tied:
            print("      {" + ",".join(NAMES[i] for i in nodes) + "}")
        print("  Which one comes back is the solver's choice, not the objective's.")
    # What the factor-2 guarantee actually permits, beside what happened. The
    # Lagrangian-preserving form (GW95) is
    #     c(T) + 2*pi(complement of T)  <=  2*c(OPT) + 2*pi(complement of OPT)
    # which implies the plain statement below, since penalties are non-negative.
    print(f"\n  the factor-2 guarantee permits any objective up to 2 x {best[0]:g} = {2 * best[0]:g}")
    print(f"  the solver returned                                    {got_value:g}")
    print(f"  so it used {got_value / best[0]:.2f} of its allowance -- the guarantee never bound here.")
    print("  On a 15,810-node patient the optimum is unknowable, and this bound")
    print("  is then the only thing standing between the answer and the unknown best.")


if __name__ == "__main__":
    main()
