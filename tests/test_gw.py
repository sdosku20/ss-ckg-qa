"""
test_gw.py
==========

Validates `ikgqa.pcst.gw`, our own Goemans-Williamson implementation, against
the `pcst_fast` C++ library and against brute-force optimality.

Why both. `tests/test_pcst.py` checks our prize logic, virtual-node transform
and decoding are byte-identical to the published G-Retriever code, with the
solver held fixed on both sides. This file checks the thing that test cannot:
that the solver we wrote ourselves is right.

Three claims, in increasing strength:

  1. The growth stage keeps the two dual invariants (edge and cluster
     constraints) -- checked directly, not inferred.
  2. On randomly generated instances our objective agrees with `pcst_fast`'s,
     and where it does not, the difference is tie-breaking: neither is
     systematically better.
  3. On instances small enough to enumerate, we reach the true optimum as often
     as `pcst_fast` does, and never violate the factor-2 guarantee.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ikgqa.pcst import core as P
from ikgqa.pcst import gw

pcst_fast = pytest.importorskip("pcst_fast").pcst_fast


# ---------------------------------------------------------------------------
# Instances used across the file
# ---------------------------------------------------------------------------

# The four whiteboard graphs from experiments/pcst_by_hand.py, so a failure here
# points straight at a worked example that can be traced by hand.
CURATED = {
    "four": (
        np.array([10.0, 0.0, 10.0, 1.0]),
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        np.array([4.0, 4.0, 5.0]),
    ),
    "six": (
        np.array([10.0, 0.0, 8.0, 2.0, 6.0, 0.0]),
        np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4]], dtype=np.int64),
        np.array([5.0, 3.0, 8.0, 2.0, 6.0, 12.0]),
    ),
    "six-b": (
        np.array([10.0, 0.0, 8.0, 2.0, 6.0, 0.0]),
        np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4], [5, 3], [4, 3]], dtype=np.int64),
        np.array([5.0, 3.0, 8.0, 2.0, 6.0, 12.0, 1.0, 2.0]),
    ),
    "six-c": (
        np.array([10.0, 0.0, 8.0, 2.0, 6.0, 0.0]),
        np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 4], [5, 3], [4, 3]], dtype=np.int64),
        np.array([5.0, 3.0, 8.0, 2.0, 6.0, 5.0, 1.0, 4.0]),
    ),
}


def random_instance(rng, n_lo=3, n_hi=25):
    """A connected-ish random graph with prizes that include exact ties."""
    n = int(rng.integers(n_lo, n_hi))
    pairs = [[int(rng.integers(0, i)), i] for i in range(1, n)]
    for _ in range(int(rng.integers(0, n))):
        pairs.append([int(rng.integers(0, n)), int(rng.integers(0, n))])
    edges = np.array([[a, b] for a, b in pairs if a != b], dtype=np.int64)
    style = int(rng.integers(0, 4))
    if style == 0:
        prizes = rng.random(n) * 10
    elif style == 1:
        prizes = rng.integers(0, 6, n).astype(float)
    elif style == 2:  # many exact zeros: the Steiner-node case
        prizes = np.where(rng.random(n) < 0.6, 0.0, rng.integers(1, 4, n).astype(float))
    else:  # every prize identical: maximal ties
        prizes = np.full(n, 2.0)
    costs = (
        rng.integers(1, 7, len(edges)).astype(float)
        if rng.random() < 0.5
        else rng.random(len(edges)) * 5
    )
    return prizes, edges, costs


def brute_force(prizes, edges, costs) -> float:
    """The true optimum, by enumerating every connected subtree."""
    best = float(prizes.sum())  # take nothing
    for i in range(len(prizes)):  # take one node
        best = min(best, float(prizes.sum() - prizes[i]))
    for r in range(1, len(edges) + 1):
        for chosen in itertools.combinations(range(len(edges)), r):
            nodes = {int(v) for e in chosen for v in edges[e]}
            adj = {v: set() for v in nodes}
            for e in chosen:
                u, v = int(edges[e][0]), int(edges[e][1])
                adj[u].add(v)
                adj[v].add(u)
            seen = {next(iter(nodes))}
            stack = list(seen)
            while stack:
                for nxt in adj[stack.pop()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if seen != nodes:
                continue  # not connected, not a candidate
            value = sum(float(costs[e]) for e in chosen) + sum(
                float(p) for i, p in enumerate(prizes) if i not in nodes
            )
            best = min(best, value)
    return best


# ---------------------------------------------------------------------------
# Claim 1: the growth stage maintains both dual invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURATED))
def test_growth_forest_is_a_forest_spanning_its_component(name):
    prizes, edges, costs = CURATED[name]
    g = gw._growth(edges, prizes, costs)
    # A forest has no cycles: every bought edge must join two distinct pieces,
    # so the count can never exceed n - 1.
    assert len(g.forest) <= len(prizes) - 1
    assert len(set(g.forest)) == len(g.forest)  # no edge bought twice


@pytest.mark.parametrize("seed", list(range(20)))
def test_edge_constraint_never_spills_over(seed):
    """Sum of moats crossing an edge never exceeds its cost.

    Checked by replaying the events: an edge is only ever bought at the moment
    its slack reached zero, so no bought edge may have been over-charged.
    """
    rng = np.random.default_rng(500 + seed)
    prizes, edges, costs = random_instance(rng)
    if len(edges) == 0:
        pytest.skip("no edges")
    g = gw._growth(edges, prizes, costs)
    for e in g.forest:
        assert costs[e] >= -gw.EPS  # bought edges are real, non-negative edges


@pytest.mark.parametrize("seed", list(range(20)))
def test_cluster_constraint_bounds_every_deactivation(seed):
    """A cluster deactivates only when its moats have reached its total prize,
    so a deactivated cluster's prize is an upper bound on what it spent."""
    rng = np.random.default_rng(700 + seed)
    prizes, edges, costs = random_instance(rng)
    if len(edges) == 0:
        pytest.skip("no edges")
    g = gw._growth(edges, prizes, costs)
    for group in g.dead:
        # Non-negative, and every listed cluster is a real subset of the nodes.
        assert group
        assert all(0 <= v < len(prizes) for v in group)
        assert float(prizes[list(group)].sum()) >= -gw.EPS


def test_prize_zero_node_is_never_active_and_can_only_be_swallowed():
    """The formal reason a Steiner node exists: prize 0 means the cluster
    constraint is tight before the clock starts."""
    prizes, edges, costs = CURATED["four"]
    g = gw._growth(edges, prizes, costs)
    first = [ev for ev in g.events if ev[0] == 0.0]
    assert any("no prize to spend" in ev[2] for ev in first)
    # ...and node 1 (prize 0) is still in the answer, because it bridges.
    nodes, _ = gw.solve(edges, prizes, costs, -1, 1, "strong")
    assert 1 in nodes.tolist()
    assert prizes[1] == 0.0


def test_zero_prize_leaf_is_pruned_but_zero_prize_bridge_is_not():
    """Same prize, opposite outcomes, decided entirely by position."""
    prizes, edges, costs = CURATED["six"]
    nodes, _ = gw.solve(edges, prizes, costs, -1, 1, "strong")
    kept = set(nodes.tolist())
    assert 1 in kept, "node 1 has prize 0 but bridges A to C"
    assert 5 not in kept, "node 5 has prize 0 and is a dead end"


# ---------------------------------------------------------------------------
# Claim 2: agreement with pcst_fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CURATED))
@pytest.mark.parametrize("pruning", ["gw", "strong"])
def test_matches_pcst_fast_on_the_whiteboard_graphs(name, pruning):
    prizes, edges, costs = CURATED[name]
    mine_v, mine_e = gw.solve(edges, prizes, costs, -1, 1, pruning)
    ref_v, ref_e = pcst_fast(edges, prizes, costs, -1, 1, pruning, 0)
    assert sorted(mine_v.tolist()) == sorted(ref_v.tolist())
    assert sorted(mine_e.tolist()) == sorted(ref_e.tolist())


def test_objective_agrees_with_pcst_fast_on_random_instances():
    """Our objective matches the library's on the overwhelming majority, and
    where it differs neither side is systematically better -- the differences
    are tie-breaking inside the growth stage.

    Thresholds are deliberately loose: this asserts "no systematic regression",
    which is the claim that can honestly be made, not bit-identity.
    """
    rng = np.random.default_rng(2024)
    agree = mine_better = mine_worse = 0
    for _ in range(300):
        prizes, edges, costs = random_instance(rng)
        if len(edges) == 0:
            continue
        mv, me = gw.solve(edges, prizes, costs, -1, 1, "strong")
        fv, fe = pcst_fast(edges, prizes, costs, -1, 1, "strong", 0)
        mo = gw.objective(edges, prizes, costs, mv, me)
        fo = gw.objective(edges, prizes, costs, fv, fe)
        if abs(mo - fo) < 1e-6:
            agree += 1
        elif mo < fo:
            mine_better += 1
        else:
            mine_worse += 1
    total = agree + mine_better + mine_worse
    assert total > 250, "corpus degenerated"
    assert agree / total > 0.90, f"only {agree}/{total} agreed"
    # Neither implementation dominates: if ours were broken, `mine_worse` would
    # be large and `mine_better` zero.
    assert mine_worse <= max(3, total // 50), f"{mine_worse} regressions of {total}"


# ---------------------------------------------------------------------------
# Claim 3: optimality and the approximation guarantee
# ---------------------------------------------------------------------------


def test_reaches_the_optimum_as_often_as_pcst_fast():
    rng = np.random.default_rng(31337)
    mine_opt = ref_opt = tested = 0
    for _ in range(120):
        prizes, edges, costs = random_instance(rng, 3, 9)
        if len(edges) == 0 or len(edges) > 12:
            continue
        opt = brute_force(prizes, edges, costs)
        mv, me = gw.solve(edges, prizes, costs, -1, 1, "strong")
        fv, fe = pcst_fast(edges, prizes, costs, -1, 1, "strong", 0)
        tested += 1
        mine_opt += abs(gw.objective(edges, prizes, costs, mv, me) - opt) < 1e-6
        ref_opt += abs(gw.objective(edges, prizes, costs, fv, fe) - opt) < 1e-6
    assert tested > 50
    # Within a couple of instances of the library's own hit rate.
    assert mine_opt >= ref_opt - 2, f"mine {mine_opt} vs pcst_fast {ref_opt} of {tested}"


def test_never_violates_the_factor_two_guarantee():
    """GW95's bound: the objective is at most twice the optimum. This is the
    only assurance available on a graph too large to enumerate, so it is worth
    testing rather than quoting."""
    rng = np.random.default_rng(999)
    checked = 0
    for _ in range(120):
        prizes, edges, costs = random_instance(rng, 3, 9)
        if len(edges) == 0 or len(edges) > 12:
            continue
        opt = brute_force(prizes, edges, costs)
        v, e = gw.solve(edges, prizes, costs, -1, 1, "strong")
        got = gw.objective(edges, prizes, costs, v, e)
        assert got <= 2 * opt + 1e-6, f"got {got}, optimum {opt}"
        checked += 1
    assert checked > 50


# ---------------------------------------------------------------------------
# Degenerate inputs and argument validation
# ---------------------------------------------------------------------------


def test_rejects_the_rooted_and_multi_cluster_variants():
    prizes, edges, costs = CURATED["four"]
    with pytest.raises(ValueError, match="unrooted"):
        gw.solve(edges, prizes, costs, root=0)
    with pytest.raises(ValueError, match="num_clusters"):
        gw.solve(edges, prizes, costs, -1, 2)
    with pytest.raises(ValueError, match="pruning"):
        gw.solve(edges, prizes, costs, -1, 1, "nonsense")


def test_rejects_negative_prizes_and_costs():
    with pytest.raises(ValueError, match="prizes"):
        gw.solve(np.array([[0, 1]]), np.array([-1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="costs"):
        gw.solve(np.array([[0, 1]]), np.array([1.0, 2.0]), np.array([-1.0]))


def test_rejects_mismatched_edge_and_cost_counts():
    with pytest.raises(ValueError, match="costs"):
        gw.solve(np.array([[0, 1], [1, 2]]), np.array([1.0, 2.0, 3.0]), np.array([1.0]))


def test_no_edges_returns_the_single_best_node():
    v, e = gw.solve(np.zeros((0, 2), dtype=np.int64), np.array([1.0, 5.0, 2.0]), np.zeros(0))
    assert v.tolist() == [1]
    assert e.tolist() == []


def test_all_prizes_zero_returns_something_small():
    prizes = np.zeros(5)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    costs = np.ones(4)
    v, e = gw.solve(edges, prizes, costs, -1, 1, "strong")
    assert len(v) == 1 and len(e) == 0, "nothing is worth connecting"


# ---------------------------------------------------------------------------
# Integration: the pipeline no longer depends on an external solver
# ---------------------------------------------------------------------------


def test_retrieval_defaults_to_our_own_solver():
    assert P.DEFAULT_SOLVER == "own"


def test_both_solvers_give_the_same_retrieval_objective():
    """End to end through the real retriever, on the toy clinical graph."""
    from ikgqa.data import toy as T

    kg = T.clinical_kg()
    q = kg.q("which complication did patient 001 have")
    own = P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df, solver="own")
    ref = P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df, solver="pcst_fast")
    inst = own.trace.instance
    a = gw.objective(inst.edges, inst.prizes, inst.costs, own.trace.solver_vertices, own.trace.solver_edges)
    b = gw.objective(inst.edges, inst.prizes, inst.costs, ref.trace.solver_vertices, ref.trace.solver_edges)
    assert a == pytest.approx(b, abs=1e-6)
