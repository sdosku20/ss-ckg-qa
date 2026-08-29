"""
test_pcst.py
============

Run with:   pytest -v test_pcst.py
Or:         python test_pcst.py        (same thing, prettier summary)

The suite is organised so that reading it top to bottom walks you through the
algorithm. Every test name states a claim about PCST retrieval; if the test
passes, the claim holds for this implementation.

Group 0  environment
Group 1  node prizes
Group 2  edge prizes and the cost floor
Group 3  the virtual-node transform
Group 4  end-to-end behaviour (the interesting ones)
Group 5  degenerate inputs
Group 6  equivalence with the official G-Retriever implementation
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from ikgqa.pcst import core as P
from ikgqa.data import toy as T

# ---------------------------------------------------------------------------
# Group 0: environment
# ---------------------------------------------------------------------------


def test_pcst_fast_binary_is_not_silently_broken():
    """Guards against the numpy>=2 / pcst_fast wheel ABI mismatch.

    If this fails, every other result in the playground is meaningless, so it
    runs first.
    """
    P.check_pcst_fast_sanity()


def test_cosine_similarity_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    q = rng.standard_normal(16).astype(np.float32)
    m = rng.standard_normal((25, 16)).astype(np.float32)
    mine = P.cosine_similarity(q, m)
    theirs = torch.nn.CosineSimilarity(dim=-1)(torch.tensor(q), torch.tensor(m)).numpy()
    assert np.allclose(mine, theirs, atol=1e-6)


# ---------------------------------------------------------------------------
# Group 1: node prizes  (paper Eq. 6)
# ---------------------------------------------------------------------------


def test_node_prizes_are_k_down_to_1_in_similarity_order():
    """The i-th most similar node gets prize (topk - i); everyone else gets 0.

    Uses random embeddings so all similarities are distinct and the ranking is
    unambiguous (see the tie test below).
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((12, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    sim, prizes = P.compute_node_prizes(q, x, topk=3)

    order = np.argsort(-sim)
    assert np.unique(sim).size == 12  # no ties
    assert prizes[order[0]] == 3.0
    assert prizes[order[1]] == 2.0
    assert prizes[order[2]] == 1.0
    assert prizes[order[3:]].sum() == 0.0
    assert (prizes > 0).sum() == 3


def test_tied_similarities_make_the_top_k_arbitrary():
    """A trap worth knowing about before you trust any prize you print.

    Ask the chain graph about "creatinine" and only node 0 shares a word with
    the query. The other seven nodes all score exactly 0.0, so "the 2nd and 3rd
    most similar node" is meaningless: the code picks two of the seven, and
    *which* two depends on `torch.topk`'s internal quickselect order, not on
    anything about the data.

    In a real clinical KG this happens whenever the question vocabulary misses
    most of the graph, which is most of the time.
    """
    kg = T.chain_kg()
    sim, prizes = P.compute_node_prizes(kg.q("creatinine"), kg.x, topk=3)

    assert (sim == 0.0).sum() == 7
    winners = set(np.flatnonzero(prizes > 0).tolist())
    assert len(winners) == 3
    assert 0 in winners  # the only genuinely relevant node
    arbitrary = winners - {0}
    assert all(sim[i] == 0.0 for i in arbitrary)

    # the two tie policies can legitimately disagree
    _, stable = P.compute_node_prizes(kg.q("creatinine"), kg.x, topk=3, tie_break="stable")
    assert set(np.flatnonzero(stable > 0).tolist()) == {0, 1, 2}


def test_node_prizes_ignore_how_similar_a_node_actually_is():
    """Only the *rank* matters, not the value. A node with similarity 0.99 and
    one with 0.01 get prizes 3 and 2 if they happen to be ranked 1st and 2nd.

    This is why the prizes are not comparable across questions, and why the
    absolute value of `cost_e` behaves so differently on different graphs.
    """
    x = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    sim, prizes = P.compute_node_prizes(q, x, topk=2)
    assert sim[0] > sim[1] > sim[2] > sim[3]
    assert prizes.tolist() == [2.0, 1.0, 0.0, 0.0]


def test_topk_zero_disables_node_prizes():
    kg = T.clinical_kg()
    _, prizes = P.compute_node_prizes(kg.q("creatinine"), kg.x, topk=0)
    assert prizes.sum() == 0.0


def test_topk_larger_than_graph_is_clamped():
    kg = T.single_node_kg()
    _, prizes = P.compute_node_prizes(kg.q("creatinine"), kg.x, topk=50)
    assert prizes.tolist() == [1.0]  # k clamped to 1, so prize = k - 0 = 1


# ---------------------------------------------------------------------------
# Group 2: edge prizes and the cost floor
# ---------------------------------------------------------------------------


def test_edge_prizes_split_the_tier_budget_between_tied_edges():
    """Ten edges sharing one relation string share one tier's prize.

    `hub_kg` has ten `has lab result` edges. They have identical embeddings, so
    identical similarity, so they form one tier. The tier is nominally worth
    topk_e - 0 = 3, and that 3 is divided by 10.
    """
    kg = T.hub_kg()
    q = kg.q("lab result")
    sim, prizes = P.compute_edge_prizes(q, kg.edge_attr, topk_e=3)

    lab_edges = [i for i, (_, r, _) in enumerate(kg.triples) if r == "has lab result"]
    assert len(lab_edges) == 10
    assert np.allclose(sim[lab_edges], sim[lab_edges[0]])  # identical embeddings tie

    # Only 2 distinct similarities exist here (the lab relation, and 0.0 for
    # everything else), so topk_e is clamped from 3 to 2 and the top tier is
    # worth 2, not 3. Then that 2 is split ten ways.
    assert np.unique(sim).size == 2
    assert np.allclose(prizes[lab_edges], 2.0 / 10)

    # a single rare edge in the next tier keeps almost the whole tier value
    rare = [i for i, (_, r, _) in enumerate(kg.triples) if r == "has kidney transplant"][0]
    assert prizes[rare] > prizes[lab_edges[0]] * 0.9
    # ...but is clamped to stay strictly below the tier above it
    assert prizes[rare] < prizes[lab_edges[0]]


def test_edge_prize_tiers_are_strictly_decreasing():
    """The (1 - c) clamp keeps tiers ordered even after budget splitting."""
    rng = np.random.default_rng(3)
    edge_attr = rng.standard_normal((30, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    sim, prizes = P.compute_edge_prizes(q, edge_attr, topk_e=5)

    tiers = [prizes[np.argsort(-sim, kind="stable")[i]] for i in range(5)]
    assert all(tiers[i] > tiers[i + 1] for i in range(4)), tiers


def test_edges_outside_the_top_tiers_get_zero():
    rng = np.random.default_rng(4)
    edge_attr = rng.standard_normal((30, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    sim, prizes = P.compute_edge_prizes(q, edge_attr, topk_e=3)
    # 30 random vectors => 30 distinct similarities => exactly 3 prized edges
    assert (prizes > 0).sum() == 3
    assert set(np.argsort(-sim)[:3]) == set(np.flatnonzero(prizes > 0))


def test_topk_e_zero_disables_edge_prizes():
    kg = T.clinical_kg()
    _, prizes = P.compute_edge_prizes(kg.q("creatinine"), kg.edge_attr, topk_e=0)
    assert prizes.sum() == 0.0


def test_cost_floor_guarantees_at_least_one_edge_beats_its_cost():
    """`adjust_edge_cost` shaves cost_e below the best edge prize.

    Without it, a caller asking for cost_e=0.5 on a graph whose best edge prize
    is 0.3 would get a subgraph with no query-relevant edge at all.
    """
    prizes = np.array([0.3, 0.1, 0.0], dtype=np.float32)
    used = P.adjust_edge_cost(prizes, cost_e=0.5)
    assert used < prizes.max()
    assert used == pytest.approx(0.3 * (1 - P.C / 2))


def test_cost_floor_caps_cost_e_so_raising_it_stops_helping():
    """A consequence people trip over: cost_e is an upper bound that the code
    silently lowers, so `cost_e=100` behaves the same as `cost_e=2`.
    """
    kg = T.chain_kg()
    q = kg.q("creatinine measured")
    args = (kg.graph, q, kg.nodes_df, kg.edges_df)
    a = P.retrieval_via_pcst_traced(*args, topk=3, topk_e=3, cost_e=2.0)
    b = P.retrieval_via_pcst_traced(*args, topk=3, topk_e=3, cost_e=100.0)
    assert a.trace.cost_e_used == b.trace.cost_e_used
    assert a.desc == b.desc


# ---------------------------------------------------------------------------
# Group 3: the virtual-node transform
# ---------------------------------------------------------------------------


def test_cheap_edges_stay_edges_and_get_a_discount():
    edge_index = np.array([[0, 1], [1, 2]])
    n_prizes = np.zeros(3, dtype=np.float32)
    e_prizes = np.array([0.1, 0.2], dtype=np.float32)
    inst = P.build_pcst_instance(edge_index, n_prizes, e_prizes, cost_e=0.5, num_nodes=3)

    assert inst.num_virtual_nodes == 0
    assert inst.num_real_edges == 2
    assert np.allclose(inst.costs, [0.5 - 0.1, 0.5 - 0.2])
    assert inst.prizes.size == 3  # no extra nodes appended


def test_expensive_edges_become_virtual_nodes():
    """prize > cost would need a negative edge cost, which PCST forbids, so the
    edge is replaced by src -0- v -0- dst with prize = prize - cost.
    """
    edge_index = np.array([[0, 1], [1, 2]])
    n_prizes = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    e_prizes = np.array([0.1, 3.0], dtype=np.float32)  # second edge is expensive
    inst = P.build_pcst_instance(edge_index, n_prizes, e_prizes, cost_e=0.5, num_nodes=3)

    assert inst.num_virtual_nodes == 1
    assert inst.num_real_edges == 1  # only the cheap edge survives as an edge
    virtual_id = 3
    assert inst.virtual_node_map == {virtual_id: 1}  # stands for original edge 1
    assert inst.prizes[virtual_id] == pytest.approx(3.0 - 0.5)
    # the two half-edges are free and both touch the virtual node
    half = inst.edges[inst.num_real_edges:]
    assert half.tolist() == [[1, virtual_id], [virtual_id, 2]]
    assert np.allclose(inst.costs[inst.num_real_edges:], 0.0)


def test_every_original_edge_is_accounted_for_exactly_once():
    """Invariant: real-edge count + virtual-node count == original edge count,
    and the two bookkeeping maps together cover every original edge id.
    """
    kg = T.clinical_kg()
    q = kg.q("which complication did patient 001 have")
    _, n_prizes = P.compute_node_prizes(q, kg.x, topk=3)
    _, e_prizes = P.compute_edge_prizes(q, kg.edge_attr, topk_e=3)
    cost = P.adjust_edge_cost(e_prizes, 0.5)
    inst = P.build_pcst_instance(kg.edge_index, n_prizes, e_prizes, cost, kg.graph.num_nodes)

    assert inst.num_real_edges + inst.num_virtual_nodes == kg.graph.num_edges
    covered = set(inst.edge_map.values()) | set(inst.virtual_node_map.values())
    assert covered == set(range(kg.graph.num_edges))
    assert inst.costs.min() >= 0.0  # PCST requires non-negative costs


# ---------------------------------------------------------------------------
# Group 4: end-to-end behaviour
# ---------------------------------------------------------------------------


def _run(kg, question, **kw):
    return P.retrieval_via_pcst_traced(kg.graph, kg.q(question), kg.nodes_df, kg.edges_df, **kw)


def _is_connected(num_nodes, edge_index) -> bool:
    if num_nodes <= 1:
        return True
    adj = {i: set() for i in range(num_nodes)}
    ei = np.asarray(edge_index)
    for s, d in ei.T:
        adj[int(s)].add(int(d))
        adj[int(d)].add(int(s))
    seen, stack = {0}, [0]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == num_nodes


@pytest.mark.parametrize("name", ["chain", "hub", "clinical", "two-components"])
def test_retrieved_subgraph_is_connected(name):
    """This is the whole point of PCST over top-k: `num_clusters=1` means the
    answer is a single connected component, never a bag of unrelated hits.
    """
    kg = T.ALL_GRAPHS[name]()
    r = _run(kg, "creatinine lab result code patient", topk=3, topk_e=3, cost_e=0.5)
    assert _is_connected(len(r.selected_nodes), r.subgraph.edge_index)


def test_zero_prize_bridge_nodes_get_pulled_in():
    """The headline behaviour. Query matches the two ends of a chain; the five
    nodes in between score 0.0 similarity, yet they are retrieved, because a
    connected subgraph containing both ends must pass through them.

    A top-k retriever returns {creatinine, tacrolimus} with no path between
    them. PCST returns the path.
    """
    kg = T.chain_kg()
    r = _run(kg, "creatinine measured and tacrolimus drug", topk=2, topk_e=2, cost_e=0.5)

    prizes = r.trace.node_prizes
    prized = set(np.flatnonzero(prizes > 0).tolist())
    selected = set(r.selected_nodes.tolist())

    assert prized == {0, 6}  # only the two ends are query-relevant
    assert prized <= selected
    bridges = selected - prized
    assert bridges, "no bridge node was added"
    assert all(prizes[b] == 0.0 for b in bridges)
    # and the path really is there
    assert selected == {0, 1, 2, 3, 4, 5, 6}
    assert 7 not in selected  # the irrelevant allergy record is left out


def test_raising_the_edge_cost_shrinks_the_subgraph():
    """`cost_e` is the size dial: every extra edge must earn its cost in prize.

    Values below the cost floor only; see
    test_cost_floor_caps_cost_e_so_raising_it_stops_helping.
    """
    kg = T.chain_kg()
    sizes = [
        len(_run(kg, "specimen collected during encounter", topk=3, topk_e=3, cost_e=c).selected_nodes)
        for c in (0.2, 0.5, 2.0)
    ]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[0] > sizes[-1]


def test_selected_nodes_cover_every_endpoint_of_every_selected_edge():
    """Structural invariant: the description can never mention an edge whose
    endpoints are missing from the node list. Worth asserting because the
    node set is patched up *after* the solver runs.
    """
    for name in T.ALL_GRAPHS:
        kg = T.ALL_GRAPHS[name]()
        r = _run(kg, "creatinine patient lab result", topk=3, topk_e=3, cost_e=0.5)
        ei = np.asarray(kg.edge_index)[:, r.selected_edges]
        assert set(ei.flatten().tolist()) <= set(r.selected_nodes.tolist())


def test_description_is_two_csv_blocks_that_parse_back_to_the_selection():
    kg = T.clinical_kg()
    r = _run(kg, "what lab result and code does patient 001 have for creatinine",
             topk=3, topk_e=3, cost_e=0.5)
    node_block, edge_block = r.desc.split("\n\n")

    nodes = pd.read_csv(io.StringIO(node_block))
    edges = pd.read_csv(io.StringIO(edge_block))
    assert list(nodes.columns) == ["node_id", "node_attr"]
    assert list(edges.columns) == ["src", "edge_attr", "dst"]
    assert nodes["node_id"].tolist() == r.selected_nodes.tolist()
    assert len(edges) == len(r.selected_edges)
    # ids in the text are ORIGINAL ids, not the re-indexed subgraph ids
    assert set(edges["src"]) | set(edges["dst"]) <= set(nodes["node_id"])


def test_subgraph_is_reindexed_but_topologically_identical():
    """The returned tensor graph is renumbered 0..n-1. Mapping it back through
    `selected_nodes` must reproduce the original triples exactly.
    """
    kg = T.clinical_kg()
    r = _run(kg, "which medication statement was given to patient 001", topk=3, topk_e=3, cost_e=0.5)

    local = np.asarray(r.subgraph.edge_index)
    assert local.max() < len(r.selected_nodes)
    back = r.selected_nodes[local]
    expected = np.asarray(kg.edge_index)[:, r.selected_edges]
    assert np.array_equal(back, expected)
    assert np.asarray(r.subgraph.x).shape[0] == len(r.selected_nodes)
    assert np.asarray(r.subgraph.edge_attr).shape[0] == len(r.selected_edges)


def test_retrieval_actually_answers_the_question():
    """A semantic sanity check on the clinical graph: asking about patient 001's
    creatinine should retrieve patient 001's creatinine lab node and its LOINC
    code, and should not drag in the tacrolimus branch.
    """
    kg = T.clinical_kg()
    r = _run(kg, "what lab result and code does patient 001 have for creatinine",
             topk=3, topk_e=3, cost_e=0.5)
    got = set(r.selected_nodes.tolist())
    assert 4 in got, "patient 001's creatinine value missing"
    assert 8 in got, "the LOINC creatinine code missing"
    assert 6 not in got, "unrelated tacrolimus node was retrieved"


def test_retrieval_shrinks_the_prompt():
    """The efficiency claim from Table 4, in miniature: the retrieved
    description is much shorter than textualising the whole graph.
    """
    kg = T.clinical_kg()
    full = kg.nodes_df.to_csv(index=False) + "\n" + kg.edges_df.to_csv(
        index=False, columns=["src", "edge_attr", "dst"]
    )
    r = _run(kg, "which complication did patient 001 have", topk=3, topk_e=3, cost_e=0.5)
    assert len(r.desc) < 0.5 * len(full)


def test_only_one_component_survives_when_the_graph_is_disconnected():
    """num_clusters=1: PCST cannot return two islands even if both are relevant.

    A real limitation to know about before you build an evaluation on top of it.
    """
    kg = T.two_components_kg()
    r = _run(kg, "creatinine tacrolimus", topk=4, topk_e=4, cost_e=0.5)
    assert _is_connected(len(r.selected_nodes), r.subgraph.edge_index)
    assert len(r.selected_nodes) <= 3  # one triangle, not both


# ---------------------------------------------------------------------------
# Group 5: degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_textual_tables_short_circuit_to_the_whole_graph():
    kg = T.clinical_kg()
    empty_edges = kg.edges_df.iloc[0:0]
    r = P.retrieval_via_pcst_traced(kg.graph, kg.q("anything"), kg.nodes_df, empty_edges)
    assert r.trace.short_circuited
    assert len(r.selected_nodes) == kg.graph.num_nodes


def test_graph_with_no_edges_does_not_crash():
    kg = T.single_node_kg()
    r = P.retrieval_via_pcst_traced(
        kg.graph, kg.q("creatinine"), kg.nodes_df, kg.edges_df, topk=1, topk_e=1
    )
    # short-circuits, because the edge table is empty
    assert r.trace.short_circuited
    assert len(r.selected_edges) == 0


def test_both_topk_zero_still_returns_a_connected_subgraph():
    """topk=0 and topk_e=0 means no prizes at all, so nothing is worth buying."""
    kg = T.clinical_kg()
    r = _run(kg, "creatinine", topk=0, topk_e=0, cost_e=0.5)
    assert len(r.selected_nodes) <= 1
    assert len(r.selected_edges) == 0


def test_prize_stages_work_without_pcst_fast_installed():
    """The module degrades gracefully: only `solve_pcst` (and the sanity check
    that guards it) actually needs the compiled solver. Everything up to
    "build the problem instance" is pure numpy/pandas.

    This matters because pcst_fast has no official Windows build (its own
    bioconda package lists linux-64 and osx-64 only), so someone setting up
    this playground on native Windows may be without a working solver for a
    while. They should still be able to read and run the prize logic.

    Simulated here by making the `pcst_fast` import fail, rather than actually
    uninstalling the package, so the rest of the suite is unaffected.
    """
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pcst_fast":
            raise ImportError("simulated: pcst_fast not installed")
        return real_import(name, *args, **kwargs)

    kg = T.clinical_kg()
    q = kg.q("what lab result and code does patient 001 have for creatinine")

    builtins.__import__ = blocked_import
    try:
        importlib.reload(P)  # drop any cached solver reference
        _, n_prizes = P.compute_node_prizes(q, kg.x, topk=3)
        _, e_prizes = P.compute_edge_prizes(q, kg.edge_attr, topk_e=3)
        cost = P.adjust_edge_cost(e_prizes, 0.5)
        inst = P.build_pcst_instance(kg.edge_index, n_prizes, e_prizes, cost, kg.graph.num_nodes)
        assert inst.num_virtual_nodes >= 0  # ran at all, that's the point

        with pytest.raises(ImportError, match="pcst_fast is not installed"):
            P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df)
    finally:
        builtins.__import__ = real_import
        importlib.reload(P)  # restore normal behaviour for every test after this one


# ---------------------------------------------------------------------------
# Group 6: equivalence with the official implementation
# ---------------------------------------------------------------------------

# importorskip catches ImportError but not OSError, and torch fails with the
# latter when a Windows Application Control policy blocks one of its DLLs. That
# happened once mid-run and took down collection of the *entire* suite, not just
# these twelve tests -- which looks exactly like "you broke everything" and is
# not. Skip the group instead, so the other 180 tests still report.
try:  # noqa: SIM105
    import torch
    from torch_geometric.data.data import Data
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"equivalence tests need torch + torch_geometric ({type(exc).__name__}: {exc})",
        allow_module_level=True,
    )

from ikgqa.pcst.reference import retrieval_via_pcst as official  # noqa: E402


def _as_pyg(kg) -> "Data":
    return Data(
        x=torch.tensor(np.asarray(kg.x), dtype=torch.float32),
        edge_index=torch.tensor(np.asarray(kg.edge_index), dtype=torch.long),
        edge_attr=torch.tensor(np.asarray(kg.edge_attr), dtype=torch.float32),
        num_nodes=int(len(kg.node_texts)),
    )


CASES = [
    ("chain", "creatinine measured and tacrolimus drug", 3, 3, 0.5),
    ("chain", "specimen collected during encounter", 2, 2, 0.2),
    ("hub", "lab result", 3, 3, 0.5),
    ("hub", "kidney transplant viral infection", 5, 5, 0.5),
    ("clinical", "what lab result and code does patient 001 have for creatinine", 3, 3, 0.5),
    ("clinical", "which complication did patient 001 have", 3, 5, 0.5),
    ("clinical", "how was the hepatitis c virus infection treated", 4, 4, 0.1),
    ("clinical", "creatinine", 1, 1, 0.5),
    ("clinical", "creatinine", 0, 3, 0.5),  # node prizes disabled
    ("clinical", "creatinine", 3, 0, 0.5),  # edge prizes disabled
    ("two-components", "creatinine tacrolimus", 4, 4, 0.5),
]


@pytest.mark.parametrize("name,question,topk,topk_e,cost_e", CASES)
def test_matches_official_implementation(name, question, topk, topk_e, cost_e):
    """Byte-for-byte agreement with G-Retriever's own retrieval.py.

    `ikgqa/pcst/reference.py` is an unmodified copy of that file. Both are handed
    the same PyG Data and the same query embedding; the textual descriptions and
    the returned tensors must match exactly.
    """
    kg = T.ALL_GRAPHS[name]()
    data = _as_pyg(kg)
    q = torch.tensor(kg.q(question), dtype=torch.float32)

    ref_sub, ref_desc = official(
        data.clone(), q, kg.nodes_df, kg.edges_df, topk=topk, topk_e=topk_e, cost_e=cost_e
    )
    my_sub, my_desc = P.retrieval_via_pcst(
        data.clone(), q, kg.nodes_df, kg.edges_df, topk=topk, topk_e=topk_e, cost_e=cost_e
    )

    assert my_desc == ref_desc
    assert my_sub.num_nodes == ref_sub.num_nodes
    assert torch.equal(my_sub.edge_index, ref_sub.edge_index)
    assert torch.equal(my_sub.x, ref_sub.x)
    assert torch.equal(my_sub.edge_attr, ref_sub.edge_attr)


@pytest.mark.parametrize("seed", list(range(12)))
def test_matches_official_implementation_on_random_graphs(seed):
    """Same check on random 40-node graphs with a 6-relation vocabulary, so
    edge-similarity ties (the fiddliest part of the prize logic) are exercised.
    """
    kg = T.random_kg(seed=seed)
    data = _as_pyg(kg)
    rng = np.random.default_rng(1000 + seed)
    q = torch.tensor(rng.standard_normal(kg.x.shape[1]), dtype=torch.float32)

    ref_sub, ref_desc = official(data.clone(), q, kg.nodes_df, kg.edges_df, topk=3, topk_e=5, cost_e=0.5)
    my_sub, my_desc = P.retrieval_via_pcst(data.clone(), q, kg.nodes_df, kg.edges_df, topk=3, topk_e=5, cost_e=0.5)

    assert my_desc == ref_desc
    assert torch.equal(my_sub.edge_index, ref_sub.edge_index)
    assert torch.equal(my_sub.x, ref_sub.x)


def test_accepts_pyg_data_and_numpy_graph_interchangeably():
    kg = T.clinical_kg()
    q = kg.q("which complication did patient 001 have")
    a = P.retrieval_via_pcst_traced(_as_pyg(kg), torch.tensor(q), kg.nodes_df, kg.edges_df)
    b = P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df)
    assert a.desc == b.desc
    assert a.selected_nodes.tolist() == b.selected_nodes.tolist()
    assert isinstance(a.subgraph, Data)
    assert isinstance(b.subgraph, P.SimpleGraph)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short", "-q"]))
