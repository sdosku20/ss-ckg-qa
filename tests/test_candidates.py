"""
Tests for two-stage retrieval.

The index mapping is the dangerous part. Stage one builds a smaller graph with
its own node ids, and if a result is returned in *those* ids instead of the
original graph's, every recall number in the thesis is silently wrong -- the
answers still look like plausible node ids, the metric still computes, and
nothing raises. So the mapping is tested directly, with a generator and an inner
retriever chosen so that the two index spaces cannot coincide by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from ikgqa.data.replica import PATIENT_0_SMALL, build_replica, planted_questions
from ikgqa.data.sphn import default_encoder
from ikgqa.data.toy import chain_kg
from ikgqa.graph import TextualGraph
from ikgqa.retrieval import (
    PCST,
    Retrieval,
    SeedExpansion,
    TopKTriples,
    TopNSimilar,
    TwoStage,
    assert_valid,
    induce,
)


@pytest.fixture(scope="module")
def replica():
    graph, table, _, rows = build_replica(PATIENT_0_SMALL, seed=0)
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = planted_questions(graph, table, rows, n_per_kind=3)
    return graph, table, encoder, questions


class FixedGenerator:
    """Returns a set chosen so sub-ids and original ids cannot coincide."""

    name = "fixed"
    params: dict = {}

    def __init__(self, keep):
        self._keep = np.asarray(keep, dtype=np.int64)

    def select(self, graph, q_emb):
        return self._keep


class FirstNodeRetriever:
    """Always returns sub-node 0 and no edges."""

    name = "first"
    params: dict = {}
    size_dial = None

    def retrieve(self, graph, q_emb):
        return Retrieval(np.array([0], dtype=np.int64), np.zeros(0, dtype=np.int64))


# ---------------------------------------------------------------------------
# induce
# ---------------------------------------------------------------------------


def test_induce_keeps_only_edges_with_both_endpoints():
    graph = chain_kg().textual
    keep = np.array([0, 1, 3])           # 1-2 and 2-3 must both go
    sub, node_map, edge_map = induce(graph, keep)
    assert sub.num_nodes == 3
    for j in range(sub.num_edges):
        s = int(sub.edges["src"].iloc[j])
        d = int(sub.edges["dst"].iloc[j])
        assert node_map[s] in keep and node_map[d] in keep


def test_induce_preserves_the_original_endpoints_of_every_kept_edge():
    """The remapping is only correct if sub-edge j joins the same two *original*
    nodes as original edge edge_map[j]."""
    graph = chain_kg().textual
    keep = np.array([1, 2, 3])
    sub, node_map, edge_map = induce(graph, keep)
    for j in range(sub.num_edges):
        original = int(edge_map[j])
        assert node_map[int(sub.edges["src"].iloc[j])] == int(graph.edges["src"].iloc[original])
        assert node_map[int(sub.edges["dst"].iloc[j])] == int(graph.edges["dst"].iloc[original])
        assert sub.edges["edge_attr"].iloc[j] == graph.edges["edge_attr"].iloc[original]


def test_induce_carries_embeddings_and_both_text_columns(replica):
    graph, _, _, _ = replica
    keep = np.array([0, 5, 11, 40, 100])
    sub, node_map, _ = induce(graph, keep)
    assert np.array_equal(sub.node_emb, graph.node_emb[keep])
    assert "embed_attr" in sub.nodes.columns, "embed text must survive, or KAPING ranks display text"
    assert list(sub.embed_texts) == [graph.embed_texts[i] for i in keep]
    assert list(sub.node_texts) == [graph.node_texts[i] for i in keep]


def test_induce_deduplicates_and_sorts_its_input():
    graph = chain_kg().textual
    sub_a, map_a, _ = induce(graph, np.array([3, 1, 1, 0]))
    sub_b, map_b, _ = induce(graph, np.array([0, 1, 3]))
    assert np.array_equal(map_a, map_b)
    assert sub_a.num_nodes == sub_b.num_nodes == 3


def test_induce_rejects_out_of_range_ids():
    graph = chain_kg().textual
    with pytest.raises(ValueError, match="candidate node ids"):
        induce(graph, np.array([0, graph.num_nodes]))


def test_induce_of_everything_is_the_same_graph():
    graph = chain_kg().textual
    sub, node_map, edge_map = induce(graph, np.arange(graph.num_nodes))
    assert sub.num_nodes == graph.num_nodes
    assert sub.num_edges == graph.num_edges
    assert np.array_equal(node_map, np.arange(graph.num_nodes))
    assert np.array_equal(edge_map, np.arange(graph.num_edges))


def test_induce_of_an_isolated_set_gives_a_valid_edgeless_graph():
    graph = chain_kg().textual
    sub, _, edge_map = induce(graph, np.array([0, 3]))
    assert isinstance(sub, TextualGraph)
    assert sub.num_edges == 0 and edge_map.size == 0
    assert sub.edge_emb.shape[1] == graph.node_emb.shape[1], "dim must survive for cosine"


# ---------------------------------------------------------------------------
# The index space -- the thing that would corrupt every result
# ---------------------------------------------------------------------------


def test_results_come_back_in_the_original_index_space(replica):
    """Sub-node 0 is original node 40 here. Returning 0 would be a plausible
    node id, would pass every shape check, and would be wrong."""
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    two = TwoStage(generator=FixedGenerator([40, 41, 90]), inner=FirstNodeRetriever())
    got = two.retrieve(graph, q)
    assert got.node_ids.tolist() == [40], "ids were not translated back"


def test_edge_ids_come_back_in_the_original_index_space():
    graph = chain_kg().textual

    class AllEdges:
        name = "all"
        params: dict = {}
        size_dial = None

        def retrieve(self, g, q_emb):
            return Retrieval(
                np.arange(g.num_nodes, dtype=np.int64),
                np.arange(g.num_edges, dtype=np.int64),
            )

    keep = np.array([1, 2, 3])
    _, node_map, edge_map = induce(graph, keep)
    got = TwoStage(generator=FixedGenerator(keep), inner=AllEdges()).retrieve(
        graph, np.zeros(graph.node_emb.shape[1], dtype=np.float32)
    )
    assert got.node_ids.tolist() == sorted(node_map.tolist())
    assert got.edge_ids.tolist() == sorted(edge_map.tolist())


def test_two_stage_output_satisfies_the_retrieval_contract(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    for generator in (TopNSimilar(n=150), SeedExpansion(k=5, hops=2, cap=200)):
        for inner in (
            PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable"),
            TopKTriples(k=10, encoder=encoder),
        ):
            got = TwoStage(generator=generator, inner=inner).retrieve(graph, q)
            assert_valid(got, graph)


def test_an_empty_candidate_set_returns_an_empty_result(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    two = TwoStage(generator=FixedGenerator([]), inner=FirstNodeRetriever())
    got = two.retrieve(graph, q)
    assert got.num_nodes == 0 and got.num_edges == 0


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_top_n_similar_returns_exactly_n_sorted_unique_nodes(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    keep = TopNSimilar(n=137).select(graph, q)
    assert keep.size == 137
    assert np.all(np.diff(keep) > 0)


def test_top_n_similar_is_deterministic_under_massive_ties(replica):
    """Thousands of nodes share one text (F3), so ties are the normal case, not
    an edge case. Two calls must agree or no sweep is reproducible."""
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    assert np.array_equal(TopNSimilar(n=300).select(graph, q), TopNSimilar(n=300).select(graph, q))


def test_seed_expansion_contains_its_seeds_and_respects_the_cap(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    seeds = TopNSimilar(n=5).select(graph, q)
    keep = SeedExpansion(k=5, hops=2, cap=250).select(graph, q)
    assert keep.size <= 250
    assert set(seeds.tolist()) <= set(keep.tolist()), "seeds must survive their own expansion"


def test_seed_expansion_at_zero_hops_is_just_the_seeds(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    assert np.array_equal(
        SeedExpansion(k=7, hops=0, cap=10_000).select(graph, q),
        TopNSimilar(n=7).select(graph, q),
    )


def test_more_hops_never_shrink_the_region(replica):
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    sizes = [
        SeedExpansion(k=5, hops=h, cap=10_000).select(graph, q).size for h in (0, 1, 2)
    ]
    assert sizes[0] <= sizes[1] <= sizes[2]


def test_seed_expansion_keeps_the_nearest_nodes_when_the_cap_binds(replica):
    """On a graph where one hop can reach most of the nodes, the cap decides
    what a clinician sees. It must cut by distance, not by array order."""
    graph, _, encoder, questions = replica
    q = encoder.encode_one(questions[0].text)
    seeds = TopNSimilar(n=3).select(graph, q)
    capped = SeedExpansion(k=3, hops=3, cap=50).select(graph, q)
    assert capped.size == 50
    assert set(seeds.tolist()) <= set(capped.tolist()), "distance-0 nodes must survive the cap"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_ceiling_bounds_the_recall_actually_achieved(replica):
    """A selection cannot recover an answer stage one discarded. If this ever
    fails, either the ceiling or the id mapping is wrong."""
    graph, _, encoder, questions = replica
    for question in questions:
        q = encoder.encode_one(question.text)
        two = TwoStage(
            generator=SeedExpansion(k=10, hops=2, cap=400),
            inner=PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable"),
        )
        got = two.retrieve(graph, q)
        gold = set(question.answer_nodes)
        achieved = len(gold & set(got.node_ids.tolist())) / len(gold)
        ceiling = two.ceiling_for(graph, q, question.answer_nodes)
        assert achieved <= ceiling + 1e-9, f"{question.qid}: recall above its own ceiling"


def test_params_and_size_dial_describe_both_stages(replica):
    two = TwoStage(
        generator=SeedExpansion(k=4, hops=2, cap=100),
        inner=PCST(topk=3, topk_e=0, cost_e=0.5),
    )
    assert two.size_dial == "cost_e", "the dial is the inner retriever's"
    assert two.params["gen_k"] == 4 and two.params["gen_cap"] == 100
    assert two.params["inner_cost_e"] == 0.5
    assert "PCST" in two.name and "seed expansion" in two.name
