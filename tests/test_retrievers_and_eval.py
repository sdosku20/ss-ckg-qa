"""
Tests for the retriever interface and the evaluation layer.

Two things are being protected here:

  * the uniform interface -- every retriever returns sorted, in-range ids whose
    node set covers its edge set, because every metric assumes it;
  * the honesty of the metrics -- an aggregate question must not be scored as a
    miss, an undefined recall must not be averaged as zero, and a mean must
    always be reported with the count of items behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ikgqa.data import toy as T
from ikgqa.eval import metrics as M
from ikgqa.eval.sweep import Sweep, sweep, to_curve
from ikgqa.graph import TextualGraph
from ikgqa.retrieval import (
    DEFAULT_RETRIEVERS,
    PCST,
    BFSExpansion,
    Retrieval,
    ShortestPaths,
    TopKNodesPlusNeighbors,
    TopKTriples,
    assert_valid,
)
from ikgqa.retrieval.paths import retrieve_bfs_expansion
from ikgqa.retrieval.similarity import retrieve_topk_nodes_plus_neighbors


@pytest.fixture
def kg():
    return T.clinical_kg()


@pytest.fixture
def graph(kg) -> TextualGraph:
    return TextualGraph.from_texts(kg.node_texts, kg.triples, kg.encoder, name="clinical")


@pytest.fixture
def questions(kg) -> list:
    """Entity questions with the answer node identified by its text.

    Deliberately small and hand-checkable: these are unit-test fixtures, not
    the thesis gold set (which is 50 human-validated questions on the real
    graph, see thesis Section 4.3).
    """

    def node(text: str) -> int:
        hits = [i for i, t in enumerate(kg.node_texts) if t == text]
        assert len(hits) == 1, f"{text!r} matched {len(hits)} nodes, need exactly 1"
        return hits[0]

    return [
        M.Question(
            text="which complication did patient 001 have",
            answer_nodes=(node("graft rejection episode"),),
            answer_edges=(9,),
            qid="q1",
            validated=True,
        ),
        M.Question(
            text="what lab result and code does patient 001 have for creatinine",
            answer_nodes=(
                node("creatinine serum 2 1 mg per dl"),
                node("loinc 2160 0 creatinine"),
            ),
            answer_edges=(2, 5),
            qid="q2",
            validated=True,
        ),
    ]


# ---------------------------------------------------------------------------
# The interface contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retriever", DEFAULT_RETRIEVERS, ids=lambda r: r.name)
def test_every_retriever_honours_the_contract(retriever, graph, kg):
    selection = retriever.retrieve(graph, kg.q("creatinine level for patient 001"))
    assert isinstance(selection, Retrieval)
    assert_valid(selection, graph)


@pytest.mark.parametrize("retriever", DEFAULT_RETRIEVERS, ids=lambda r: r.name)
def test_labels_are_stable_and_carry_parameters(retriever):
    assert retriever.label() == retriever.label()
    for key in retriever.params:
        assert key in retriever.label() or not retriever.params


def test_params_exclude_collaborators_and_caches(kg):
    """An encoder in a results table would be noise; a cache would be unstable."""
    r = TopKTriples(k=4, encoder=kg.encoder)
    assert r.params == {"k": 4}
    assert "encoder" not in r.label()


def test_only_pcst_declares_a_true_size_dial():
    dials = {r.name: r.size_dial for r in DEFAULT_RETRIEVERS}
    assert dials["PCST"] == "cost_e"
    # Neighbourhood expansion has no honest size control: k picks seeds, degree
    # decides the size. Recording that as None is the point.
    assert dials["top-k nodes + neighbours"] is None


def test_pcst_rejects_a_graph_without_text_instead_of_guessing(graph, kg):
    with pytest.raises(TypeError, match="TextualGraph"):
        PCST().retrieve(graph.as_simple_graph(), kg.q("creatinine"))


def test_kaping_needs_triple_embeddings_to_find_entity_worded_questions(graph, kg):
    """Relation-text-only would understate the published baseline.

    "graft rejection episode" appears in node text and in no relation text, so
    ranking on relation text alone cannot find edge 9 -- every relation scores
    exactly 0 and the tie is broken by index. Given triple embeddings, edge 9
    is the obvious top hit. This is why TopKTriples takes an encoder.
    """
    q = kg.q("graft rejection episode")
    without = TopKTriples(k=3).retrieve(graph, q)
    with_triples = TopKTriples(k=3, encoder=kg.encoder).retrieve(graph, q)

    assert without.num_edges == with_triples.num_edges == 3
    assert 9 in with_triples.edge_ids.tolist()
    assert 9 not in without.edge_ids.tolist()


def test_bfs_at_one_hop_is_a_superset_of_nodes_plus_neighbours(graph, kg):
    """They are related but not equal; the ablation depends on knowing which."""
    q = kg.q("creatinine")
    _, bfs_edges = retrieve_bfs_expansion(graph.as_simple_graph(), q, k=3, hops=1)
    _, nbr_edges = retrieve_topk_nodes_plus_neighbors(graph.as_simple_graph(), q, k=3)
    assert set(nbr_edges.tolist()) <= set(bfs_edges.tolist())


def test_bfs_rejects_negative_hops(graph, kg):
    with pytest.raises(ValueError, match="hops must be >= 0"):
        BFSExpansion(hops=-1).retrieve(graph, kg.q("creatinine"))


def test_pcst_is_connected_where_the_seeded_baselines_need_not_be(graph, kg):
    q = kg.q("creatinine tacrolimus hepatitis")
    pcst = PCST(topk=3, topk_e=5).retrieve(graph, q)
    assert M.is_connected(graph.num_nodes, graph.edge_index, pcst.node_ids, pcst.edge_ids)

    triples = TopKTriples(k=3, encoder=kg.encoder).retrieve(graph, q)
    # Not an assertion that it *is* disconnected -- on a small graph it may not
    # be. The claim is only that nothing in its construction guarantees it.
    assert triples.num_edges <= 3


# ---------------------------------------------------------------------------
# Metrics: the honesty properties
# ---------------------------------------------------------------------------


def test_recall_of_an_empty_gold_set_is_undefined_not_zero():
    assert np.isnan(M.recall([1, 2], []))
    assert M.recall([1, 2], [1, 3]) == 0.5


def test_precision_of_an_empty_retrieval_is_undefined_not_zero():
    assert np.isnan(M.precision([], [1]))


def test_an_entity_question_without_an_answer_node_is_a_construction_error():
    with pytest.raises(ValueError, match="mark it 'aggregate'"):
        M.Question(text="how many patients", kind=M.KIND_ENTITY)


def test_an_unknown_question_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        M.Question(text="x", answer_nodes=(0,), kind="entity-ish")


def test_aggregate_questions_are_skipped_with_a_reason_not_scored_zero(graph, kg):
    """A count has no answer node; scoring it 0 would corrupt every mean."""
    q = M.Question(text="how many patients had a rejection", kind=M.KIND_AGGREGATE, qid="agg")
    selection = PCST().retrieve(graph, kg.q(q.text))
    row = M.measure(graph, selection, q)
    assert np.isnan(row["answer_node_recall"])
    assert "aggregate" in row["skipped_reason"]


def test_evaluate_produces_one_row_per_retriever_and_question(graph, kg, questions):
    retrievers = [PCST(), TopKTriples(k=5, encoder=kg.encoder)]
    out = M.evaluate(graph, retrievers, questions, kg.encoder)
    assert len(out) == len(retrievers) * len(questions)
    assert set(out["qid"]) == {"q1", "q2"}
    assert "param_cost_e" in out.columns


def test_evaluate_refuses_a_mismatched_question_encoder(graph, kg, questions):
    """The classic silent-garbage bug: questions embedded in another space."""
    wrong = np.zeros((len(questions), graph.dim + 5), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        M.evaluate(graph, [PCST()], questions, kg.encoder, q_embeddings=wrong)


def test_evaluate_refuses_an_empty_question_set(graph, kg):
    with pytest.raises(ValueError, match="no questions"):
        M.evaluate(graph, [PCST()], [], kg.encoder)


def test_summarise_reports_how_many_questions_were_actually_scored(graph, kg, questions):
    mixed = list(questions) + [
        M.Question(text="how many patients in total", kind=M.KIND_AGGREGATE, qid="agg")
    ]
    out = M.evaluate(graph, [PCST()], mixed, kg.encoder)
    agg = M.summarise(out)
    assert agg["n_questions"].iloc[0] == 3
    assert agg["n_scored"].iloc[0] == 2  # the aggregate one contributed nothing
    assert 0.0 <= agg["answer_node_recall"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# Sweeps: the recall-versus-size curve
# ---------------------------------------------------------------------------


def test_sweep_returns_one_row_per_configuration(graph, kg, questions):
    out = sweep(
        graph,
        [Sweep(lambda c: PCST(cost_e=c), [0.1, 0.5, 1.0], dial="cost_e")],
        questions,
        kg.encoder,
    )
    assert len(out) == 3
    assert set(out["dial_value"]) == {0.1, 0.5, 1.0}
    assert (out["retriever"] == "PCST").all()


def test_sweep_can_compare_families_with_different_dials(graph, kg, questions):
    out = sweep(
        graph,
        [
            Sweep(lambda c: PCST(cost_e=c), [0.1, 1.0], dial="cost_e"),
            Sweep(lambda k: TopKTriples(k=k, encoder=kg.encoder), [1, 5], dial="k"),
        ],
        questions,
        kg.encoder,
    )
    assert set(out["retriever"]) == {"PCST", "top-k triples (KAPING)"}
    assert set(out["dial"]) == {"cost_e", "k"}


def test_raising_the_edge_cost_does_not_grow_the_subgraph(graph, kg, questions):
    """The size dial has to actually control size, or the curve is meaningless."""
    out = sweep(
        graph,
        [Sweep(lambda c: PCST(cost_e=c), [0.05, 0.5, 5.0], dial="cost_e")],
        questions,
        kg.encoder,
    ).sort_values("dial_value")
    sizes = out["nodes"].tolist()
    assert sizes == sorted(sizes, reverse=True), f"expected non-increasing sizes, got {sizes}"


def test_to_curve_keeps_only_what_a_plot_needs(graph, kg, questions):
    out = sweep(graph, [Sweep(lambda c: PCST(cost_e=c), [0.5], dial="cost_e")], questions, kg.encoder)
    curve = to_curve(out, size_axis="chars")
    assert list(curve.columns) == [
        "retriever", "config", "dial", "dial_value", "size", "answer_node_recall", "n_scored",
    ]


def test_to_curve_rejects_a_size_axis_that_does_not_exist(graph, kg, questions):
    out = sweep(graph, [Sweep(lambda c: PCST(cost_e=c), [0.5], dial="cost_e")], questions, kg.encoder)
    with pytest.raises(ValueError, match="size_axis must be one of"):
        to_curve(out, size_axis="tokens")


def test_sweep_refuses_to_run_with_no_sweeps(graph, kg, questions):
    with pytest.raises(ValueError, match="no sweeps"):
        sweep(graph, [], questions, kg.encoder)
