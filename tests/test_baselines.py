"""
test_baselines.py
=================

Run with:   pytest -v test_baselines.py

Checks the three Appendix D.1 baselines behave as described, and pins down what
actually separates PCST from them on the toy clinical graph. Read the last two
tests first: they are the interesting ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ikgqa import retrieval as B
from ikgqa.eval import metrics as M
from ikgqa.eval import toy_report as R
from ikgqa.pcst import core as P
from ikgqa.data import toy as T

GOLD = T.clinical_gold_set()
RETRIEVERS = ["topk_triples", "topk_nodes_plus_neighbors", "shortest_paths"]


def _call(name, kg, q, k=5):
    if name == "topk_triples":
        return B.retrieve_topk_triples(kg.graph, q, k=k)
    if name == "topk_nodes_plus_neighbors":
        return B.retrieve_topk_nodes_plus_neighbors(kg.graph, q, k=k)
    if name == "shortest_paths":
        return B.retrieve_shortest_paths(kg.graph, q, k=k)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Shared contract: every retriever returns the same shape of answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", RETRIEVERS)
def test_returns_valid_sorted_ids(name):
    kg = T.clinical_kg()
    nodes, edges = _call(name, kg, kg.q("creatinine lab result for patient 001"))
    assert nodes.tolist() == sorted(set(nodes.tolist()))
    assert edges.tolist() == sorted(set(edges.tolist()))
    assert nodes.max(initial=-1) < kg.graph.num_nodes
    assert edges.max(initial=-1) < kg.graph.num_edges


@pytest.mark.parametrize("name", RETRIEVERS)
def test_node_set_covers_every_selected_edge_endpoint(name):
    """Same invariant PCST maintains, so the descriptions stay comparable."""
    kg = T.clinical_kg()
    nodes, edges = _call(name, kg, kg.q("which complication did patient 001 have"))
    ei = np.asarray(kg.edge_index)[:, edges]
    assert set(ei.flatten().tolist()) <= set(nodes.tolist())


@pytest.mark.parametrize("name", RETRIEVERS)
def test_descriptions_are_buildable_from_any_retriever(name):
    kg = T.clinical_kg()
    nodes, edges = _call(name, kg, kg.q("creatinine"))
    desc = P.build_description(kg.nodes_df, kg.edges_df, nodes, edges)
    assert "node_id,node_attr" in desc and "src,edge_attr,dst" in desc


# ---------------------------------------------------------------------------
# What each baseline does wrong
# ---------------------------------------------------------------------------


def test_topk_triples_returns_exactly_k_triples():
    kg = T.clinical_kg()
    for k in (1, 3, 5, 8):
        _, edges = B.retrieve_topk_triples(kg.graph, kg.q("creatinine"), k=k)
        assert len(edges) == k


def test_topk_triples_can_return_a_disconnected_bag_of_facts():
    """The structural weakness PCST is designed to fix.

    KAPING picks the k best-scoring triples independently, so nothing forces
    them to touch. Here two of the five gold questions produce a result with
    more than one component, and the LLM gets facts with no path between them.
    """
    kg = T.clinical_kg()
    triple_texts = [f"{kg.node_texts[s]} {r} {kg.node_texts[d]}" for s, r, d in kg.triples]
    triple_emb = kg.encoder.encode(triple_texts)

    disconnected = 0
    for question in GOLD:
        nodes, edges = B.retrieve_topk_triples(kg.graph, kg.q(question), k=5, triple_emb=triple_emb)
        if not M.is_connected(kg.graph.num_nodes, kg.edge_index, nodes, edges):
            disconnected += 1
    assert disconnected > 0


def test_topk_nodes_plus_neighbors_has_no_size_dial():
    """Its output size is set by node degree, not by anything you can tune.

    `k` chooses how many *seeds* to expand, not how much to return, so the
    output jumps in whatever increments the degrees happen to be: on this graph
    6 edges at k=1, then 11, then 12. There is no equivalent of `cost_e` for
    saying "that is too much, give me less". By k=3 it is already returning more
    than half the graph, and PCST answers the same question with fewer edges.
    """
    kg = T.clinical_kg()
    q = kg.q("which complication did patient 001 have")
    sizes = [len(B.retrieve_topk_nodes_plus_neighbors(kg.graph, q, k=k)[1]) for k in (1, 2, 3)]

    assert sizes == sorted(sizes)  # monotone, but in degree-sized steps
    assert sizes[-1] > kg.graph.num_edges / 2
    at_k5 = len(B.retrieve_topk_nodes_plus_neighbors(kg.graph, q, k=5)[1])
    pcst_size = len(P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df).selected_edges)
    assert pcst_size < min(sizes[1:]) < at_k5


def test_shortest_paths_needs_the_answer_node_to_match_the_question():
    """The sharpest difference between PCST and the node-seeded baselines.

    Ask "which complication did patient 001 have". The answer node is
    "graft rejection episode", which shares no word with the question, so it
    never enters the top-k node seeds and no shortest path can reach it.

    PCST finds it anyway, because it scores *edges* too, and the relation
    "has complication" does match the question. The edge prize turns into a
    virtual node, and buying that virtual node drags in both of its endpoints.
    """
    kg = T.clinical_kg()
    question = "which complication did patient 001 have"
    gold_edge = 9  # (patient 001) -[has complication]-> (graft rejection episode)
    answer_node = 17  # graft rejection episode

    q = kg.q(question)
    node_sim = P.cosine_similarity(q, kg.x)
    assert node_sim[answer_node] == 0.0  # invisible to any node-similarity seed

    _, sp_edges = B.retrieve_shortest_paths(kg.graph, q, k=5)
    assert gold_edge not in sp_edges.tolist()

    r = P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df, topk=3, topk_e=5)
    assert gold_edge in r.selected_edges.tolist()
    assert answer_node in r.selected_nodes.tolist()
    assert r.trace.edge_prizes[gold_edge] > 0  # found via the relation, not the node


# ---------------------------------------------------------------------------
# The comparison, and what it does and does not show
# ---------------------------------------------------------------------------


def test_score_returns_one_row_per_retriever():
    kg = T.clinical_kg()
    question, gold = next(iter(GOLD.items()))
    df = R.score(kg, question, gold, k=5)
    assert len(df) == 4
    assert set(df.columns) == {
        "retriever", "recall", "precision", "nodes", "edges", "chars", "connected"
    }
    assert df["recall"].between(0, 1).all()


def test_pcst_is_the_only_retriever_that_is_always_connected():
    """This, not recall, is what PCST buys you on a graph this small.

    On 20 edges with a bag-of-words encoder, top-k triples matches PCST's
    recall. The paper's accuracy gap comes from WebQSP, where graphs average
    1370 nodes and 4252 edges and independently chosen triples fragment badly.
    What is already visible here is the guarantee: PCST returns one connected
    component every time, by construction, and the alternatives do not.
    """
    kg = T.clinical_kg()
    agg = R.score_all(kg, GOLD, k=5)
    assert bool(agg.loc["PCST", "always_connected"]) is True
    assert not bool(agg.loc["top-k triples (KAPING)", "always_connected"])


def test_pcst_beats_the_node_seeded_baselines_on_this_gold_set():
    """Recall against the hand-written gold triples. Reported as a fact about
    this toy set, not as a reproduction of Table 10.
    """
    kg = T.clinical_kg()
    agg = R.score_all(kg, GOLD, k=5)
    assert agg.loc["PCST", "recall"] > agg.loc["shortest paths", "recall"]
    assert agg.loc["PCST", "recall"] >= agg.loc["top-k nodes + neighbours", "recall"]
    # and it does it with a far smaller prompt than the neighbourhood baseline
    assert agg.loc["PCST", "chars"] < 0.6 * agg.loc["top-k nodes + neighbours", "chars"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short", "-q"]))
