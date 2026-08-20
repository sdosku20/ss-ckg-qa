"""
Tests for the data layer: TextualGraph and the encoders.

The point of these is not coverage for its own sake. Every assertion here
corresponds to a way a research result could come out wrong without anything
crashing -- misaligned embeddings, a silently growing vocabulary, an unnormalised
vector making cosine similarity mean something other than what the paper says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ikgqa.data import toy as T
from ikgqa.encoders import BagOfWordsEncoder, Encoder, l2_normalise, tokenize
from ikgqa.graph import TextualGraph
from ikgqa.pcst import core as P


@pytest.fixture
def clinical() -> TextualGraph:
    kg = T.clinical_kg()
    return TextualGraph.from_texts(kg.node_texts, kg.triples, kg.encoder, name="clinical")


# ---------------------------------------------------------------------------
# TextualGraph: construction
# ---------------------------------------------------------------------------


def test_from_texts_keeps_rows_aligned_with_embeddings(clinical):
    assert clinical.num_nodes == len(clinical.node_texts) == clinical.node_emb.shape[0]
    assert clinical.num_edges == len(clinical.edge_texts) == clinical.edge_emb.shape[0]
    assert clinical.nodes["node_attr"].iloc[3] == clinical.node_texts[3]


def test_edge_index_matches_the_edge_table(clinical):
    ei = clinical.edge_index
    assert ei.shape == (2, clinical.num_edges)
    assert ei.dtype == np.int64
    np.testing.assert_array_equal(ei[0], clinical.edges["src"].to_numpy())
    np.testing.assert_array_equal(ei[1], clinical.edges["dst"].to_numpy())


def test_triple_texts_include_both_endpoints(clinical):
    triples = clinical.triple_texts()
    assert len(triples) == clinical.num_edges
    src0 = clinical.node_texts[int(clinical.edges["src"].iloc[0])]
    dst0 = clinical.node_texts[int(clinical.edges["dst"].iloc[0])]
    assert src0 in triples[0] and dst0 in triples[0]


def test_graph_is_immutable(clinical):
    """Frozen so an experiment loop cannot mutate the graph between runs."""
    with pytest.raises(Exception):
        clinical.name = "changed"


# ---------------------------------------------------------------------------
# TextualGraph: the failure modes worth catching early
# ---------------------------------------------------------------------------


def _valid_parts():
    nodes = pd.DataFrame({"node_id": [0, 1], "node_attr": ["a", "b"]})
    edges = pd.DataFrame({"src": [0], "edge_attr": ["r"], "dst": [1]})
    return nodes, edges, np.eye(2, dtype=np.float32), np.ones((1, 2), dtype=np.float32)


def test_wrong_number_of_node_embeddings_is_rejected():
    nodes, edges, _, edge_emb = _valid_parts()
    with pytest.raises(ValueError, match="2 nodes but 3 node embeddings"):
        TextualGraph(nodes, edges, np.zeros((3, 2), dtype=np.float32), edge_emb)


def test_mismatched_node_and_edge_dimensions_are_rejected():
    """One query vector is compared against both, so the dims must agree."""
    nodes, edges, node_emb, _ = _valid_parts()
    with pytest.raises(ValueError, match="share a dimension"):
        TextualGraph(nodes, edges, node_emb, np.ones((1, 7), dtype=np.float32))


def test_out_of_range_edge_endpoint_is_rejected():
    nodes, _, node_emb, edge_emb = _valid_parts()
    bad_edges = pd.DataFrame({"src": [0], "edge_attr": ["r"], "dst": [99]})
    with pytest.raises(ValueError, match="edge endpoints must be node ids"):
        TextualGraph(nodes, bad_edges, node_emb, edge_emb)


def test_missing_columns_are_named_in_the_error():
    _, edges, node_emb, edge_emb = _valid_parts()
    with pytest.raises(ValueError, match="node_attr"):
        TextualGraph(pd.DataFrame({"node_id": [0, 1]}), edges, node_emb, edge_emb)


def test_a_graph_with_no_edges_is_legal():
    """Isolated nodes happen after filtering a cohort; must not crash."""
    nodes = pd.DataFrame({"node_id": [0, 1], "node_attr": ["a", "b"]})
    empty = pd.DataFrame({"src": [], "edge_attr": [], "dst": []})
    graph = TextualGraph(nodes, empty, np.eye(2, dtype=np.float32), np.zeros((0, 2), np.float32))
    assert graph.num_edges == 0
    assert graph.edge_index.shape == (2, 0)


# ---------------------------------------------------------------------------
# TextualGraph: interoperability with the PCST core
# ---------------------------------------------------------------------------


def test_as_simple_graph_feeds_the_pcst_core_unchanged(clinical):
    kg = T.clinical_kg()
    q = kg.q("which complication did patient 001 have")

    from_container = P.retrieval_via_pcst_traced(
        clinical.as_simple_graph(), q, clinical.nodes, clinical.edges
    )
    from_toy = P.retrieval_via_pcst_traced(kg.graph, q, kg.nodes_df, kg.edges_df)

    np.testing.assert_array_equal(from_container.selected_nodes, from_toy.selected_nodes)
    np.testing.assert_array_equal(from_container.selected_edges, from_toy.selected_edges)
    assert from_container.desc == from_toy.desc


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Creatinine, 2.1 mg/dL!") == ["creatinine", "2", "1", "mg", "dl"]


def test_bag_of_words_rows_are_unit_length():
    """Cosine similarity is only a dot product if rows are normalised."""
    enc = BagOfWordsEncoder(["kidney transplant", "liver biopsy"])
    mat = enc.encode(["kidney transplant", "liver biopsy"])
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-6)


def test_bag_of_words_ranks_shared_vocabulary_higher():
    enc = BagOfWordsEncoder(["kidney transplant rejection", "liver enzyme level"])
    q = enc.encode_one("kidney rejection")
    sim = enc.encode(["kidney transplant rejection", "liver enzyme level"]) @ q
    assert sim[0] > sim[1]


def test_vocabulary_is_fixed_so_earlier_vectors_keep_their_meaning():
    """An unseen word is ignored, not appended: a growing space would make
    embeddings from different calls incomparable."""
    enc = BagOfWordsEncoder(["kidney"])
    before = enc.encode_one("kidney")
    _ = enc.encode(["tacrolimus trough level"])
    np.testing.assert_array_equal(enc.encode_one("kidney"), before)
    assert enc.dim == 1


def test_text_sharing_no_vocabulary_gives_a_zero_vector_not_a_crash():
    enc = BagOfWordsEncoder(["kidney"])
    vec = enc.encode_one("completely unrelated words")
    np.testing.assert_array_equal(vec, np.zeros_like(vec))


def test_l2_normalise_survives_an_all_zero_row():
    out = l2_normalise(np.zeros((1, 3), dtype=np.float32))
    assert np.isfinite(out).all()


def test_bag_of_words_satisfies_the_encoder_protocol():
    assert isinstance(BagOfWordsEncoder(["a"]), Encoder)


def test_sentence_transformer_encoder_fails_with_actionable_advice():
    """If the real encoder is unavailable, the message must say what to do."""
    from ikgqa.encoders import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="BagOfWordsEncoder"):
            enc.dim
    else:  # pragma: no cover - only on machines with the real model installed
        assert enc.dim > 0
