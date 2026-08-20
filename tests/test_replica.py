"""
Tests for the measured-shape replica.

The replica's whole value is that it behaves like the real graph. These tests
pin the properties that value depends on, so a convenient-looking change to the
generator cannot quietly make offline results optimistic.
"""

from __future__ import annotations

import pytest

from ikgqa.data.replica import (
    PATIENT_0,
    PATIENT_0_SMALL,
    build_replica,
    describe_fidelity,
    generate_rows,
    planted_questions,
)
from ikgqa.data.sphn import default_encoder
from ikgqa.eval.metrics import evaluate
from ikgqa.retrieval import PCST, TopKTriples, assert_valid


@pytest.fixture(scope="module")
def small():
    return build_replica(PATIENT_0_SMALL, seed=0)


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------


def test_structural_counts_match_the_measured_patient():
    """The full profile must reproduce patient #0 exactly, not approximately.

    Node and edge counts are the load-bearing ones: if they drift, every size
    measured on the replica stops being comparable to the real thing.
    """
    graph, _, stats, _ = build_replica(PATIENT_0, seed=0)
    assert graph.num_nodes == PATIENT_0.nodes
    assert graph.num_edges == PATIENT_0.edges
    assert stats.labs == PATIENT_0.lab_observations
    assert stats.lab_types == PATIENT_0.analytes
    assert stats.diagnoses == PATIENT_0.diagnoses
    assert stats.unnamed_diagnoses == PATIENT_0.unnamed_diagnoses
    assert stats.drug_administrations == PATIENT_0.drug_administrations
    assert stats.drugs == PATIENT_0.drugs
    assert stats.cases == PATIENT_0.cases


def test_generation_is_deterministic():
    """Two runs at the same seed must be identical, or no result is reproducible."""
    a = generate_rows(PATIENT_0_SMALL, seed=7)
    b = generate_rows(PATIENT_0_SMALL, seed=7)
    assert a == b
    c = generate_rows(PATIENT_0_SMALL, seed=8)
    assert a != c, "different seeds must give different data"


def test_fidelity_report_shows_the_gap_rather_than_hiding_it(small):
    _, _, stats, _ = small
    text = describe_fidelity(PATIENT_0_SMALL, stats, small[0])
    assert "measured -> achieved" in text
    assert "lab observations" in text


def test_measurements_are_skewed_not_uniform():
    """Real lab activity is heavily skewed. A uniform split would understate how
    decisive tie-breaking is, which is the property the replica exists to show."""
    rows = generate_rows(PATIENT_0_SMALL, seed=0)
    counts = {}
    for row in rows["labs"]:
        counts[row["loinc_code"]] = counts.get(row["loinc_code"], 0) + 1
    ordered = sorted(counts.values(), reverse=True)
    assert ordered[0] > 5 * ordered[-1], "distribution is too flat to be realistic"


# ---------------------------------------------------------------------------
# The properties that make the real graph hard
# ---------------------------------------------------------------------------


def test_text_is_shared_at_high_multiplicity(small):
    """The core difficulty: many more nodes than distinct texts."""
    graph, _, stats, _ = small
    assert graph.num_nodes / stats.distinct_embed_texts > 5


def test_the_graph_is_bilingual(small):
    """An English-only encoder must be measurably disadvantaged somewhere, or the
    terminology work has nothing to fix."""
    _, table, _, _ = small
    dx = table[table["sphn_label"] == "BilledDiagnosis"]["display_text"]
    assert any("Stadium" in t or "Erkrankung" in t for t in dx), "no German text present"
    labs = table[table["sphn_label"] == "LabTestType"]["display_text"]
    assert any("in Serum or Plasma" in t for t in labs), "no English text present"


def test_code_only_diagnoses_carry_no_description(small):
    _, table, stats, rows = small
    assert stats.unnamed_diagnoses == PATIENT_0_SMALL.unnamed_diagnoses
    codeonly = [r for r in rows["diagnoses"] if not r["diagnosis_name"]]
    assert codeonly and all(r["_true_name"] for r in codeonly), (
        "the withheld description must still be recorded, or the terminology "
        "gap cannot be measured"
    )


# ---------------------------------------------------------------------------
# Planted questions
# ---------------------------------------------------------------------------


def test_questions_never_quote_the_code_they_look_for(small):
    """A question containing "S80" finds the S80 node by string overlap and
    measures nothing. This is the trap the first version of the generator fell
    into, so it is pinned here."""
    graph, table, _, rows = small
    questions = planted_questions(graph, table, rows, n_per_kind=4)
    codeonly = [q for q in questions if q.qid.startswith("dx-codeonly")]
    assert codeonly, "no code-only questions generated"
    for q in codeonly:
        node = int(q.answer_nodes[0])
        display = table.at[node, "display_text"]
        code = display.split("10-GM-")[1].split(";")[0].split()[-1]
        assert code.lower() not in q.text.lower(), f"question leaks the code {code}"


def test_named_diagnoses_are_reachable_and_code_only_ones_are_not(small):
    """The gap between these two is the terminology cross-walk's target, and the
    experiment is only meaningful if the gap starts wide."""
    graph, table, _, rows = small
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = [
        q
        for q in planted_questions(graph, table, rows, n_per_kind=4)
        if q.qid.startswith("dx-")
    ]
    results = evaluate(
        graph,
        [PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable")],
        questions,
        encoder,
    )
    results = results.assign(kind=results["qid"].str.replace(r"-\d+$", "", regex=True))
    means = results.groupby("kind")["answer_node_recall"].mean()
    assert means["dx-named"] == pytest.approx(1.0)
    assert means["dx-codeonly"] == pytest.approx(0.0)


def test_every_retriever_returns_a_valid_selection(small):
    graph, table, _, rows = small
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = planted_questions(graph, table, rows, n_per_kind=2)
    q_emb = encoder.encode_one(questions[0].text)
    for retriever in (
        PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable"),
        TopKTriples(k=10, encoder=encoder),
    ):
        assert_valid(retriever.retrieve(graph, q_emb), graph)


def test_kaping_needs_its_encoder_to_be_a_fair_baseline(small):
    """Without an encoder TopKTriples ranks on relation text alone. With six
    relation types every triple of a type ties, so it degenerates to "the first
    k edges by index". Pinning this stops a sweep from silently understating the
    baseline, which is how the first version of sweep_replica.py was wrong."""
    graph, table, _, rows = small
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = planted_questions(graph, table, rows, n_per_kind=4)
    crippled = evaluate(graph, [TopKTriples(k=10)], questions, encoder)
    fair = evaluate(graph, [TopKTriples(k=10, encoder=encoder)], questions, encoder)
    assert fair["answer_node_recall"].mean() > crippled["answer_node_recall"].mean()
