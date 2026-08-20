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


def test_every_diagnosis_has_a_distinct_label():
    """With labels repeated, a planted question's gold answer was always the
    lowest-indexed member of its tied group -- which is exactly what
    tie_break="stable" selects, so PCST scored 1.000 by construction while a
    retriever indexing in another order scored 0.000. Uniqueness removes the
    coupling; real ties are measured by the analyte questions instead."""
    from ikgqa.data.replica import generate_rows

    for profile in (PATIENT_0_SMALL, PATIENT_0):
        rows = generate_rows(profile, seed=0)
        labels = [r["_true_name"] for r in rows["diagnoses"]]
        assert len(set(labels)) == len(labels), f"{profile.label} repeats labels"


def test_the_two_language_vocabularies_share_no_token():
    """The cross-language condition is only a language gap if the two label sets
    have no token in common. "Stadium 3" and "stage 3" shared the token "3",
    which was enough for a lexical encoder to bridge the gap and make the
    experiment report a gain that was not there."""
    from ikgqa.data import replica as R
    from ikgqa.encoders import tokenize

    de = {t for phrase in R._DE_STEMS + R._DE_QUALIFIERS + R._DE_EXTRA for t in tokenize(phrase)}
    en = {t for phrase in R._EN_STEMS + R._EN_QUALIFIERS + R._EN_EXTRA for t in tokenize(phrase)}
    assert not (de & en), f"shared tokens leak across languages: {sorted(de & en)}"


def test_code_only_questions_span_the_resolution_tiers(small):
    """synthetic_catalogue assigns tiers by position, so sampling the first n
    code-only rows would report the coverage of the best-covered codes as if it
    were the coverage of all of them."""
    from ikgqa.data.replica import synthetic_catalogue
    from ikgqa.data.terminology import TIER_NONE, enrich_diagnosis_rows

    graph, table, _, rows = small
    catalogue = synthetic_catalogue(rows, language="de")
    questions = planted_questions(graph, table, rows, n_per_kind=20)
    asked = {q.text for q in questions if q.qid.startswith("dx-codeonly")}
    assert asked, "no code-only questions"

    codeonly = [r for r in rows["diagnoses"] if not r.get("diagnosis_name")]
    enriched, _ = enrich_diagnosis_rows(codeonly, catalogue)
    tiers_asked = {
        r["resolution_tier"] for r in enriched if str(r["_true_name"]).lower() in asked
    }
    assert len(tiers_asked) > 1, f"questions only cover tier(s) {tiers_asked}"
    assert TIER_NONE in tiers_asked, "the unresolvable tail must be asked about too"


def test_text_scoring_baselines_see_the_same_text_as_the_embeddings(small):
    """PCST scores node_emb (built from embed text) while KAPING scores triple
    text. If the latter came from display text the two would be ranking
    different graphs, and a change to embed text -- such as resolving a code --
    would be invisible to the baseline."""
    graph, table, _, _ = small
    triples = graph.triple_texts()
    embed_sample = str(table["embed_text"].iloc[0])
    display_only = str(table["display_text"].iloc[0])
    joined = " ".join(triples)
    assert embed_sample in joined
    # A timestamp appears in display text and never in embed text, so it is a
    # reliable marker that display text has not leaked into the ranking input.
    assert "recorded:" not in joined and "observed" not in joined
    assert display_only not in joined


def test_enrichment_changes_what_the_baseline_can_rank(small):
    """The end-to-end guarantee: resolving a code must alter the text a
    text-scoring retriever sees, or the cross-walk cannot help it."""
    from ikgqa.data.replica import build_replica, synthetic_catalogue

    _, _, _, rows = small
    catalogue = synthetic_catalogue(rows, language="de")
    plain, _, _, _ = build_replica(PATIENT_0_SMALL, seed=0)
    enriched, _, _, _ = build_replica(PATIENT_0_SMALL, seed=0, terminology=catalogue)
    assert set(plain.embed_texts) != set(enriched.embed_texts)
    assert " ".join(plain.triple_texts()) != " ".join(enriched.triple_texts())


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
