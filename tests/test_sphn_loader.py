"""
Tests for the SPHN loader, using fixture rows only -- no database, no clinical data.

The rows below have the exact column names the module's Cypher returns, so the
assembly logic is covered end to end. What is *not* covered here is whether the
Cypher itself matches the live schema; that is verified by running it on the
server, which is why fetch and build are separate functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ikgqa.data.sphn import (
    Neo4jSettings,
    build_patient_graph,
    diagnosis_text,
    lab_observation_text,
    lab_test_type_text,
)
from ikgqa.encoders import BagOfWordsEncoder
from ikgqa.retrieval import PCST, assert_valid


def fixture_rows():
    """Two analytes measured twice each, one drug, two diagnoses, two cases.

    Shaped after the real ratios: repeated measurements of few analytes, codes
    sometimes missing a name (the enrichment gap).
    """
    return {
        "cases": [
            {"case_id": "case-1", "admitted_at": "2019-04-10T08:00:00", "discharged_at": "2019-04-20T11:00:00"},
            {"case_id": "case-2", "admitted_at": "2020-01-05T09:30:00", "discharged_at": "2020-01-07T14:00:00"},
        ],
        "labs": [
            {"event_uid": "ev-1", "observed_at": "2019-04-11T07:00:00", "loinc_code": "2160-0",
             "code_system": "LOINC", "test_name": "Creatinine [Mass/volume] in Serum or Plasma",
             "numeric_value": 2.1, "unit": "mg/dL", "text_value": None,
             "reference_range": "0.6-1.2", "case_id": "case-1", "material": "Serum"},
            {"event_uid": "ev-2", "observed_at": "2019-04-14T07:00:00", "loinc_code": "2160-0",
             "code_system": "LOINC", "test_name": "Creatinine [Mass/volume] in Serum or Plasma",
             "numeric_value": 1.8, "unit": "mg/dL", "text_value": None,
             "reference_range": "0.6-1.2", "case_id": "case-1", "material": "Serum"},
            {"event_uid": "ev-3", "observed_at": "2020-01-06T06:30:00", "loinc_code": "1975-2",
             "code_system": "LOINC", "test_name": "Bilirubin.total [Mass/volume] in Serum or Plasma",
             "numeric_value": 0.4, "unit": "mg/dL", "text_value": None,
             "reference_range": None, "case_id": "case-2", "material": "Serum"},
            # No name: only a bare code. Counts toward the enrichment gap.
            {"event_uid": "ev-4", "observed_at": "2020-01-06T06:30:00", "loinc_code": "17861-6",
             "code_system": "LOINC", "test_name": None, "numeric_value": 9.1, "unit": "mg/dL",
             "text_value": None, "reference_range": None, "case_id": "case-2", "material": None},
        ],
        "diagnoses": [
            {"recorded_at": "2019-04-12T00:00:00", "icd_code": "N18.5", "code_system": "10-GM-2022",
             "diagnosis_name": "Chronische Nierenkrankheit, Stadium 5", "case_id": "case-1"},
            # No name: the 56% of diagnosis uses that need ICD-10-GM enrichment.
            {"recorded_at": "2020-01-06T00:00:00", "icd_code": "B18.2", "code_system": "10-GM-2018",
             "diagnosis_name": None, "case_id": "case-2"},
        ],
        "drugs": [
            {"started_at": "2019-04-11T12:00:00", "ended_at": "2019-04-19T12:00:00", "drug_key": 101,
             "substance": "tacrolimus", "article": "Prograf 1mg", "dose_value": 5.0,
             "dose_unit": "mg", "case_id": "case-1"},
            {"started_at": "2019-04-12T12:00:00", "ended_at": "2019-04-19T12:00:00", "drug_key": 101,
             "substance": "tacrolimus", "article": "Prograf 1mg", "dose_value": 5.0,
             "dose_unit": "mg", "case_id": "case-1"},
        ],
    }


@pytest.fixture
def built():
    rows = fixture_rows()
    corpus = [str(v) for group in rows.values() for row in group for v in row.values()]
    encoder = BagOfWordsEncoder(corpus)
    return build_patient_graph(rows, encoder, patient_index=0)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_shared_entities_are_deduplicated(built):
    """Two measurements of creatinine share one analyte node; two doses of
    tacrolimus share one drug node. That sharing *is* the graph structure."""
    _, table, stats = built
    assert stats.labs == 4
    assert stats.lab_types == 3          # 2160-0, 1975-2, 17861-6
    assert stats.drug_administrations == 2
    assert stats.drugs == 1              # both administrations, one drug
    assert (table["sphn_label"] == "LabTestType").sum() == 3
    assert (table["sphn_label"] == "Drug").sum() == 1


def test_graph_is_not_a_star(built):
    """If everything hung off the patient, connectivity would be vacuous and
    PCST would have nothing to decide."""
    graph, table, _ = built
    patient = int(table.index[table["sphn_label"] == "SubjectPseudoIdentifier"][0])
    ei = graph.edge_index
    non_patient_edges = [
        i for i in range(graph.num_edges)
        if patient not in (int(ei[0, i]), int(ei[1, i]))
    ]
    assert len(non_patient_edges) > 0, "no structure beyond the patient hub"


def test_observations_are_linked_to_their_hospital_case(built):
    """The case node is what ties labs, diagnoses and drugs of one stay."""
    graph, table, _ = built
    cases = set(table.index[table["sphn_label"] == "AdministrativeCase"])
    ei = graph.edge_index
    touching = {
        int(ei[0, i]) for i in range(graph.num_edges) if int(ei[1, i]) in cases
    }
    labels = set(table.loc[sorted(touching), "sphn_label"])
    assert {"LabObservation", "BilledDiagnosis", "DrugAdministrationEvent"} <= labels


def test_a_case_referenced_but_not_listed_is_still_created(built):
    """Observations may reference a case the case query did not return."""
    rows = fixture_rows()
    rows["cases"] = []                        # nothing listed
    encoder = BagOfWordsEncoder(["x"])
    _, table, stats = build_patient_graph(rows, encoder)
    assert stats.cases == 2                   # recovered from the observations
    assert (table["sphn_label"] == "AdministrativeCase").sum() == 2


def test_empty_patient_produces_a_valid_single_node_graph():
    """A patient with min_degree 4 exists in the real data; must not crash."""
    encoder = BagOfWordsEncoder(["patient subject"])
    graph, table, stats = build_patient_graph(
        {"labs": [], "diagnoses": [], "drugs": [], "cases": []}, encoder
    )
    assert graph.num_nodes == 1 and graph.num_edges == 0
    assert stats.labs == 0


# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------


def test_embed_text_excludes_values_so_it_deduplicates(built):
    """The measured 9,000-fold saving depends on this: two creatinine results
    must share one embed_text while keeping distinct display_text."""
    _, table, stats = built
    obs = table[table["sphn_label"] == "LabObservation"]
    creatinine = obs[obs["display_text"].str.contains("Creatinine")]
    assert len(creatinine) == 2
    assert creatinine["embed_text"].nunique() == 1
    assert creatinine["display_text"].nunique() == 2
    assert stats.distinct_embed_texts < len(table)


def test_display_text_carries_the_value_the_llm_needs(built):
    _, table, _ = built
    obs = table[table["display_text"].str.contains("Creatinine")]
    assert any("2.1" in t and "mg/dL" in t for t in obs["display_text"])
    assert any("0.6-1.2" in t for t in obs["display_text"])


def test_codes_without_names_are_counted_not_hidden(built):
    """These nodes cannot be found by similarity. The count is the size of the
    terminology-enrichment gap, and it belongs in the results."""
    _, _, stats = built
    assert stats.unnamed_labs == 1
    assert stats.unnamed_diagnoses == 1
    assert "UNNAMED" in stats.summary()


def test_a_nameless_code_still_produces_usable_text():
    embed, display = lab_test_type_text(
        {"test_name": None, "loinc_code": "17861-6", "code_system": "LOINC"}
    )
    assert "17861-6" in embed and "LOINC" in display


def test_missing_everything_does_not_produce_the_string_none():
    """A null rendered as "None" would poison the embedding space."""
    embed, display = lab_observation_text(
        {"test_name": None, "loinc_code": None, "numeric_value": None, "unit": None}
    )
    assert "none" not in embed.lower() and "none" not in display.lower()
    embed, display = diagnosis_text({"diagnosis_name": None, "icd_code": None})
    assert "none" not in embed.lower() and "none" not in display.lower()


def test_diagnosis_text_keeps_the_coding_system_version(built):
    """ICD-10-GM is versioned by year (10-GM-2022 vs 10-GM-2018); the same code
    can mean different things across versions."""
    _, table, _ = built
    dx = table[table["sphn_label"] == "BilledDiagnosis"]
    assert any("10-GM-2022" in t for t in dx["display_text"])


# ---------------------------------------------------------------------------
# Interoperability with the retrievers
# ---------------------------------------------------------------------------


def test_the_loaded_graph_works_with_every_retriever(built):
    graph, table, _ = built
    encoder = BagOfWordsEncoder(list(table["embed_text"]) + ["creatinine kidney"])
    # Rebuild in the encoder's space: the question and the graph must share one.
    rows = fixture_rows()
    graph, table, _ = build_patient_graph(rows, encoder)
    q = encoder.encode_one("creatinine lab result")

    # topk_e=0 on purpose: with few, heavily shared relation types the edge-prize
    # mechanism collapses cost_e to ~0. See experiments/synthetic_clinical.py.
    selection = PCST(topk=3, topk_e=0, cost_e=0.5).retrieve(graph, q)
    assert_valid(selection, graph)
    assert selection.num_nodes >= 1


def test_pcst_finds_the_creatinine_observations(built):
    rows = fixture_rows()
    table_texts = ["creatinine", "bilirubin", "tacrolimus", "nierenkrankheit"]
    corpus = [str(v) for g in rows.values() for r in g for v in r.values()] + table_texts
    encoder = BagOfWordsEncoder(corpus)
    graph, table, _ = build_patient_graph(rows, encoder)

    q = encoder.encode_one("creatinine")
    selection = PCST(topk=3, topk_e=0, cost_e=0.5).retrieve(graph, q)
    got = set(table.loc[selection.node_ids, "display_text"])
    assert any("Creatinine" in t for t in got)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_error_names_the_missing_variables_and_the_docker_trap(monkeypatch):
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="source ~/.ikgqa.env"):
        Neo4jSettings.from_env()


def test_settings_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", '"quoted-password"')
    s = Neo4jSettings.from_env()
    assert s.database == "neo4j"
    assert s.password.startswith('"'), "quote characters in the password must survive"
