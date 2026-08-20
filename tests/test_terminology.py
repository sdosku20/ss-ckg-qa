"""
Tests for the terminology cross-walk.

The tiers are the point of this module: a single "coverage" percentage hides
whether a description came from the code's own catalogue entry, from a different
year of the same classification, or from a parent concept that is broader than
the node deserves. These tests pin each tier's behaviour and the counting that
reports it.
"""

from __future__ import annotations

import pytest

from ikgqa.data.terminology import (
    TIER_EXACT,
    TIER_LOCAL,
    TIER_NONE,
    TIER_PARENT,
    TIER_VERSION,
    Concept,
    Terminology,
    bootstrap_from_rows,
    enrich_diagnosis_rows,
    parent_code_of,
    system_family,
)


def catalogue():
    return Terminology(
        [
            Concept("10-GM-2022", "N18.5", "Chronic kidney disease, stage 5", "N18",
                    "Chronic kidney disease", "en"),
            Concept("10-GM-2022", "N18", "Chronic kidney disease", language="en"),
            Concept("10-GM-2019", "B18.2", "Chronic viral hepatitis C", language="en"),
            Concept("LOINC", "2160-0", "Creatinine [Mass/volume] in Serum", language="en"),
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_system_family_strips_only_a_four_digit_year():
    assert system_family("10-GM-2022") == "10-GM"
    assert system_family("LOINC") == "LOINC"
    assert system_family("ATC") == "ATC"
    # Not a year: must be left alone, or unrelated systems would be merged.
    assert system_family("SNOMED-CT") == "SNOMED-CT"
    assert system_family("ICD-10") == "ICD-10"


def test_parent_code_is_the_prefix_before_the_dot():
    assert parent_code_of("N18.5") == "N18"
    assert parent_code_of("N18") == ""
    assert parent_code_of("") == ""


# ---------------------------------------------------------------------------
# The tiers
# ---------------------------------------------------------------------------


def test_exact_match_wins():
    got = catalogue().resolve("N18.5", "10-GM-2022")
    assert got.tier == TIER_EXACT
    assert got.label == "Chronic kidney disease, stage 5"
    assert got.parent_label == "Chronic kidney disease"
    assert got.resolved and got.reachable


def test_a_different_year_of_the_same_system_resolves_as_version():
    """ICD-10-GM is reissued annually and most codes are stable across years,
    so this is a good trade -- but it is an assumption, which is why it is its
    own tier and gets counted separately."""
    got = catalogue().resolve("B18.2", "10-GM-2024")
    assert got.tier == TIER_VERSION
    assert got.label == "Chronic viral hepatitis C"
    assert got.source_system == "10-GM-2019", "must record which entry answered"


def test_a_parent_concept_resolves_as_parent_and_is_flagged_as_such():
    """N18.9 is not catalogued, but N18 is. The label is broader than the node
    deserves, so the tier has to be visible in the results."""
    got = catalogue().resolve("N18.9", "10-GM-2022")
    assert got.tier == TIER_PARENT
    assert got.label == "Chronic kidney disease"
    assert got.resolved


def test_the_graphs_own_label_is_kept_when_the_catalogue_has_nothing():
    got = catalogue().resolve("Z99.9", "10-GM-2022", local_label="Sonstige Zustaende")
    assert got.tier == TIER_LOCAL
    assert got.label == "Sonstige Zustaende"
    assert not got.resolved, "a local label is not a catalogue resolution"
    assert got.reachable, "but the node does have words on it"


def test_nothing_anywhere_is_reported_as_unreachable():
    got = catalogue().resolve("Z99.9", "10-GM-2022")
    assert got.tier == TIER_NONE
    assert got.label == ""
    assert not got.reachable


def test_a_local_label_is_never_overwritten_by_a_worse_tier():
    """The graph's own label beats nothing, but a catalogue entry beats the
    graph's label -- otherwise enrichment could not change anything."""
    term = catalogue()
    assert term.resolve("N18.5", "10-GM-2022", local_label="Nierenkrankheit").tier == TIER_EXACT
    assert term.resolve("Q00.0", "10-GM-2022", local_label="Nierenkrankheit").tier == TIER_LOCAL


def test_systems_do_not_leak_into_each_other():
    """A LOINC code must not resolve against an ICD entry just because the
    strings happen to collide."""
    term = Terminology(
        [
            Concept("LOINC", "2160-0", "Creatinine", language="en"),
            Concept("10-GM-2022", "2160-0", "Something unrelated", language="de"),
        ]
    )
    assert term.resolve("2160-0", "LOINC").label == "Creatinine"
    assert term.resolve("2160-0", "10-GM-2022").label == "Something unrelated"


# ---------------------------------------------------------------------------
# Construction and persistence
# ---------------------------------------------------------------------------


def test_csv_roundtrip_preserves_every_field(tmp_path):
    path = tmp_path / "catalogue.csv"
    written = catalogue().to_csv(path)
    assert written == 4
    reloaded = Terminology.from_csv(path)
    assert len(reloaded) == 4
    got = reloaded.resolve("N18.5", "10-GM-2022")
    assert got.tier == TIER_EXACT
    assert got.parent_label == "Chronic kidney disease"
    assert got.language == "en"


def test_csv_missing_a_required_column_says_which(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("system,code\nLOINC,2160-0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        Terminology.from_csv(path)


def test_bootstrap_from_the_graphs_own_labels_needs_no_download():
    """The no-download floor: a code labelled under one year fills the same code
    appearing bare under another. Nothing external, nothing to trust."""
    rows = {
        "diagnoses": [
            {"icd_code": "N18.5", "code_system": "10-GM-2022",
             "diagnosis_name": "Chronische Nierenkrankheit, Stadium 5"},
            {"icd_code": "N18.5", "code_system": "10-GM-2018", "diagnosis_name": None},
        ]
    }
    term = bootstrap_from_rows(rows)
    assert len(term) == 1
    got = term.resolve("N18.5", "10-GM-2018")
    assert got.tier == TIER_VERSION
    assert got.label == "Chronische Nierenkrankheit, Stadium 5"


def test_bootstrap_ignores_null_shaped_labels():
    rows = {
        "diagnoses": [
            {"icd_code": "A00", "code_system": "10-GM-2020", "diagnosis_name": "None"},
            {"icd_code": "A01", "code_system": "10-GM-2020", "diagnosis_name": "  "},
        ]
    }
    assert len(bootstrap_from_rows(rows)) == 0


# ---------------------------------------------------------------------------
# Applying it to rows
# ---------------------------------------------------------------------------


def diagnosis_rows():
    return [
        {"icd_code": "N18.5", "code_system": "10-GM-2022", "diagnosis_name": None},
        {"icd_code": "B18.2", "code_system": "10-GM-2024", "diagnosis_name": None},
        {"icd_code": "N18.9", "code_system": "10-GM-2022", "diagnosis_name": None},
        {"icd_code": "Q10.1", "code_system": "10-GM-2022", "diagnosis_name": "Ein Befund"},
        {"icd_code": "Z99.9", "code_system": "10-GM-2022", "diagnosis_name": None},
    ]


def test_enrichment_counts_every_tier():
    _, report = enrich_diagnosis_rows(diagnosis_rows(), catalogue())
    assert report.tiers == {
        TIER_EXACT: 1,
        TIER_VERSION: 1,
        TIER_PARENT: 1,
        TIER_LOCAL: 1,
        TIER_NONE: 1,
    }
    assert report.total == 5
    assert report.reachable == 4
    assert report.resolved_by_catalogue == 3
    assert "reachable by similarity: 4/5" in report.summary()


def test_enrichment_does_not_modify_its_input():
    """The database is read-only and the source rows are evidence. Enrichment
    derives a text layer; it must not rewrite what was fetched."""
    rows = diagnosis_rows()
    before = [dict(r) for r in rows]
    enrich_diagnosis_rows(rows, catalogue())
    assert rows == before


def test_display_keeps_the_original_label_while_embed_gets_the_resolved_one():
    """Two audiences: the encoder needs canonical text, a clinician needs the
    wording their own system recorded."""
    from ikgqa.data.sphn import diagnosis_text

    enriched, _ = enrich_diagnosis_rows(diagnosis_rows(), catalogue())
    resolved = enriched[0]
    embed, display = diagnosis_text(resolved)
    assert "Chronic kidney disease, stage 5" in embed
    assert "N18.5" in display and "Chronic kidney disease" not in display

    kept = enriched[3]  # had a German label already
    embed, display = diagnosis_text(kept)
    assert "Ein Befund" in display


def test_without_enrichment_the_composer_is_unchanged():
    """The un-enriched run is the baseline every gain is measured against, so it
    must not shift when this module is added."""
    from ikgqa.data.sphn import diagnosis_text

    row = diagnosis_rows()[0]
    assert diagnosis_text(row) == diagnosis_text(dict(row))
    embed, _ = diagnosis_text(row)
    assert embed == "diagnosis: N18.5"


def test_hierarchy_expansion_adds_the_parents_words_without_inventing_any():
    plain, _ = enrich_diagnosis_rows(diagnosis_rows(), catalogue())
    wide, report = enrich_diagnosis_rows(
        diagnosis_rows(), catalogue(), expand_hierarchy=True
    )
    assert report.hierarchy_expanded >= 1
    assert plain[0]["embed_name"] == "Chronic kidney disease, stage 5"
    assert wide[0]["embed_name"] == (
        "Chronic kidney disease, stage 5; Chronic kidney disease"
    )


def test_an_empty_catalogue_changes_nothing_but_still_reports():
    rows = diagnosis_rows()
    enriched, report = enrich_diagnosis_rows(rows, Terminology())
    assert report.tiers[TIER_LOCAL] == 1
    assert report.tiers[TIER_NONE] == 4
    assert all(r["embed_name"] in ("", "Ein Befund") for r in enriched)
