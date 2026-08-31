"""
The gold question set and the multi-patient loop.

Both exist to stop a plausible-looking number from being wrong. The gold set's
checks each correspond to a way a question set has already gone wrong here or
would measure something other than retrieval; the loop's tests are about what
happens when a patient fails, which on a cohort of 1,197 is not hypothetical.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ikgqa.eval.goldset import (
    ERROR,
    WARNING,
    GoldQuestion,
    GoldSet,
    Problem,
)
from ikgqa.eval.metrics import KIND_AGGREGATE, KIND_ENTITY
from ikgqa.eval.multipatient import (
    PatientCase,
    RunReport,
    completed_patients,
    run,
    summarise,
)


def good(qid="q1", **overrides):
    fields = dict(
        qid=qid,
        text="Welche Medikamente hat die Patientin erhalten?",
        language="de",
        kind=KIND_ENTITY,
        patient="p1",
        answer_keys=("drug:aaa", "drug:bbb"),
        validated_by="Dr Example",
        validated_on="2026-09-01",
    )
    fields.update(overrides)
    return GoldQuestion(**fields)


def severities(problems, severity):
    return [p for p in problems if p.severity == severity]


# --- format and round trip --------------------------------------------------

def test_a_set_survives_a_round_trip_through_jsonl(tmp_path):
    original = GoldSet((good("q1"), good("q2", language="en")))
    path = tmp_path / "gold" / "questions.jsonl"
    original.to_jsonl(path)

    assert GoldSet.from_jsonl(path).questions == original.questions


def test_unicode_is_preserved_because_the_questions_are_german(tmp_path):
    question = good(text="Welche Abstossungsreaktion trat auf? Grösse?")
    path = tmp_path / "q.jsonl"
    GoldSet((question,)).to_jsonl(path)

    assert "Abstossungsreaktion" in path.read_text(encoding="utf-8")
    assert GoldSet.from_jsonl(path).questions[0].text == question.text


def test_a_malformed_line_names_its_line_number(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"qid": "q1", "text": "ok"}\nnot json at all\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        GoldSet.from_jsonl(path)


def test_an_unknown_field_is_refused_rather_than_ignored(tmp_path):
    # A typo in a hand-edited file would otherwise be silently dropped, and the
    # question would score with a default nobody chose.
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"qid": "q1", "text": "x", "anwser_keys": ["a"]}) + "\n",
                    encoding="utf-8")

    with pytest.raises(ValueError, match="anwser_keys"):
        GoldSet.from_jsonl(path)


def test_blank_lines_and_comments_are_skipped(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("// notes for the validator\n\n"
                    + json.dumps(good().to_dict()) + "\n", encoding="utf-8")

    assert len(GoldSet.from_jsonl(path)) == 1


# --- the checks -------------------------------------------------------------

def test_a_clean_set_has_no_errors():
    assert severities(GoldSet((good("q1"), good("q2"))).validate(), ERROR) == []


def test_an_entity_question_without_answers_is_an_error():
    problems = GoldSet((good(answer_keys=()),)).validate()
    assert any("no answer_keys" in p.message for p in severities(problems, ERROR))


def test_an_aggregate_question_with_answers_is_an_error():
    # Recall is undefined for a count. Carrying answer keys means it would be
    # scored as though it were not.
    problems = GoldSet((good(kind=KIND_AGGREGATE, answer_keys=("drug:a",)),)).validate()
    assert any("undefined" in p.message for p in severities(problems, ERROR))


def test_an_aggregate_question_without_answers_is_fine():
    assert severities(GoldSet((good(kind=KIND_AGGREGATE, answer_keys=()),)).validate(),
                      ERROR) == []


def test_a_question_quoting_its_own_answer_key_is_flagged():
    # This exact defect made PCST score 1.000 in the planted question generator.
    problems = GoldSet((good(text="Welche Patienten haben N18.5?",
                             answer_keys=("dx:N18.5",)),)).validate()
    assert any("string overlap" in p.message for p in severities(problems, WARNING))


def test_a_short_key_fragment_does_not_trigger_a_false_positive():
    # A two-character identifier appears inside ordinary words constantly.
    problems = GoldSet((good(text="Welche Medikamente?", answer_keys=("dx:me",)),)).validate()
    assert not any("string overlap" in p.message for p in problems)


def test_duplicate_qids_are_an_error():
    problems = GoldSet((good("same"), good("same"))).validate()
    assert any("appears 2 times" in p.message for p in severities(problems, ERROR))


def test_an_unvalidated_question_is_warned_about_not_rejected():
    # Using a production system's output as ground truth is circular, but such
    # questions are still useful in bulk; they must be reported separately.
    problems = GoldSet((good(validated_by=""),)).validate()
    assert severities(problems, ERROR) == []
    assert any("another system" in p.message for p in severities(problems, WARNING))


def test_an_unknown_language_is_an_error():
    problems = GoldSet((good(language="fr"),)).validate()
    assert any("language" in p.message for p in severities(problems, ERROR))


def test_errors_sort_before_warnings():
    problems = GoldSet((good("q1", validated_by=""), good("q2", language="fr"))).validate()
    assert problems[0].severity == ERROR


def test_a_skipped_question_is_not_scoreable_and_is_not_nagged_about():
    question = good(answer_keys=(), skip_reason="answer not in the graph")
    assert not question.scoreable
    assert GoldSet((question,)).validate() == []


# --- resolving keys to this graph's indices ---------------------------------

class FakeGraph:
    def __init__(self, keys):
        self.nodes = pd.DataFrame({"node_id": range(len(keys)), "node_key": list(keys)})
        self.num_nodes = len(keys)


def test_keys_resolve_to_the_indices_this_graph_uses():
    graph = FakeGraph(["drug:zzz", "drug:aaa", "drug:bbb"])
    resolved, problems = GoldSet((good(),)).resolve(graph)

    assert problems == []
    assert sorted(resolved[0].answer_nodes) == [1, 2], "aaa and bbb, not 0 and 1"


def test_the_same_keys_resolve_differently_when_the_graph_is_rebuilt():
    # The whole reason answers are keys and not indices: node order is not stable
    # across loads, so an index recorded today points elsewhere tomorrow.
    first = FakeGraph(["drug:aaa", "drug:bbb"])
    second = FakeGraph(["something:else", "drug:aaa", "drug:bbb"])

    a, _ = GoldSet((good(),)).resolve(first)
    b, _ = GoldSet((good(),)).resolve(second)
    assert sorted(a[0].answer_nodes) == [0, 1]
    assert sorted(b[0].answer_nodes) == [1, 2]


def test_a_missing_answer_key_is_reported_not_silently_dropped():
    graph = FakeGraph(["drug:aaa"])
    resolved, problems = GoldSet((good(),)).resolve(graph)

    assert any("not in this graph" in p.message for p in problems)
    assert resolved[0].answer_nodes == (0,), "what was found is still scored"


def test_a_question_whose_answers_are_all_missing_produces_no_row():
    graph = FakeGraph(["unrelated:node"])
    resolved, problems = GoldSet((good(),)).resolve(graph)

    assert resolved == []
    assert severities(problems, ERROR)


def test_resolving_against_a_graph_without_keys_fails_loudly():
    class Keyless:
        nodes = pd.DataFrame({"node_id": [0], "node_attr": ["x"]})

    with pytest.raises(ValueError, match="node_key"):
        GoldSet((good(),)).resolve(Keyless())


def test_resolution_can_be_scoped_to_one_patient():
    graph = FakeGraph(["drug:aaa", "drug:bbb"])
    gold = GoldSet((good("q1", patient="p1"), good("q2", patient="p2")))

    resolved, _ = gold.resolve(graph, patient="p1")
    assert [q.qid for q in resolved] == ["q1"]


def test_stats_report_what_a_result_rests_on():
    gold = GoldSet((good("q1"), good("q2", language="en", validated_by=""),
                    good("q3", kind=KIND_AGGREGATE, answer_keys=())))
    stats = gold.stats()

    assert stats["questions"] == 3
    assert stats["scoreable"] == 2
    assert stats["validated"] == 2
    assert stats["by_language"] == {"de": 2, "en": 1}


# --- the multi-patient loop -------------------------------------------------

def rows_for(patient, recall):
    return pd.DataFrame({"retriever": ["PCST"], "answer_node_recall": [recall],
                         "nodes": [11], "qid": ["q1"]})


class StubGraph:
    num_nodes = 10


def test_every_patient_contributes_rows_tagged_with_its_own_id(monkeypatch):
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    frame, report = run(["p1", "p2"],
                        lambda p: PatientCase(StubGraph(), ["q"], object()),
                        retrievers=[])

    assert sorted(frame["patient"].unique()) == ["p1", "p2"]
    assert report.evaluated == 2 and report.failed == 0


def test_one_patient_failing_does_not_end_the_run(monkeypatch):
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    def load(patient):
        if patient == "p2":
            raise TimeoutError("database went away")
        return PatientCase(StubGraph(), ["q"], object())

    frame, report = run(["p1", "p2", "p3"], load, retrievers=[])

    assert sorted(frame["patient"].unique()) == ["p1", "p3"]
    assert report.evaluated == 2
    assert report.skipped["load failed"] == 1
    assert "load failed" in report.summary()


def test_an_empty_graph_is_recorded_with_a_reason(monkeypatch):
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    class Empty:
        num_nodes = 0

    _, report = run(["p1"], lambda p: PatientCase(Empty(), ["q"], object()), retrievers=[])
    assert report.skipped["empty graph"] == 1


def test_a_patient_with_no_questions_is_skipped_not_scored_zero(monkeypatch):
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    _, report = run(["p1"], lambda p: PatientCase(StubGraph(), [], object()), retrievers=[])
    assert report.skipped["no questions"] == 1


def test_an_evaluation_error_is_caught_and_named(monkeypatch):
    import ikgqa.eval.multipatient as mp

    def explode(*a, **k):
        raise ValueError("encoder mismatch")

    monkeypatch.setattr(mp, "evaluate", explode)
    _, report = run(["p1"], lambda p: PatientCase(StubGraph(), ["q"], object()), retrievers=[])
    assert any("ValueError" in reason for reason in report.skipped)


def test_a_run_resumes_instead_of_repeating_finished_patients(tmp_path, monkeypatch):
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    partial = tmp_path / "partial.csv"
    pd.DataFrame({"patient": ["p1"], "answer_node_recall": [0.5]}).to_csv(partial, index=False)

    seen = []

    def load(patient):
        seen.append(patient)
        return PatientCase(StubGraph(), ["q"], object())

    _, report = run(["p1", "p2"], load, retrievers=[], resume_from=str(partial))

    assert seen == ["p2"], "p1 was already done"
    assert report.attempted == 1


def test_a_damaged_partial_file_costs_a_recomputation_not_a_crash(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text('patient,recall\n"unterminated\n', encoding="utf-8")
    assert completed_patients(broken) == set()
    assert completed_patients(tmp_path / "absent.csv") == set()


def test_rows_are_handed_over_as_they_are_produced(monkeypatch):
    # So a cohort-scale run can append to disk rather than hold everything.
    import ikgqa.eval.multipatient as mp
    monkeypatch.setattr(mp, "evaluate", lambda *a, **k: rows_for("x", 0.5))

    batches = []
    run(["p1", "p2"], lambda p: PatientCase(StubGraph(), ["q"], object()),
        retrievers=[], on_rows=batches.append)

    assert len(batches) == 2, "one call per patient, not one at the end"


# --- summarising ------------------------------------------------------------

def test_the_spread_across_patients_is_reported_beside_the_mean():
    rows = pd.DataFrame({
        "retriever": ["PCST"] * 4,
        "patient": ["p1", "p1", "p2", "p2"],
        "answer_node_recall": [1.0, 1.0, 0.0, 0.0],
    })
    summary = summarise(rows)

    assert summary["recall_mean"].iloc[0] == pytest.approx(0.5)
    assert summary["recall_sd"].iloc[0] > 0, "a method that works on one patient only"
    assert summary["patients"].iloc[0] == 2


def test_each_patient_counts_once_however_many_questions_it_has():
    # Otherwise a large patient with many questions dominates the cohort mean.
    rows = pd.DataFrame({
        "retriever": ["PCST"] * 5,
        "patient": ["big"] * 4 + ["small"],
        "answer_node_recall": [1.0, 1.0, 1.0, 1.0, 0.0],
    })
    assert summarise(rows)["recall_mean"].iloc[0] == pytest.approx(0.5)


def test_unscored_questions_do_not_count_towards_the_scored_total():
    rows = pd.DataFrame({
        "retriever": ["PCST"] * 3,
        "patient": ["p1"] * 3,
        "answer_node_recall": [1.0, float("nan"), 0.0],
    })
    assert summarise(rows)["questions_scored"].iloc[0] == 2


def test_summarising_nothing_returns_an_empty_frame():
    assert summarise(pd.DataFrame()).empty


def test_the_report_reads_as_a_sentence():
    report = RunReport(attempted=10, evaluated=8, seconds=12.5)
    report.skip("load failed")
    report.skip("load failed")
    assert "8/10 patients evaluated" in report.summary()
    assert "2 load failed" in report.summary()


def test_a_problem_prints_its_severity_and_question():
    assert "q7" in str(Problem("q7", ERROR, "something"))


# --- patients of different sizes --------------------------------------------

def test_scaling_a_profile_preserves_its_shape_and_changes_its_size():
    from ikgqa.data.replica import PATIENT_0

    half = PATIENT_0.scaled(0.5)
    assert half.lab_observations == round(PATIENT_0.lab_observations * 0.5)
    assert half.analytes == round(PATIENT_0.analytes * 0.5)
    assert half.measurements_per_analyte == pytest.approx(
        PATIENT_0.measurements_per_analyte, rel=0.02), "the ratio is the shape"


def test_a_scaled_profile_says_what_it_is():
    from ikgqa.data.replica import PATIENT_0
    assert "0.5" in PATIENT_0.scaled(0.5).label
    assert PATIENT_0.scaled(0.5, label="custom").label == "custom"


def test_unnamed_diagnoses_never_exceed_diagnoses_after_scaling():
    from ikgqa.data.replica import PATIENT_0
    for factor in (0.01, 0.4, 1.0, 1.2):
        scaled = PATIENT_0.scaled(factor)
        assert scaled.unnamed_diagnoses <= scaled.diagnoses


def test_scaling_never_produces_an_empty_patient():
    from ikgqa.data.replica import PATIENT_0
    tiny = PATIENT_0.scaled(0.0001)
    assert tiny.analytes >= 1 and tiny.cases >= 1


def test_a_nonpositive_factor_is_refused():
    from ikgqa.data.replica import PATIENT_0
    with pytest.raises(ValueError):
        PATIENT_0.scaled(0)


# --- encodings a hand-edited file actually arrives in -----------------------

@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le"])
def test_a_gold_file_is_readable_whatever_windows_wrote_it_as(tmp_path, encoding):
    # PowerShell's ">" writes UTF-16 with a BOM; Notepad and Excel write UTF-8
    # with one. Insisting on plain UTF-8 fails on byte zero with a message that
    # tells a clinician nothing.
    path = tmp_path / "q.jsonl"
    body = json.dumps(good().to_dict(), ensure_ascii=False) + "\n"
    path.write_bytes(body.encode(encoding))

    assert len(GoldSet.from_jsonl(path)) == 1


def test_german_text_survives_a_utf16_round_trip(tmp_path):
    path = tmp_path / "q.jsonl"
    question = good(text="Grösse der Läsion?")
    path.write_bytes((json.dumps(question.to_dict(), ensure_ascii=False) + "\n").encode("utf-16"))

    assert GoldSet.from_jsonl(path).questions[0].text == "Grösse der Läsion?"


def test_the_template_can_be_written_directly_as_utf8(tmp_path):
    from ikgqa.eval.goldset import main

    out = tmp_path / "template.jsonl"
    assert main(["template", "--out", str(out)]) == 0
    assert out.read_bytes()[:1] == b"{", "no BOM, no UTF-16"
    assert len(GoldSet.from_jsonl(out)) == 2
