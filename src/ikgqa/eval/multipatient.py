"""
ikgqa.eval.multipatient
=======================

Running the evaluation across many patients instead of one.

Everything measured so far in this project came from a single patient graph.
That is enough to establish mechanism and nothing else: patient #0 may be
atypical in size, in how much of its text is shared, or in how many diagnoses
carry a description, and each of those drives a reported result. Scaling to the
cohort is what turns "this is how the method behaves" into "this is how the
method behaves here".

Three things make the difference between a loop that finishes and one that does
not, and none of them is the retrieval:

**One patient's failure must not end the run.** Loading a graph can fail for
reasons that have nothing to do with retrieval: a patient with no diagnoses, a
timeout, a gold answer the loader no longer emits. A run of 1,197 patients that
aborts on the fortieth has produced nothing. Failures are caught, recorded with
their reason, and reported as counts beside the results.

**It must be resumable.** Loading and encoding dominate the cost, and a sweep
over the cohort is measured in hours. Rows are appended as they are produced and
completed work is skipped on restart, so an interrupted run resumes instead of
starting again.

**Per-patient results must stay separate.** Pooling every question into one mean
lets a handful of large patients dominate, and hides the variance that is itself
a finding. Rows carry their patient, and ``summarise`` reports the spread across
patients rather than only the average.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ikgqa.eval.metrics import evaluate

#: Reasons a patient contributed no rows. Counted and reported, never silent.
SKIP_NO_QUESTIONS = "no questions"
SKIP_LOAD_FAILED = "load failed"
SKIP_EMPTY_GRAPH = "empty graph"


@dataclasses.dataclass(frozen=True)
class PatientCase:
    """What a loader must return for one patient."""

    graph: Any
    questions: Sequence[Any]
    encoder: Any


@dataclasses.dataclass
class RunReport:
    """What happened, as opposed to what was measured."""

    attempted: int = 0
    evaluated: int = 0
    skipped: Dict[str, int] = dataclasses.field(default_factory=dict)
    seconds: float = 0.0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def failed(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> str:
        parts = [f"{self.evaluated}/{self.attempted} patients evaluated",
                 f"{self.seconds:.1f}s"]
        for reason, count in sorted(self.skipped.items()):
            parts.append(f"{count} {reason}")
        return ", ".join(parts)


def completed_patients(path) -> set:
    """Patients already present in a partial results file.

    Read before a run so an interrupted sweep resumes. Returns an empty set if
    the file does not exist or cannot be parsed, because a damaged partial file
    should cost a recomputation, never a crash.
    """
    import pathlib

    target = pathlib.Path(path)
    if not target.exists():
        return set()
    try:
        frame = pd.read_csv(target)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError):
        return set()
    if "patient" not in frame.columns:
        return set()
    return {str(p) for p in frame["patient"].unique()}


def run(
    patients: Iterable[str],
    load_patient: Callable[[str], Optional[PatientCase]],
    retrievers: Any,
    resume_from: Optional[str] = None,
    on_rows: Optional[Callable[[pd.DataFrame], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, RunReport]:
    """Evaluate every retriever on every question, for every patient.

    Args:
        patients: patient identifiers.
        load_patient: returns a PatientCase, or None if this patient has nothing
            to contribute. Raising is also allowed and is caught.
        retrievers: anything implementing the Retriever protocol, or a callable
            taking the PatientCase and returning them. The callable form exists
            because some baselines have to be given the patient's own encoder to
            be scored fairly: KAPING ranks triple text, and without the encoder
            it silently falls back to relation text alone, which understates it.
            A single encoder cannot be shared when each patient has its own.
        resume_from: a partial results CSV whose patients are skipped.
        on_rows: called with each patient's rows as they are produced, so a long
            run can append to disk rather than holding everything in memory.
        progress: called with a one-line status per patient.

    Returns the concatenated rows and a report of what was skipped.
    """
    report = RunReport()
    done = completed_patients(resume_from) if resume_from else set()
    collected: List[pd.DataFrame] = []
    started = time.time()

    for patient in patients:
        patient = str(patient)
        if patient in done:
            continue
        report.attempted += 1

        try:
            case = load_patient(patient)
        except Exception as exc:                     # noqa: BLE001 - reported, not raised
            report.skip(SKIP_LOAD_FAILED)
            if progress:
                progress(f"{patient}: load failed ({type(exc).__name__}: {exc})")
            continue

        if case is None or case.graph is None or case.graph.num_nodes == 0:
            report.skip(SKIP_EMPTY_GRAPH)
            if progress:
                progress(f"{patient}: empty graph")
            continue
        if not case.questions:
            report.skip(SKIP_NO_QUESTIONS)
            if progress:
                progress(f"{patient}: no questions")
            continue

        try:
            built = retrievers(case) if callable(retrievers) else retrievers
            rows = evaluate(case.graph, built, case.questions, case.encoder)
        except Exception as exc:                     # noqa: BLE001
            report.skip(f"evaluation failed: {type(exc).__name__}")
            if progress:
                progress(f"{patient}: evaluation failed ({exc})")
            continue

        rows = rows.assign(patient=patient)
        report.evaluated += 1
        collected.append(rows)
        if on_rows:
            on_rows(rows)
        if progress:
            progress(f"{patient}: {len(rows)} rows, {case.graph.num_nodes} nodes")

    report.seconds = round(time.time() - started, 1)
    frame = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    return frame, report


def summarise(rows: pd.DataFrame, by: Sequence[str] = ("retriever",)) -> pd.DataFrame:
    """Mean recall and its spread across patients.

    The per-patient standard deviation is reported alongside the mean because a
    single average over pooled questions hides whether a method works everywhere
    or works spectacularly on a few large patients. ``patients`` is the number
    contributing, so a mean over three of them cannot be mistaken for a cohort
    result.
    """
    if rows.empty:
        return pd.DataFrame()

    by = list(by)
    per_patient = (rows.groupby(by + ["patient"], sort=False)["answer_node_recall"]
                   .mean().reset_index())

    summary = (per_patient.groupby(by, sort=False)["answer_node_recall"]
               .agg(recall_mean="mean", recall_sd="std", patients="count")
               .reset_index())

    scored = (rows.groupby(by, sort=False)["answer_node_recall"]
              .apply(lambda s: int(s.notna().sum())).reset_index(name="questions_scored"))

    return summary.merge(scored, on=by, how="left")
