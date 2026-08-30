"""
ikgqa.eval.goldset
==================

The human-validated question set, its format, and the checks that stop a bad
entry from quietly corrupting a result.

    python -m ikgqa.eval.goldset template > gold/questions.jsonl
    python -m ikgqa.eval.goldset check gold/questions.jsonl

**The file is data, not code.** A gold question names a real patient and real
clinical entities, so ``gold/`` is excluded from version control and the file
stays on BioMedIT infrastructure. What is committed is this module and the
synthetic fixtures in the tests. Anyone reproducing the thesis reproduces the
tooling; the answers are governed.

**Answers are recorded as stable keys, never as node indices.** A node's index
is assigned by whatever order the loader happened to walk the database, and it
moves when the loader changes, when a limit changes, or when the data does. A
gold answer stored as index 4{,}812 would keep scoring after it had begun
pointing at a different node, and nothing would report an error. Keys come from
the SPHN uid, so ``resolve`` can say plainly that an answer no longer exists
rather than silently scoring the wrong thing.

**Validation is not a formality.** Every check here corresponds to a way a
question set has already gone wrong in this project or would score something
other than retrieval:

* an entity question with no answers scores zero against every method;
* an aggregate question with answers gets scored as though recall meant
  something for it;
* a question that quotes the code it is looking for is answerable by string
  overlap and measures nothing (this one was caught in the planted question
  generator, where it made PCST look perfect);
* a question with no validator has another system's output as its ground truth,
  so agreement with that system is not evidence of correctness.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ikgqa.eval.metrics import KIND_AGGREGATE, KIND_ENTITY, Question

#: Question languages the pipeline knows how to encode. Mixing them silently is
#: the single most expensive mistake available here: a catalogue in the wrong
#: language took named-diagnosis recall from 1.00 to 0.00.
LANGUAGES = ("de", "en")

KINDS = (KIND_ENTITY, KIND_AGGREGATE)

ERROR, WARNING = "error", "warning"

FIELDS = ("qid", "text", "language", "kind", "patient", "answer_keys",
          "validated_by", "validated_on", "source", "notes", "skip_reason")


@dataclasses.dataclass(frozen=True)
class Problem:
    """Something wrong with one question, or with the set as a whole."""

    qid: str
    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity:7s}] {self.qid or '(set)'}: {self.message}"


@dataclasses.dataclass(frozen=True)
class GoldQuestion:
    """One validated question.

    Args:
        qid: stable identifier, used to join results across runs.
        text: the question as a user would ask it.
        language: one of LANGUAGES. Decides which encoder and which terminology
            catalogue are correct, so it is required rather than inferred.
        kind: KIND_ENTITY or KIND_AGGREGATE.
        patient: the patient the question is scoped to. Empty means cohort-wide,
            which the per-patient metric cannot score.
        answer_keys: stable node keys constituting a correct answer.
        validated_by: who confirmed the answers. Empty means unvalidated.
        validated_on: ISO date of that confirmation.
        source: where the question came from, such as a user log or an author.
        notes: free text for the validator.
        skip_reason: set to exclude the question from scoring with a stated
            reason, which is reported rather than counted as a failure.
    """

    qid: str
    text: str
    language: str = "de"
    kind: str = KIND_ENTITY
    patient: str = ""
    answer_keys: Tuple[str, ...] = ()
    validated_by: str = ""
    validated_on: str = ""
    source: str = ""
    notes: str = ""
    skip_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_keys", tuple(str(k) for k in self.answer_keys))

    @property
    def validated(self) -> bool:
        return bool(self.validated_by)

    @property
    def scoreable(self) -> bool:
        """Can answer-node recall mean anything for this question?"""
        return (self.kind == KIND_ENTITY
                and bool(self.answer_keys)
                and not self.skip_reason)

    def to_dict(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["answer_keys"] = list(self.answer_keys)
        return record

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "GoldQuestion":
        unknown = set(record) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown field(s) {sorted(unknown)} in {record.get('qid')!r}")
        known = {k: v for k, v in record.items() if k in FIELDS}
        known["answer_keys"] = tuple(known.get("answer_keys") or ())
        return cls(**known)


def _quotes_its_own_answer(question: GoldQuestion) -> Optional[str]:
    """Does the question text contain the identifier it is meant to find?

    A question like "which patients have N18.5" retrieved against a graph whose
    node text is "N18.5" is answered by string overlap alone. It scores well for
    every method and measures nothing about retrieval.
    """
    text = question.text.casefold()
    for key in question.answer_keys:
        identifier = key.split(":", 1)[-1].strip()
        if len(identifier) >= 3 and identifier.casefold() in text:
            return identifier
    return None


@dataclasses.dataclass(frozen=True)
class GoldSet:
    """A collection of gold questions, with its checks."""

    questions: Tuple[GoldQuestion, ...] = ()

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    # -- persistence ------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path) -> "GoldSet":
        """One JSON object per line.

        JSON Lines rather than CSV because answers are a list, and a list in a
        CSV cell needs an escaping convention that someone eventually gets
        wrong by hand. One line per question also means a malformed entry names
        its own line number.
        """
        questions = []
        for number, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                questions.append(GoldQuestion.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}, line {number}: {exc}") from exc
        return cls(tuple(questions))

    def to_jsonl(self, path) -> None:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(q.to_dict(), ensure_ascii=False) for q in self.questions]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- checks -----------------------------------------------------------

    def validate(self) -> List[Problem]:
        """Everything wrong with the set, worst first."""
        problems: List[Problem] = []

        seen: Dict[str, int] = {}
        for question in self.questions:
            if not question.qid:
                problems.append(Problem("", ERROR, "question has no qid"))
            seen[question.qid] = seen.get(question.qid, 0) + 1

        for qid, count in seen.items():
            if count > 1 and qid:
                problems.append(Problem(qid, ERROR, f"qid appears {count} times"))

        for question in self.questions:
            qid = question.qid
            if not question.text.strip():
                problems.append(Problem(qid, ERROR, "question text is empty"))
            if question.language not in LANGUAGES:
                problems.append(Problem(qid, ERROR,
                                        f"language {question.language!r} not in {LANGUAGES}"))
            if question.kind not in KINDS:
                problems.append(Problem(qid, ERROR,
                                        f"kind {question.kind!r} not in {KINDS}"))

            if question.kind == KIND_ENTITY and not question.answer_keys and not question.skip_reason:
                problems.append(Problem(qid, ERROR,
                                        "entity question has no answer_keys; it would score "
                                        "zero against every method"))
            if question.kind == KIND_AGGREGATE and question.answer_keys:
                problems.append(Problem(qid, ERROR,
                                        "aggregate question carries answer_keys; recall is "
                                        "undefined for it and would be scored anyway"))

            if len(set(question.answer_keys)) != len(question.answer_keys):
                problems.append(Problem(qid, WARNING, "answer_keys contains duplicates"))

            leaked = _quotes_its_own_answer(question)
            if leaked:
                problems.append(Problem(qid, WARNING,
                                        f"text contains {leaked!r}, which is part of its own "
                                        "answer key; it is answerable by string overlap"))

            if not question.validated and not question.skip_reason:
                problems.append(Problem(qid, WARNING,
                                        "no validated_by; its ground truth is another "
                                        "system's output"))

        problems.sort(key=lambda p: (p.severity != ERROR, p.qid))
        return problems

    # -- use --------------------------------------------------------------

    def resolve(self, graph, patient: str = "") -> Tuple[List[Question], List[Problem]]:
        """Turn stable keys into the node indices this graph happens to use.

        Returns the scoreable questions alongside any answer key that the graph
        does not contain. A missing key is reported, never dropped: it means the
        loader no longer emits a node the gold set was built against, and
        scoring around it would understate every method equally and invisibly.
        """
        if "node_key" not in graph.nodes.columns:
            raise ValueError(
                "graph has no node_key column, so gold answers cannot be resolved; "
                "rebuild it with a loader that records stable keys")

        index_of = {str(key): int(idx) for idx, key
                    in zip(graph.nodes["node_id"], graph.nodes["node_key"])}

        resolved: List[Question] = []
        problems: List[Problem] = []
        for question in self.questions:
            if patient and question.patient and question.patient != patient:
                continue
            if not question.scoreable:
                continue

            found, missing = [], []
            for key in question.answer_keys:
                (found if key in index_of else missing).append(key)
            if missing:
                problems.append(Problem(question.qid, ERROR,
                                        f"{len(missing)} answer key(s) not in this graph: "
                                        f"{', '.join(sorted(missing)[:3])}"))
            if not found:
                continue

            resolved.append(Question(
                text=question.text,
                answer_nodes=tuple(index_of[key] for key in found),
                qid=question.qid,
                kind=question.kind,
                validated=question.validated,
            ))
        return resolved, problems

    def stats(self) -> Dict[str, Any]:
        """A summary to print beside any result computed from this set."""
        kinds: Dict[str, int] = {}
        languages: Dict[str, int] = {}
        for question in self.questions:
            kinds[question.kind] = kinds.get(question.kind, 0) + 1
            languages[question.language] = languages.get(question.language, 0) + 1
        return {
            "questions": len(self.questions),
            "scoreable": sum(1 for q in self.questions if q.scoreable),
            "validated": sum(1 for q in self.questions if q.validated),
            "patients": len({q.patient for q in self.questions if q.patient}),
            "by_kind": dict(sorted(kinds.items())),
            "by_language": dict(sorted(languages.items())),
        }


TEMPLATE = [
    GoldQuestion(
        qid="q001",
        text="Welche Medikamente hat die Patientin nach der Transplantation erhalten?",
        language="de",
        kind=KIND_ENTITY,
        patient="PATIENT-PLACEHOLDER",
        answer_keys=("drug:ATC-PLACEHOLDER-1", "drug:ATC-PLACEHOLDER-2"),
        validated_by="NAME OF THE CLINICIAN OR DATA MANAGER",
        validated_on="2026-09-01",
        source="user log",
        notes="Replace every PLACEHOLDER. Keys come from the node_key column.",
    ),
    GoldQuestion(
        qid="q002",
        text="Wie viele Patienten hatten 2023 eine Abstossungsreaktion?",
        language="de",
        kind=KIND_AGGREGATE,
        patient="",
        answer_keys=(),
        validated_by="NAME OF THE CLINICIAN OR DATA MANAGER",
        validated_on="2026-09-01",
        source="user log",
        notes="A count exists nowhere in the graph, so this is recorded and reported, not scored.",
    ),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gold question set tooling.")
    parser.add_argument("action", choices=("check", "template"))
    parser.add_argument("path", nargs="?", default="gold/questions.jsonl")
    args = parser.parse_args(argv)

    if args.action == "template":
        for question in TEMPLATE:
            print(json.dumps(question.to_dict(), ensure_ascii=False))
        return 0

    gold = GoldSet.from_jsonl(args.path)
    problems = gold.validate()

    for key, value in gold.stats().items():
        print(f"  {key:14s} {value}")

    errors = [p for p in problems if p.severity == ERROR]
    print(f"\n{len(problems)} problem(s), {len(errors)} of them errors")
    for problem in problems:
        print(f"  {problem}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
