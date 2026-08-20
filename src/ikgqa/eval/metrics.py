"""
ikgqa.eval.metrics
==================

What counts as a good retrieval, measured without an LLM.

The thesis metric (Section 4.3) is *answer-node recall at a given subgraph
size*: a method scores well if the entities that constitute a correct answer
appear in its retrieved subgraph while the subgraph stays small. Reported as a
curve, not a single number, because returning the whole graph would trivially
achieve perfect recall.

Three deliberate design decisions, each guarding against a way the evaluation
could quietly lie:

1. Answer *nodes* are the primary target, edges secondary. It is tempting to
   score edge overlap because gold triples are easier to author, but the thesis
   research question is about answer entities, and the two can diverge: a
   retriever can return the answer node without the specific edge that
   justifies it. Both are reported; recall is the node one.

2. Aggregate questions are refused, not scored zero. "How many patients had
   graft rejection in 2023" has no answer node -- the count exists nowhere in
   the graph. Scoring it as recall 0 would silently punish every retriever and
   make the headline number meaningless. Question.kind marks these, and
   evaluate() records NaN plus a reason instead. If most of your production
   questions turn out to be aggregates, that is a finding to report, not a bug
   to patch.

3. Size is recorded three ways -- nodes, edges and prompt characters. They
   disagree: a retriever can return few nodes whose text is enormous. The
   characters column is what actually consumes the LLM context window, so it is
   the fairest size axis when comparing methods whose dials differ.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ikgqa.pcst import core as P
from ikgqa.retrieval.base import Retrieval

#: Questions whose answer is a set of graph entities. Only these get a recall.
KIND_ENTITY = "entity"
#: Questions whose answer is a count, mean or other derived value.
KIND_AGGREGATE = "aggregate"


@dataclasses.dataclass(frozen=True)
class Question:
    """One evaluation item.

    Args:
        text: the natural-language question, as a user asked it.
        answer_nodes: node ids that constitute a correct answer. For
            KIND_ENTITY questions this must be non-empty.
        answer_edges: optional gold triples, for the secondary edge metric.
        qid: stable identifier, used to join result tables across runs.
        kind: KIND_ENTITY or KIND_AGGREGATE (see module docstring).
        validated: True only for the human-checked gold set (thesis Section
            4.3, ~50 questions). Results over unvalidated questions must be
            reported separately -- their "ground truth" is another system's
            output, so agreement with it is not evidence of correctness.
    """

    text: str
    answer_nodes: tuple = ()
    answer_edges: tuple = ()
    qid: str = ""
    kind: str = KIND_ENTITY
    validated: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (KIND_ENTITY, KIND_AGGREGATE):
            raise ValueError(
                f"kind must be {KIND_ENTITY!r} or {KIND_AGGREGATE!r}, got {self.kind!r}"
            )
        if self.kind == KIND_ENTITY and not self.answer_nodes:
            raise ValueError(
                f"question {self.qid or self.text!r} is marked {KIND_ENTITY!r} but has no "
                f"answer_nodes; mark it {KIND_AGGREGATE!r} if its answer is a count or "
                "other derived value"
            )
        object.__setattr__(self, "answer_nodes", tuple(int(n) for n in self.answer_nodes))
        object.__setattr__(self, "answer_edges", tuple(int(e) for e in self.answer_edges))


# ---------------------------------------------------------------------------
# Single-selection measurements
# ---------------------------------------------------------------------------


def is_connected(num_nodes: int, edge_index: np.ndarray, nodes: np.ndarray, edges: np.ndarray) -> bool:
    """Is the retrieved node set connected using only the retrieved edges?"""
    nodes = np.asarray(nodes)
    edges = np.asarray(edges)
    if nodes.size <= 1:
        return True
    ei = np.asarray(edge_index)[:, edges] if edges.size else np.zeros((2, 0), dtype=np.int64)
    adj = {int(n): set() for n in nodes}
    for s, d in ei.T:
        adj[int(s)].add(int(d))
        adj[int(d)].add(int(s))
    start = int(nodes[0])
    seen, stack = {start}, [start]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == nodes.size


def recall(retrieved: Iterable[int], gold: Iterable[int]) -> float:
    """Fraction of gold ids that were retrieved. NaN when gold is empty.

    NaN rather than 0.0 or 1.0 on purpose: "no gold answer defined" is not the
    same as "retrieved nothing relevant", and pandas skips NaN when averaging,
    so an undefined item cannot drag a mean down.
    """
    gold_set = set(int(g) for g in gold)
    if not gold_set:
        return float("nan")
    got = set(int(r) for r in retrieved)
    return len(gold_set & got) / len(gold_set)


def precision(retrieved: Iterable[int], gold: Iterable[int]) -> float:
    """Fraction of retrieved ids that are gold. NaN when nothing was retrieved."""
    got = set(int(r) for r in retrieved)
    if not got:
        return float("nan")
    gold_set = set(int(g) for g in gold)
    return len(gold_set & got) / len(got)


def prompt_chars(graph: Any, selection: Retrieval) -> int:
    """Length of the textualised subgraph -- what fills the context window."""
    return len(
        P.build_description(
            graph.nodes,
            graph.edges,
            np.asarray(selection.node_ids, dtype=np.int64),
            np.asarray(selection.edge_ids, dtype=np.int64),
        )
    )


def measure(graph: Any, selection: Retrieval, question: Question) -> Dict[str, Any]:
    """All metrics for one (retriever, question) pair, as a flat row."""
    is_entity = question.kind == KIND_ENTITY
    return {
        "qid": question.qid or question.text,
        "question": question.text,
        "kind": question.kind,
        "validated": question.validated,
        "answer_node_recall": (
            recall(selection.node_ids, question.answer_nodes) if is_entity else float("nan")
        ),
        "skipped_reason": "" if is_entity else "aggregate question has no answer node",
        "edge_recall": recall(selection.edge_ids, question.answer_edges),
        "edge_precision": precision(selection.edge_ids, question.answer_edges),
        "nodes": selection.num_nodes,
        "edges": selection.num_edges,
        "chars": prompt_chars(graph, selection),
        "node_fraction": selection.num_nodes / max(graph.num_nodes, 1),
        "connected": is_connected(
            graph.num_nodes,
            graph.edge_index,
            np.asarray(selection.node_ids),
            np.asarray(selection.edge_ids),
        ),
    }


# ---------------------------------------------------------------------------
# Whole-experiment measurement
# ---------------------------------------------------------------------------


def evaluate(
    graph: Any,
    retrievers: Sequence[Any],
    questions: Sequence[Question],
    encoder: Any,
    q_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run every retriever on every question. One row per pair.

    Args:
        graph: a TextualGraph.
        retrievers: anything implementing the Retriever protocol.
        questions: the evaluation set.
        encoder: used to embed question text into the graph's space. Must be
            the same encoder the graph was built with, or similarities are
            meaningless -- there is no way to check this automatically, so it is
            the one thing to get right by hand.
        q_embeddings: optional precomputed [num_questions, d] matrix, to avoid
            re-encoding across a parameter sweep.

    Returns:
        A tidy DataFrame with one row per (retriever, question), carrying the
        retriever label and parameters alongside the metrics so the frame is
        self-describing when written to disk.
    """
    if not questions:
        raise ValueError("no questions to evaluate")

    if q_embeddings is None:
        q_embeddings = encoder.encode([q.text for q in questions])
    q_embeddings = np.asarray(q_embeddings, dtype=np.float32)
    if q_embeddings.shape[0] != len(questions):
        raise ValueError(
            f"{len(questions)} questions but {q_embeddings.shape[0]} question embeddings"
        )
    if q_embeddings.shape[-1] != graph.dim:
        raise ValueError(
            f"question embeddings have dim {q_embeddings.shape[-1]} but the graph has "
            f"dim {graph.dim}; the question encoder and the graph encoder must match"
        )

    rows = []
    for retriever in retrievers:
        params = getattr(retriever, "params", {})
        label = retriever.label() if hasattr(retriever, "label") else retriever.name
        for question, q_emb in zip(questions, q_embeddings):
            selection = retriever.retrieve(graph, q_emb)
            row = {"retriever": retriever.name, "config": label}
            row.update({f"param_{k}": v for k, v in params.items()})
            row.update(measure(graph, selection, question))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, by: str = "config") -> pd.DataFrame:
    """Average a result frame per retriever configuration.

    Means skip NaN, so aggregate questions do not distort answer_node_recall
    and questions without gold edges do not distort the edge columns. The
    n_scored column says how many questions actually contributed, which is the
    number to quote alongside any mean.
    """
    grouped = results.groupby(by, sort=False)
    out = grouped.agg(
        answer_node_recall=("answer_node_recall", "mean"),
        n_scored=("answer_node_recall", "count"),
        edge_recall=("edge_recall", "mean"),
        edge_precision=("edge_precision", "mean"),
        nodes=("nodes", "mean"),
        edges=("edges", "mean"),
        chars=("chars", "mean"),
        node_fraction=("node_fraction", "mean"),
        always_connected=("connected", "all"),
        n_questions=("qid", "nunique"),
    )
    return out.round(4)
