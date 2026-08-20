"""
ikgqa.eval.sweep
================

The thesis's headline artifact: a recall-versus-size curve.

A single recall number is not interpretable, because any retriever can reach
recall 1.0 by returning more. The comparison that means something is: *at the
same subgraph size, who recalls more answer entities?* So each retriever is run
across its own size dial, and the resulting points are compared on a shared
size axis.

The honest way to compare dials that are not the same quantity -- PCST's
per-edge cost against a baseline's k -- is to plot both against a measured size
(nodes, or prompt characters) rather than against their own parameter. That is
what sweep() produces, and why the size columns matter as much as the recall
column.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ikgqa.eval.metrics import Question, evaluate, summarise


@dataclasses.dataclass(frozen=True)
class Sweep:
    """One retriever family swept over one parameter.

    Args:
        factory: value -> configured retriever, e.g. ``lambda c: PCST(cost_e=c)``.
        values: the parameter values to try, in the order to plot them.
        dial: the parameter name, recorded in the output for the legend.
    """

    factory: Callable[[Any], Any]
    values: Sequence[Any]
    dial: str

    def retrievers(self) -> list:
        return [self.factory(v) for v in self.values]


def sweep(
    graph: Any,
    sweeps: Iterable[Sweep],
    questions: Sequence[Question],
    encoder: Any,
    q_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate every configuration of every sweep. One row per config.

    Question embeddings are computed once and reused across all configurations,
    which is where most of the runtime saving comes from in a large sweep.

    Returns:
        A frame with one row per configuration, carrying dial/dial_value plus
        the averaged metrics from summarise(). Sorted by retriever then by mean
        node count, which is the order a curve should be drawn in.
    """
    sweeps = list(sweeps)
    if not sweeps:
        raise ValueError("no sweeps given")

    if q_embeddings is None:
        q_embeddings = encoder.encode([q.text for q in questions])

    frames = []
    for spec in sweeps:
        for value, retriever in zip(spec.values, spec.retrievers()):
            rows = evaluate(graph, [retriever], questions, encoder, q_embeddings=q_embeddings)
            agg = summarise(rows).reset_index()
            agg.insert(0, "dial", spec.dial)
            agg.insert(1, "dial_value", value)
            agg.insert(0, "retriever", retriever.name)
            frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["retriever", "nodes"], kind="stable").reset_index(drop=True)


def to_curve(swept: pd.DataFrame, size_axis: str = "nodes") -> pd.DataFrame:
    """Reduce a sweep frame to the (size, recall) pairs a plot needs.

    Args:
        swept: output of sweep().
        size_axis: "nodes", "edges", "chars" or "node_fraction". Use "chars"
            when comparing retrievers whose dials differ, since it measures
            what the LLM actually pays for.
    """
    allowed = {"nodes", "edges", "chars", "node_fraction"}
    if size_axis not in allowed:
        raise ValueError(f"size_axis must be one of {sorted(allowed)}, got {size_axis!r}")
    cols = ["retriever", "config", "dial", "dial_value", size_axis, "answer_node_recall", "n_scored"]
    return swept[cols].rename(columns={size_axis: "size"})
