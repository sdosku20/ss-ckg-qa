"""
ikgqa.retrieval.pcst
====================

The Retriever wrapper around G-Retriever's PCST selection.

All the algorithm lives in ikgqa.pcst.core, which is held byte-equivalent to
the published implementation and tested against a verbatim copy of it. This
module only adapts it to the uniform retriever interface, so the equivalence
guarantee is never at risk from interface churn.

Two parameters are worth understanding before reading any result table:

  * cost_e is the real size dial. Higher cost, tighter subgraph. It is the
    knob the recall-vs-size curve sweeps.
  * topk / topk_e set how many nodes/edges receive a non-zero prize at all.
    They bound what *can* be reached, so sweeping cost_e with topk fixed at 3
    explores only a narrow band. The thesis compares against baselines whose
    dial is k, so keep in mind the two dials are not the same quantity: this is
    exactly the comparability caveat to state in the Methodology chapter.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ikgqa.pcst import core as P
from ikgqa.retrieval.base import BaseRetriever, Retrieval


def retrieve_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=5, cost_e=0.5):
    """Functional form, kept for the playground and for direct comparisons."""
    r = P.retrieval_via_pcst_traced(
        graph, q_emb, textual_nodes, textual_edges, topk=topk, topk_e=topk_e, cost_e=cost_e
    )
    return r.selected_nodes, r.selected_edges


@dataclasses.dataclass(frozen=True)
class PCST(BaseRetriever):
    """Prize-Collecting Steiner Tree retrieval (He et al. 2024, Section 5.3).

    Args:
        topk: nodes given a non-zero prize (prize k down to 1 by rank).
        topk_e: edge prize tiers.
        cost_e: per-edge cost. The size/relevance trade-off.
        tie_break: "auto" matches upstream exactly when torch is installed;
            "stable" is deterministic without torch. See core._topk_indices.
    """

    topk: int = 3
    topk_e: int = 5
    cost_e: float = 0.5
    tie_break: str = "auto"

    @property
    def name(self) -> str:
        return "PCST"

    @property
    def size_dial(self) -> str:
        return "cost_e"

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        if not hasattr(graph, "nodes"):
            raise TypeError(
                "PCST needs the text tables as well as the embeddings, so it takes a "
                "TextualGraph rather than a bare SimpleGraph "
                f"(got {type(graph).__name__})."
            )
        result = P.retrieval_via_pcst_traced(
            graph.as_simple_graph(),
            q_emb,
            graph.nodes,
            graph.edges,
            topk=self.topk,
            topk_e=self.topk_e,
            cost_e=self.cost_e,
            tie_break=self.tie_break,
        )
        # Upstream returns edge ids in *solver* order, not sorted: decode_solution
        # concatenates the solver's real edges with the edges recovered from
        # virtual nodes and never re-sorts. np.unique sorts and de-duplicates, so
        # the Retrieval contract holds and every retriever is comparable.
        #
        # Deliberately done here rather than in ikgqa.pcst.core: the core stays
        # byte-identical to the published implementation, including the row order
        # of the textual description it produces.
        return Retrieval(
            np.unique(np.asarray(result.selected_nodes, dtype=np.int64)),
            np.unique(np.asarray(result.selected_edges, dtype=np.int64)),
        )

    def trace(self, graph: Any, q_emb: np.ndarray):
        """Full RetrievalTrace, for inspecting prizes and the solver instance.

        Same computation as retrieve(), but returns every intermediate value.
        Use it when a result looks wrong: the prize table almost always
        explains why.
        """
        return P.retrieval_via_pcst_traced(
            graph.as_simple_graph(),
            q_emb,
            graph.nodes,
            graph.edges,
            topk=self.topk,
            topk_e=self.topk_e,
            cost_e=self.cost_e,
            tie_break=self.tie_break,
        )
