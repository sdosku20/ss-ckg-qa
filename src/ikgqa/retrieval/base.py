"""
ikgqa.retrieval.base
====================

The one interface every retriever implements.

A retriever answers exactly one question: *given a graph and an embedded
query, which nodes and edges should the LLM see?* It does not embed text, does
not score itself and does not talk to a model. Keeping it that narrow is what
lets the evaluation harness treat PCST and its baselines as interchangeable,
which is the entire premise of the comparison in the thesis.

Every retriever also exposes:

  * name   -- a stable label for result tables
  * params -- the knobs it was configured with, so a row in a results file can
              be traced back to an exact configuration
  * size_dial -- the name of the parameter that trades size against recall, or
              None if it has none. "Has no size dial" is a *finding*, not a
              gap: the recall-vs-size curve in the thesis needs one axis to
              sweep, and a retriever without one contributes a single point.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np


class Retrieval(NamedTuple):
    """What every retriever returns.

    Both arrays hold ids into the *original* graph, sorted ascending, and
    node_ids is closed under edge_ids: every endpoint of a returned edge is in
    the node set. Metrics rely on both invariants; assert_valid checks them.
    """

    node_ids: np.ndarray
    edge_ids: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(np.asarray(self.node_ids).size)

    @property
    def num_edges(self) -> int:
        return int(np.asarray(self.edge_ids).size)


@runtime_checkable
class Retriever(Protocol):
    """Structural type for retrievers. Implementations need not subclass."""

    name: str

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval: ...


@dataclasses.dataclass(frozen=True)
class BaseRetriever:
    """Shared plumbing: a name, a parameter dict and a size dial declaration."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def params(self) -> Dict[str, Any]:
        """The reportable knobs: scalar fields only.

        Collaborators (an encoder, a cache) are fields too, but printing them
        in a results table is noise and makes rows unstable across runs, so
        only str/int/float/bool/None fields are reported.
        """
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[f.name] = value
        return out

    @property
    def size_dial(self) -> Optional[str]:
        return None

    def label(self) -> str:
        """name plus parameters, e.g. "PCST(topk=3, cost_e=0.5)"."""
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({inner})"

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:  # pragma: no cover
        raise NotImplementedError


def assert_valid(retrieval: Retrieval, graph: Any) -> None:
    """Check the two invariants every retriever promises. Used in tests.

    Raises:
        AssertionError: with a message naming which invariant broke.
    """
    nodes = np.asarray(retrieval.node_ids, dtype=np.int64)
    edges = np.asarray(retrieval.edge_ids, dtype=np.int64)

    assert nodes.size == 0 or (nodes.min() >= 0 and nodes.max() < graph.num_nodes), (
        f"node ids out of range for a {graph.num_nodes}-node graph"
    )
    assert edges.size == 0 or (edges.min() >= 0 and edges.max() < graph.num_edges), (
        f"edge ids out of range for a {graph.num_edges}-edge graph"
    )
    assert np.all(np.diff(nodes) > 0) if nodes.size > 1 else True, "node ids not sorted/unique"
    assert np.all(np.diff(edges) > 0) if edges.size > 1 else True, "edge ids not sorted/unique"

    if edges.size:
        ei = np.asarray(graph.edge_index)[:, edges]
        missing = set(ei.reshape(-1).tolist()) - set(nodes.tolist())
        assert not missing, f"selected edges touch nodes {sorted(missing)} that are not returned"
