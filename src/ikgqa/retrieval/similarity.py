"""
ikgqa.retrieval.similarity
==========================

The two similarity-only baselines from G-Retriever Appendix D.1, plus their
class wrappers.

Reported there on WebQSP (k=5, llama2-7b-chat, Table 10):

    PCST retrieval                   Hit@1  66.17
    top-k triples (KAPING)           Hit@1  52.64
    top-k nodes plus its neighbours  Hit@1  49.82
    shortest path retrieval          Hit@1  55.20

Those are end-to-end QA numbers and need an LLM. What this package measures
instead is retrieval quality, which needs no model; see ikgqa.eval.metrics.

The paper describes each baseline in one sentence and the released code does
not include them, so two reading choices are made explicit here:

  * KAPING scores whole *triples*, so retrieve_topk_triples takes a triple_emb
    matrix built from concatenated head/relation/tail text
    (TextualGraph.triple_texts). Falling back to relation text alone would
    handicap the baseline unfairly.
  * "top-k nodes plus neighbours" is read as: top-k nodes, then every incident
    edge, then those edges' endpoints.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

from ikgqa.pcst import core as P
from ikgqa.retrieval.base import BaseRetriever, Retrieval


def retrieve_topk_triples(graph: Any, q_emb: Any, k: int = 5, triple_emb: Optional[Any] = None):
    """Rank triples by similarity to the question, keep the best k.

    The simplest thing that could work, and the one most RAG-over-KG systems
    do. Its weakness is structural: the k triples need not touch each other, so
    the LLM gets a bag of disconnected facts and no path between them.
    """
    emb = triple_emb if triple_emb is not None else graph.edge_attr
    sim = P.cosine_similarity(q_emb, emb)
    k = min(k, sim.shape[0])
    edges = np.sort(np.argsort(-sim, kind="stable")[:k])
    ei = np.asarray(P._to_numpy(graph.edge_index, dtype=np.int64))
    nodes = np.unique(ei[:, edges]) if edges.size else np.array([], dtype=np.int64)
    return nodes, edges


def retrieve_topk_nodes_plus_neighbors(graph: Any, q_emb: Any, k: int = 5):
    """Take the k best nodes, then everything one hop away.

    Captures local context, but the amount of it is dictated by node degree
    rather than by the question. On a hub node this explodes; on a leaf it
    returns almost nothing. There is no size dial.
    """
    sim = P.cosine_similarity(q_emb, graph.x)
    k = min(k, sim.shape[0])
    seeds = set(np.argsort(-sim, kind="stable")[:k].tolist())

    ei = np.asarray(P._to_numpy(graph.edge_index, dtype=np.int64))
    edges = [i for i in range(ei.shape[1]) if int(ei[0, i]) in seeds or int(ei[1, i]) in seeds]
    edges = np.asarray(sorted(edges), dtype=np.int64)
    nodes = np.unique(
        np.concatenate([np.asarray(sorted(seeds), dtype=np.int64), ei[:, edges].reshape(-1)])
        if edges.size
        else np.asarray(sorted(seeds), dtype=np.int64)
    )
    return nodes, edges


@dataclasses.dataclass(frozen=True)
class TopKTriples(BaseRetriever):
    """KAPING-style: the k most similar triples, connected or not.

    Args:
        k: how many triples to keep.
        encoder: used to embed "head relation tail" strings. Optional but
            recommended: without it the ranking falls back to relation text
            alone, which is a materially weaker baseline than KAPING as
            published, and understating a baseline is as much a result error as
            overstating your own method.

    Triple embeddings are cached per graph, since a sweep calls retrieve() once
    per question over the same graph and re-encoding every time would dominate
    the runtime.
    """

    k: int = 5
    encoder: Optional[Any] = None
    _triple_cache: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "top-k triples (KAPING)"

    @property
    def size_dial(self) -> str:
        return "k"

    def _triple_emb(self, graph: Any) -> Optional[np.ndarray]:
        if self.encoder is None or not hasattr(graph, "triple_texts"):
            return None
        key = id(graph)
        hit = self._triple_cache.get(key)
        if hit is None:
            # The graph itself is stored alongside the embedding so its id
            # cannot be recycled by the garbage collector while cached.
            hit = (graph, self.encoder.encode(graph.triple_texts()))
            self._triple_cache[key] = hit
        return hit[1]

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        nodes, edges = retrieve_topk_triples(
            _as_simple(graph), q_emb, k=self.k, triple_emb=self._triple_emb(graph)
        )
        return Retrieval(nodes, edges)


@dataclasses.dataclass(frozen=True)
class TopKNodesPlusNeighbors(BaseRetriever):
    """The k most similar nodes plus their one-hop neighbourhood.

    size_dial is None on purpose: k controls how many *seeds* are taken, but
    the size of the result is set by their degree. On a hub node, raising k by
    one can double the subgraph. That is the failure mode described in thesis
    Section 1.3.
    """

    k: int = 5

    @property
    def name(self) -> str:
        return "top-k nodes + neighbours"

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        nodes, edges = retrieve_topk_nodes_plus_neighbors(_as_simple(graph), q_emb, k=self.k)
        return Retrieval(nodes, edges)


def _as_simple(graph: Any) -> Any:
    """Accept a TextualGraph or anything already shaped like SimpleGraph."""
    return graph.as_simple_graph() if hasattr(graph, "as_simple_graph") else graph
