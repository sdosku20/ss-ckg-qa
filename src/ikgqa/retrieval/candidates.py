"""
ikgqa.retrieval.candidates
==========================

Two-stage retrieval: narrow the graph first, then select inside it.

Why this exists
---------------
G-Retriever assumes one modest graph *per question*. On WebQSP each sample
arrives with its own pre-extracted subgraph averaging 1,371 nodes. The graph
here is not like that: one STCS patient is 15,810 nodes and 45,562 edges after
graph preparation -- 11x a whole WebQSP sample -- and there are 1,197 patients.

So the published pipeline has a missing step. Something must produce the
question-specific region that PCST is then asked to optimise inside. In the
published benchmarks that step was done by the dataset. Here it has to be part
of the method, which makes it a methodological choice with consequences worth
measuring rather than an implementation detail.

What can go wrong, and why the metric still works
-------------------------------------------------
Narrowing can throw away an answer before the retriever ever sees it. That
failure is invisible if you only score the final subgraph, because a perfect
selector inside a bad candidate set still looks like a selector that missed. So
``TwoStage`` reports ``ceiling``: the recall the inner retriever *could* reach
given what stage one kept. A result where recall is low and ceiling is high is a
selection problem; low ceiling is a candidate-generation problem. They need
different fixes and must not be reported as one number.

Index spaces
------------
Stage one produces a smaller graph with its own 0..n-1 node ids. Every id that
leaves this module is translated back to the *original* graph's ids, because the
metric, the gold answers and every other retriever live in that space. This is
the one thing in the module that would silently corrupt results if it were
wrong, so it is tested directly rather than only through end-to-end recall.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd

from ikgqa.graph import TextualGraph
from ikgqa.pcst import core as P
from ikgqa.retrieval.base import BaseRetriever, Retrieval


@runtime_checkable
class CandidateGenerator(Protocol):
    """Stage one. Returns node ids of the original graph to keep."""

    name: str

    def select(self, graph: Any, q_emb: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Inducing a subgraph
# ---------------------------------------------------------------------------


def induce(graph: TextualGraph, keep: np.ndarray) -> Tuple[TextualGraph, np.ndarray, np.ndarray]:
    """Build the subgraph induced on ``keep``.

    An *induced* subgraph keeps an edge exactly when both endpoints survive.
    The alternative -- keeping edges with one endpoint inside and pulling the
    other in -- would quietly grow the candidate set beyond the size stage one
    was asked for, which is the failure mode that makes neighbourhood expansion
    hard to control.

    Args:
        graph: the full graph.
        keep: node ids to retain. Sorted and de-duplicated here, so callers need
            not.

    Returns:
        (subgraph, node_map, edge_map) where node_map[i] is the original id of
        sub-node i, and edge_map[j] the original id of sub-edge j. Both are what
        translate a result back into the original index space.
    """
    keep = np.unique(np.asarray(keep, dtype=np.int64))
    if keep.size and (keep.min() < 0 or keep.max() >= graph.num_nodes):
        raise ValueError(
            f"candidate node ids must lie in [0, {graph.num_nodes}), "
            f"got range [{keep.min()}, {keep.max()}]"
        )

    # position[original_id] -> new id, or -1 when dropped.
    position = np.full(graph.num_nodes, -1, dtype=np.int64)
    position[keep] = np.arange(keep.size, dtype=np.int64)

    src = graph.edges["src"].to_numpy(dtype=np.int64)
    dst = graph.edges["dst"].to_numpy(dtype=np.int64)
    edge_mask = (position[src] >= 0) & (position[dst] >= 0)
    edge_map = np.flatnonzero(edge_mask).astype(np.int64)

    nodes = pd.DataFrame({"node_id": np.arange(keep.size)})
    nodes["node_attr"] = graph.nodes["node_attr"].to_numpy()[keep]
    if "embed_attr" in graph.nodes.columns:
        nodes["embed_attr"] = graph.nodes["embed_attr"].to_numpy()[keep]

    edges = pd.DataFrame(
        {
            "src": position[src[edge_mask]],
            "edge_attr": graph.edges["edge_attr"].to_numpy()[edge_mask],
            "dst": position[dst[edge_mask]],
        }
    )

    edge_emb = graph.edge_emb[edge_map] if graph.edge_emb.size else graph.edge_emb
    if edge_map.size == 0:
        edge_emb = np.zeros((0, graph.node_emb.shape[1]), dtype=np.float32)

    sub = TextualGraph(
        nodes=nodes,
        edges=edges,
        node_emb=graph.node_emb[keep],
        edge_emb=edge_emb,
        name=f"{graph.name}/candidates",
    )
    return sub, keep, edge_map


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TopNSimilar(BaseRetriever):
    """Keep the n nodes most similar to the question.

    The simplest thing that could work, and a deliberately weak baseline for
    stage one: it has no notion of connectivity, so the region it hands over can
    be entirely disconnected and PCST is then forced to choose among isolated
    fragments. Its value is as a control -- any gain from a smarter generator
    has to be measured against this.
    """

    n: int = 1000

    @property
    def name(self) -> str:
        return "top-n similar"

    def select(self, graph: Any, q_emb: np.ndarray) -> np.ndarray:
        sim = P.cosine_similarity(q_emb, graph.node_emb)
        n = min(int(self.n), int(np.asarray(sim).shape[0]))
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        # Stable order so ties resolve identically across runs; the graph has
        # thousands of exactly tied nodes, so this is not a formality (F3).
        return np.sort(np.argsort(-np.asarray(sim), kind="stable")[:n]).astype(np.int64)


@dataclasses.dataclass(frozen=True)
class SeedExpansion(BaseRetriever):
    """Take the k best nodes as seeds, then everything within ``hops`` of them.

    This is neighbourhood expansion used as a *generator* rather than as a
    retriever, and the difference matters. As a retriever its weakness is that
    it cannot control its own size; as a generator that weakness is contained,
    because ``cap`` bounds the result and the inner retriever does the real
    selecting. It keeps what expansion is good at -- guaranteeing that the
    region around a promising node is available -- without letting it decide the
    final subgraph.

    Args:
        k: seed nodes, by similarity.
        hops: how far to expand from the seeds.
        cap: hard bound on the region. On a graph where one hop can reach 97% of
            the nodes (F9), an uncapped expansion is not a narrowing step at
            all. When the cap binds, the nodes kept are those closest to a seed,
            with similarity breaking ties -- so the cap degrades towards
            TopNSimilar rather than towards an arbitrary slice.
    """

    k: int = 10
    hops: int = 2
    cap: int = 2000

    @property
    def name(self) -> str:
        return "seed expansion"

    def select(self, graph: Any, q_emb: np.ndarray) -> np.ndarray:
        sim = np.asarray(P.cosine_similarity(q_emb, graph.node_emb))
        num_nodes = int(graph.num_nodes)
        k = max(1, min(int(self.k), num_nodes))
        seeds = np.argsort(-sim, kind="stable")[:k].astype(np.int64)

        # Undirected adjacency, built once per call. For a per-question sweep
        # over one patient this is cheap next to embedding; if it ever shows up
        # in a profile, cache it on the graph rather than complicating this.
        src = graph.edges["src"].to_numpy(dtype=np.int64)
        dst = graph.edges["dst"].to_numpy(dtype=np.int64)
        order = np.argsort(np.concatenate([src, dst]), kind="stable")
        starts = np.concatenate([src, dst])[order]
        targets = np.concatenate([dst, src])[order]
        offsets = np.searchsorted(starts, np.arange(num_nodes + 1))

        distance = np.full(num_nodes, -1, dtype=np.int64)
        distance[seeds] = 0
        frontier = seeds
        for depth in range(1, int(self.hops) + 1):
            if frontier.size == 0:
                break
            neighbours = np.concatenate(
                [targets[offsets[node] : offsets[node + 1]] for node in frontier]
            ) if frontier.size else np.zeros(0, dtype=np.int64)
            fresh = np.unique(neighbours[distance[neighbours] < 0])
            distance[fresh] = depth
            frontier = fresh

        reached = np.flatnonzero(distance >= 0).astype(np.int64)
        if reached.size <= int(self.cap):
            return np.sort(reached)
        # Closest first, then most similar. Keeps the cap principled instead of
        # letting numpy's ordering decide what a clinician gets to see.
        ranking = np.lexsort((-sim[reached], distance[reached]))
        return np.sort(reached[ranking[: int(self.cap)]])


# ---------------------------------------------------------------------------
# The composed retriever
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TwoStage(BaseRetriever):
    """Generator, then an inner retriever, with ids mapped back.

    Args:
        generator: stage one.
        inner: any retriever, run on the induced subgraph.

    The size dial is the inner retriever's, because that is what trades size
    against recall once the region is fixed. The generator's own parameter
    changes the *ceiling*, which is a different axis and is reported separately
    by ``ceiling_for``.
    """

    generator: Any = None
    inner: Any = None

    @property
    def name(self) -> str:
        gen = getattr(self.generator, "name", "none")
        inner = getattr(self.inner, "name", "none")
        return f"{inner} after {gen}"

    @property
    def size_dial(self) -> Optional[str]:
        return getattr(self.inner, "size_dial", None)

    @property
    def params(self) -> Dict[str, Any]:
        """Both stages' knobs, prefixed, so a results row is self-describing."""
        out: Dict[str, Any] = {}
        for prefix, stage in (("gen", self.generator), ("inner", self.inner)):
            for key, value in getattr(stage, "params", {}).items():
                out[f"{prefix}_{key}"] = value
        return out

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        keep = self.generator.select(graph, q_emb)
        if keep.size == 0:
            return Retrieval(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        sub, node_map, edge_map = induce(graph, keep)
        inner = self.inner.retrieve(sub, q_emb)
        # Back to the original index space. Everything downstream -- the gold
        # answers, the metric, the other retrievers -- lives there.
        return Retrieval(
            np.sort(node_map[np.asarray(inner.node_ids, dtype=np.int64)]),
            np.sort(edge_map[np.asarray(inner.edge_ids, dtype=np.int64)]),
        )

    def ceiling_for(self, graph: Any, q_emb: np.ndarray, answer_nodes) -> float:
        """The best recall the inner retriever could reach after stage one.

        Reported alongside recall so that a low score can be attributed. Recall
        far below ceiling is a selection problem; a low ceiling is a
        candidate-generation problem, and confusing the two wastes weeks.
        """
        gold = np.unique(np.asarray(list(answer_nodes), dtype=np.int64))
        if gold.size == 0:
            return float("nan")
        keep = self.generator.select(graph, q_emb)
        return float(np.isin(gold, keep).mean())
