"""
ikgqa.retrieval.paths
=====================

Shortest-path retrieval: the baseline closest in spirit to PCST, because it
also returns something connected.

The difference the paper draws out (Appendix D.2) is *control*. Shortest paths
have no way to say "that path is too long to be worth it": one distant seed
drags in a long chain of irrelevant intermediates, and seeds in different
components leave the result disconnected anyway. PCST prices every extra edge,
so it can decline.

This module is also where the thesis's breadth-first-expansion ablation
(Section 4.2) lives, since it shares the BFS machinery.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any, Optional

import numpy as np

from ikgqa.pcst import core as P
from ikgqa.retrieval.base import BaseRetriever, Retrieval


def _adjacency(ei: np.ndarray, num_nodes: int) -> dict:
    """Undirected adjacency as node -> [(neighbour, edge_id), ...]."""
    adj: dict = {i: [] for i in range(num_nodes)}
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e]), int(ei[1, e])
        adj[s].append((d, e))
        adj[d].append((s, e))
    return adj


def _shortest_path_edges(adj: dict, start: int, goal: int) -> Optional[list]:
    """BFS, returning the list of edge ids on one shortest path, or None."""
    if start == goal:
        return []
    prev: dict = {start: (None, None)}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt, eid in adj[node]:
            if nxt in prev:
                continue
            prev[nxt] = (node, eid)
            if nxt == goal:
                path, cur = [], goal
                while prev[cur][1] is not None:
                    node_prev, eid_prev = prev[cur]
                    path.append(eid_prev)
                    cur = node_prev
                return path
            queue.append(nxt)
    return None


def retrieve_shortest_paths(graph: Any, q_emb: Any, k: int = 5):
    """Take the k best nodes, then the shortest path between every pair."""
    sim = P.cosine_similarity(q_emb, graph.x)
    k = min(k, sim.shape[0])
    seeds = np.argsort(-sim, kind="stable")[:k].tolist()

    ei = np.asarray(P._to_numpy(graph.edge_index, dtype=np.int64))
    adj = _adjacency(ei, int(graph.num_nodes))

    edges: set = set()
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            path = _shortest_path_edges(adj, seeds[i], seeds[j])
            if path:
                edges.update(path)
    edges_arr = np.asarray(sorted(edges), dtype=np.int64)
    nodes = np.unique(
        np.concatenate([np.asarray(seeds, dtype=np.int64), ei[:, edges_arr].reshape(-1)])
        if edges_arr.size
        else np.asarray(seeds, dtype=np.int64)
    )
    return nodes, edges_arr


def retrieve_bfs_expansion(graph: Any, q_emb: Any, k: int = 5, hops: int = 1):
    """The thesis's breadth-first ablation: k seed nodes expanded h hops.

    Related to top-k-nodes-plus-neighbours but not equal to it, even at hops=1.
    That variant keeps every edge *incident to a seed*; this one keeps every
    edge whose both endpoints landed in the reached set, which additionally
    includes edges between two non-seed neighbours. So at hops=1 this is a
    superset -- asserted in the tests, because "these two are the same thing"
    is exactly the kind of assumption that silently invalidates an ablation.

    Args:
        k: number of seed nodes, by similarity.
        hops: how many times to expand the frontier. Every edge whose both
            endpoints are inside the final node set is returned, so the result
            is the induced subgraph rather than a tree.

    The point of including it is negative evidence: in a dense graph the node
    count grows roughly with the average degree per hop, so hops is a
    size-control parameter in name only.
    """
    if hops < 0:
        raise ValueError(f"hops must be >= 0, got {hops}")

    sim = P.cosine_similarity(q_emb, graph.x)
    k = min(k, sim.shape[0])
    frontier = set(np.argsort(-sim, kind="stable")[:k].tolist())

    ei = np.asarray(P._to_numpy(graph.edge_index, dtype=np.int64))
    adj = _adjacency(ei, int(graph.num_nodes))

    reached = set(frontier)
    for _ in range(hops):
        nxt: set = set()
        for node in frontier:
            for neighbour, _eid in adj[node]:
                if neighbour not in reached:
                    nxt.add(neighbour)
        if not nxt:
            break
        reached |= nxt
        frontier = nxt

    edges = [
        e
        for e in range(ei.shape[1])
        if int(ei[0, e]) in reached and int(ei[1, e]) in reached
    ]
    return (
        np.asarray(sorted(reached), dtype=np.int64),
        np.asarray(edges, dtype=np.int64),
    )


@dataclasses.dataclass(frozen=True)
class ShortestPaths(BaseRetriever):
    """Connect the k most similar nodes by shortest paths."""

    k: int = 5

    @property
    def name(self) -> str:
        return "shortest paths"

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        nodes, edges = retrieve_shortest_paths(_as_simple(graph), q_emb, k=self.k)
        return Retrieval(nodes, edges)


@dataclasses.dataclass(frozen=True)
class BFSExpansion(BaseRetriever):
    """Seed on the k most similar nodes, expand hops steps, induce."""

    k: int = 5
    hops: int = 1

    @property
    def name(self) -> str:
        return "BFS expansion"

    @property
    def size_dial(self) -> str:
        return "hops"

    def retrieve(self, graph: Any, q_emb: np.ndarray) -> Retrieval:
        nodes, edges = retrieve_bfs_expansion(
            _as_simple(graph), q_emb, k=self.k, hops=self.hops
        )
        return Retrieval(nodes, edges)


def _as_simple(graph: Any) -> Any:
    return graph.as_simple_graph() if hasattr(graph, "as_simple_graph") else graph
