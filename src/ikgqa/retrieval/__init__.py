"""
ikgqa.retrieval
===============

Every subgraph-selection strategy compared in the thesis, behind one interface.

    from ikgqa.retrieval import PCST, TopKTriples, ShortestPaths, BFSExpansion

    retrievers = [PCST(cost_e=c) for c in (0.1, 0.3, 0.5, 1.0)]
    for r in retrievers:
        selection = r.retrieve(graph, q_emb)

Which family each belongs to, in the terms of thesis Section 2.3:

    TopKTriples             independent scoring, no connectivity guarantee
    TopKNodesPlusNeighbors  neighbourhood expansion, no size control
    BFSExpansion            neighbourhood expansion, explicit hop count
    ShortestPaths           path-based, connected but unbounded length
    PCST                    global objective, connectivity *and* size priced in

The functional forms (retrieve_topk_triples and friends) are exported too, for
the playground and for anyone who prefers plain functions.
"""

from ikgqa.retrieval.base import BaseRetriever, Retrieval, Retriever, assert_valid
from ikgqa.retrieval.paths import (
    BFSExpansion,
    ShortestPaths,
    retrieve_bfs_expansion,
    retrieve_shortest_paths,
)
from ikgqa.retrieval.pcst import PCST, retrieve_pcst
from ikgqa.retrieval.similarity import (
    TopKNodesPlusNeighbors,
    TopKTriples,
    retrieve_topk_nodes_plus_neighbors,
    retrieve_topk_triples,
)

#: The comparison set used by default in reports. Order is the order of
#: appearance in thesis Section 2.3: weakest structural guarantee first.
DEFAULT_RETRIEVERS = (
    TopKTriples(),
    TopKNodesPlusNeighbors(),
    BFSExpansion(),
    ShortestPaths(),
    PCST(),
)

__all__ = [
    "BFSExpansion",
    "BaseRetriever",
    "DEFAULT_RETRIEVERS",
    "PCST",
    "Retrieval",
    "Retriever",
    "ShortestPaths",
    "TopKNodesPlusNeighbors",
    "TopKTriples",
    "assert_valid",
    "retrieve_bfs_expansion",
    "retrieve_pcst",
    "retrieve_shortest_paths",
    "retrieve_topk_nodes_plus_neighbors",
    "retrieve_topk_triples",
]
