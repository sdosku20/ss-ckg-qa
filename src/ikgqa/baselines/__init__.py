"""Comparators that are not graph retrievers.

The retrievers in ``ikgqa.retrieval`` all answer the same question -- given a
graph and an embedded query, which nodes and edges? SnapQuery does not: it is a
deployed service that takes a question in words and returns rows. It therefore
lives here rather than beside them, and the adaptation from rows to a subgraph
is stated explicitly in ``snapquery`` rather than hidden behind the retriever
interface.
"""
