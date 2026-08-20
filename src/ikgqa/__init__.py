"""
ikgqa -- Intent-Aware Subgraph Selection for Clinical KGQA
=========================================================

Code for the MSc thesis of the same name (University of Basel, 2026). It
implements G-Retriever's PCST subgraph selection, the baselines it is compared
against, and the evaluation that decides between them.

Layout
------

    ikgqa.graph        TextualGraph: nodes, edges, embeddings, in one validated object
    ikgqa.encoders     text -> vectors (bag-of-words for dev, SentenceBERT for real runs)
    ikgqa.pcst         the PCST algorithm, held byte-equivalent to the published code
    ikgqa.retrieval    every selection strategy behind one interface
    ikgqa.eval         metrics, sweeps and reports
    ikgqa.data         graphs to run on (toy graphs today, the STCS graph next)

Thirty-second tour
------------------

    from ikgqa.data import toy
    from ikgqa.encoders import BagOfWordsEncoder
    from ikgqa.graph import TextualGraph
    from ikgqa.retrieval import PCST

    kg = toy.clinical_kg()
    graph = TextualGraph.from_texts(kg.node_texts, kg.triples, kg.encoder, name="clinical")
    selection = PCST(cost_e=0.5).retrieve(graph, kg.q("which complication did patient 001 have"))
    print(selection.node_ids, selection.edge_ids)

To see *why* it chose that, ask for the trace:

    trace = PCST().trace(graph, kg.q("..."))
    print(trace.trace.prize_table(graph.nodes))
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
