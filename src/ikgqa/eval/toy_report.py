"""
ikgqa.eval.toy_report
=====================

The edge-overlap report used by the playground and its tests.

This is the *older*, simpler view: for a hand-written set of gold triples, how
many did each retriever return and how much noise came with them. It is kept
because it is easy to read by hand on a 20-node graph, which makes it good for
teaching and for catching gross regressions.

It is not the thesis metric. Section 4.3 measures answer-*node* recall over
real questions with human-validated answers; see ikgqa.eval.metrics. Do not
quote numbers from this module in the thesis: the toy graph and its gold set
were authored together, so any ranking they produce is a demonstration, not
evidence.
"""

from __future__ import annotations

import pandas as pd

from ikgqa.eval.metrics import is_connected
from ikgqa.pcst import core as P
from ikgqa.retrieval.paths import retrieve_shortest_paths
from ikgqa.retrieval.pcst import retrieve_pcst
from ikgqa.retrieval.similarity import (
    retrieve_topk_nodes_plus_neighbors,
    retrieve_topk_triples,
)


def score(kg, question: str, gold_edges: set, k: int = 5) -> pd.DataFrame:
    """Run all four retrievers on one question and score them.

    Columns:
      recall     fraction of the gold triples that were retrieved
      precision  fraction of retrieved triples that are gold
      nodes/edges/chars  the size of what the LLM would receive
      connected  whether the result is a single connected component
    """
    q = kg.q(question)
    triple_texts = [
        f"{kg.node_texts[s]} {r} {kg.node_texts[d]}" for s, r, d in kg.triples
    ]
    triple_emb = kg.encoder.encode(triple_texts)

    runs = {
        "PCST": retrieve_pcst(kg.graph, q, kg.nodes_df, kg.edges_df, topk=3, topk_e=k, cost_e=0.5),
        "top-k triples (KAPING)": retrieve_topk_triples(kg.graph, q, k=k, triple_emb=triple_emb),
        "top-k nodes + neighbours": retrieve_topk_nodes_plus_neighbors(kg.graph, q, k=k),
        "shortest paths": retrieve_shortest_paths(kg.graph, q, k=k),
    }

    rows = []
    for name, (nodes, edges) in runs.items():
        got = set(edges.tolist())
        desc = P.build_description(kg.nodes_df, kg.edges_df, nodes, edges)
        rows.append(
            {
                "retriever": name,
                "recall": len(gold_edges & got) / max(len(gold_edges), 1),
                "precision": len(gold_edges & got) / max(len(got), 1),
                "nodes": len(nodes),
                "edges": len(got),
                "chars": len(desc),
                "connected": is_connected(kg.graph.num_nodes, kg.edge_index, nodes, edges),
            }
        )
    return pd.DataFrame(rows)


def score_all(kg, gold: dict, k: int = 5) -> pd.DataFrame:
    """Average the per-question scores over a whole gold set."""
    frames = [score(kg, q, g, k=k) for q, g in gold.items()]
    combined = pd.concat(frames)
    agg = combined.groupby("retriever", sort=False).agg(
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        nodes=("nodes", "mean"),
        edges=("edges", "mean"),
        chars=("chars", "mean"),
        always_connected=("connected", "all"),
    )
    return agg.round(3)


def main() -> None:  # pragma: no cover - console output
    from ikgqa.data import toy as T

    pd.set_option("display.width", 200)
    kg = T.clinical_kg()
    gold = T.clinical_gold_set()

    print("=" * 78)
    print(f"Retrieval quality on the clinical toy graph, {len(gold)} questions, k=5")
    print("=" * 78)
    for question, gold_edges in gold.items():
        print(f"\nQ: {question}")
        print("   gold triples:")
        for e in sorted(gold_edges):
            row = kg.edges_df.iloc[e]
            print(
                f"     ({kg.node_texts[int(row['src'])]}) -[{row['edge_attr']}]-> "
                f"({kg.node_texts[int(row['dst'])]})"
            )
        print(score(kg, question, gold_edges, k=5).to_string(index=False))

    print("\n" + "=" * 78)
    print("Averaged over all questions")
    print("=" * 78)
    print(score_all(kg, gold, k=5).to_string())


if __name__ == "__main__":  # pragma: no cover
    main()
