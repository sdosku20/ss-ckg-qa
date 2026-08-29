"""
Does narrowing the graph before PCST help, and what does it cost?

    python experiments/candidate_generation.py            # full patient
    python experiments/candidate_generation.py --small    # faster

G-Retriever assumes one modest graph per question. One STCS patient is 15,810
nodes, so something has to produce the region PCST optimises inside. This script
compares running PCST on the whole patient graph against running it after a
candidate-generation stage, and reports three things per configuration:

  recall    what fraction of the answer entities came back
  ceiling   what fraction stage one *kept*, so recall could never exceed it
  seconds   wall-clock per question, because a method that cannot run at cohort
            scale is not a method, whatever its recall

Reporting recall without ceiling is what makes this kind of experiment
misleading: a low score can mean the selector chose badly or that the region
never contained the answer, and those need opposite fixes.

Runs on the synthetic replica, whose structure matches one real patient exactly.
Mechanism, not clinical quality -- see ikgqa.data.replica.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from ikgqa.data.replica import (
    PATIENT_0,
    PATIENT_0_SMALL,
    build_replica,
    planted_questions,
)
from ikgqa.data.sphn import default_encoder
from ikgqa.retrieval import PCST, SeedExpansion, TopNSimilar, TwoStage, assert_valid

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

# topk_e=0 and an explicit cost_e throughout: with few, heavily shared relation
# types the edge-prize mechanism drives the effective edge cost to ~0 (F2).
INNER = PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable")


def configurations(small: bool):
    """(label, generator) pairs. None means "no narrowing", the control."""
    if small:
        return [
            ("whole graph", None),
            ("top-n similar, n=100", TopNSimilar(n=100)),
            ("top-n similar, n=400", TopNSimilar(n=400)),
            ("seed expansion, cap=100", SeedExpansion(k=10, hops=2, cap=100)),
            ("seed expansion, cap=400", SeedExpansion(k=10, hops=2, cap=400)),
            ("seed expansion, cap=1000", SeedExpansion(k=10, hops=2, cap=1000)),
        ]
    return [
        ("whole graph", None),
        ("top-n similar, n=500", TopNSimilar(n=500)),
        ("top-n similar, n=2000", TopNSimilar(n=2000)),
        ("seed expansion, cap=500", SeedExpansion(k=10, hops=2, cap=500)),
        ("seed expansion, cap=2000", SeedExpansion(k=10, hops=2, cap=2000)),
        ("seed expansion, cap=5000", SeedExpansion(k=20, hops=2, cap=5000)),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--questions", type=int, default=5, help="per kind")
    args = parser.parse_args(argv)

    profile = PATIENT_0_SMALL if args.small else PATIENT_0
    tag = "small" if args.small else "full"

    print("building the replica ...", flush=True)
    graph, table, _, rows = build_replica(profile, seed=args.seed)
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = planted_questions(graph, table, rows, n_per_kind=args.questions)
    print(f"{graph}\n{len(questions)} questions\n", flush=True)

    records = []
    for label, generator in configurations(args.small):
        retriever = INNER if generator is None else TwoStage(generator=generator, inner=INNER)
        print(f"  {label} ...", flush=True)
        for question in questions:
            q_emb = encoder.encode_one(question.text)
            gold = set(int(i) for i in question.answer_nodes)

            started = time.perf_counter()
            selection = retriever.retrieve(graph, q_emb)
            elapsed = time.perf_counter() - started
            assert_valid(selection, graph)

            if generator is None:
                candidates = graph.num_nodes
                ceiling = 1.0
            else:
                kept = generator.select(graph, q_emb)
                candidates = int(kept.size)
                ceiling = float(np.isin(sorted(gold), kept).mean())

            got = set(selection.node_ids.tolist())
            records.append(
                {
                    "condition": label,
                    "qid": question.qid,
                    "qkind": question.qid.rsplit("-", 1)[0],
                    "candidates": candidates,
                    "nodes": selection.num_nodes,
                    "edges": selection.num_edges,
                    "recall": len(gold & got) / len(gold),
                    "ceiling": ceiling,
                    "seconds": elapsed,
                }
            )

    frame = pd.DataFrame(records)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"candidate_generation_{tag}.csv"
    frame.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    summary = (
        frame.groupby("condition", sort=False)
        .agg(
            candidates=("candidates", "mean"),
            nodes=("nodes", "mean"),
            edges=("edges", "mean"),
            recall=("recall", "mean"),
            ceiling=("ceiling", "mean"),
            seconds=("seconds", "mean"),
        )
        .round(4)
    )
    print("\n=== averaged over every question ===")
    print(summary.to_string())

    print("\n=== recall by question kind ===")
    by_kind = frame.pivot_table(
        index="condition", columns="qkind", values="recall", aggfunc="mean", sort=False
    )
    print(by_kind.round(4).to_string())

    print("\n=== ceiling by question kind (what stage one kept) ===")
    ceilings = frame.pivot_table(
        index="condition", columns="qkind", values="ceiling", aggfunc="mean", sort=False
    )
    print(ceilings.round(4).to_string())

    print(f"\nwrote {out}")

    baseline = summary.loc["whole graph"]
    print("\n=== summary ===")
    print(
        "Recall is only comparable at comparable subgraph size, so the node count\n"
        "is printed beside it. A configuration that returns twenty times as many\n"
        "nodes for twice the recall has not beaten anything; it has moved along\n"
        "the size axis, which is what the recall-vs-size curve exists to show."
    )
    print(
        f"\n  whole graph          {baseline['nodes']:7.1f} nodes  "
        f"recall {baseline['recall']:.4f}  ceiling 1.0000  "
        f"{baseline['seconds'] * 1000:5.0f} ms"
    )
    for label in summary.index:
        if label == "whole graph":
            continue
        row = summary.loc[label]
        speed = baseline["seconds"] / row["seconds"] if row["seconds"] else float("nan")
        comparable = "" if abs(row["nodes"] - baseline["nodes"]) <= 1 else "  <- larger subgraph"
        print(
            f"  {label:<20} {row['nodes']:7.1f} nodes  recall {row['recall']:.4f}  "
            f"ceiling {row['ceiling']:.4f}  {row['seconds'] * 1000:5.0f} ms  "
            f"{speed:.1f}x{comparable}"
        )

    codeonly = ceilings["dx-codeonly"] if "dx-codeonly" in ceilings.columns else None
    if codeonly is not None and float(codeonly.drop("whole graph", errors="ignore").max()) == 0.0:
        print(
            "\nEvery generator discarded every code-only diagnosis (ceiling 0.0000).\n"
            "A node with no matching text is neither similar to the question nor\n"
            "near a node that is, so narrowing removes it first. Terminology\n"
            "resolution is therefore a prerequisite for candidate generation, not\n"
            "an independent improvement to it."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
