"""
Recall-versus-size sweep on the measured-shape replica.

    python experiments/sweep_replica.py            # full patient (~15,800 nodes)
    python experiments/sweep_replica.py --small    # 10x smaller, for iteration

Runs every retriever across its own size dial on a graph whose structure matches
the real patient #0 exactly, evaluates the planted questions, and writes tidy
results to experiments/results/.

Read the header of ikgqa.data.replica before quoting anything from here. The
graph is faithful in shape and synthetic in content, so these numbers are
evidence about *mechanism* -- does the size dial work, does tie-breaking decide
the outcome, can a code-only node be retrieved at all -- and not about clinical
retrieval quality.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import pandas as pd

from ikgqa.data.replica import (
    PATIENT_0,
    PATIENT_0_SMALL,
    build_replica,
    describe_fidelity,
    planted_questions,
)
from ikgqa.data.sphn import default_encoder
from ikgqa.eval.metrics import evaluate, summarise
from ikgqa.eval.sweep import Sweep, sweep, to_curve
from ikgqa.retrieval import (
    PCST,
    BFSExpansion,
    ShortestPaths,
    TopKNodesPlusNeighbors,
    TopKTriples,
)

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def build_sweeps(encoder: Any) -> list[Sweep]:
    """One Sweep per retriever family, each over its own dial.

    The dials are not the same quantity -- PCST sweeps a cost, the baselines
    sweep a count -- which is exactly why the output is plotted against a
    *measured* size. Ranges were chosen to overlap in achieved subgraph size, so
    the curves can be compared at equal size rather than at equal parameter.

    The encoder is threaded through for KAPING's benefit: without it TopKTriples
    ranks on relation text alone, which on a graph with six relation types means
    every triple of a type is tied and the baseline collapses to "the first k
    edges by index". Understating a baseline is as much a result error as
    overstating your own method.
    """
    return [
        # topk_e=0 throughout: with few, heavily shared relation types the edge
        # prize splits so thin that adjust_edge_cost drives the effective cost
        # to ~0 and PCST returns the whole candidate region (finding F2). The
        # pcst-published sweep below keeps the defaults on purpose, to show it.
        Sweep(
            factory=lambda c: PCST(topk=10, topk_e=0, cost_e=c, tie_break="stable"),
            values=(3.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01),
            dial="cost_e",
        ),
        Sweep(
            factory=lambda c: PCST(topk=3, topk_e=5, cost_e=c, tie_break="stable"),
            values=(0.5,),
            dial="cost_e (published defaults)",
        ),
        Sweep(
            factory=lambda k, e=encoder: TopKTriples(k=k, encoder=e),
            values=(1, 3, 10, 30, 100, 300),
            dial="k",
        ),
        Sweep(
            factory=lambda k: TopKNodesPlusNeighbors(k=k),
            values=(1, 3, 10, 30, 100),
            dial="k",
        ),
        Sweep(
            factory=lambda h: BFSExpansion(k=3, hops=h),
            values=(1, 2, 3),
            dial="hops",
        ),
        Sweep(
            factory=lambda k: ShortestPaths(k=k),
            values=(2, 3, 5, 10),
            dial="k",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true", help="10x smaller replica")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--questions", type=int, default=5, help="per question kind")
    args = parser.parse_args(argv)

    profile = PATIENT_0_SMALL if args.small else PATIENT_0
    tag = "small" if args.small else "full"

    print("building the replica ...", flush=True)
    graph, table, stats, rows = build_replica(profile, seed=args.seed)
    print(describe_fidelity(profile, stats, graph))
    print(f"\n{graph}\n", flush=True)

    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = planted_questions(graph, table, rows, n_per_kind=args.questions)
    print(f"{len(questions)} planted questions:")
    for q in questions:
        print(f"  {q.qid:16s} {len(q.answer_nodes):>5d} answer nodes  {q.text[:44]!r}")

    # Encode once; every configuration reuses these.
    q_emb = encoder.encode([q.text for q in questions])

    print("\nsweeping ...", flush=True)
    swept = sweep(graph, build_sweeps(encoder), questions, encoder, q_embeddings=q_emb)

    RESULTS.mkdir(parents=True, exist_ok=True)
    swept_path = RESULTS / f"replica_sweep_{tag}.csv"
    swept.to_csv(swept_path, index=False)

    curve = to_curve(swept, size_axis="nodes")
    curve_path = RESULTS / f"replica_curve_{tag}.csv"
    curve.to_csv(curve_path, index=False)

    # Per-question detail for the best-behaved PCST setting, because the average
    # hides the finding that matters: which *kinds* of question are reachable.
    detail = evaluate(
        graph,
        [PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable"),
         TopKTriples(k=10, encoder=encoder)],
        questions,
        encoder,
        q_embeddings=q_emb,
    )
    detail_path = RESULTS / f"replica_per_question_{tag}.csv"
    detail.to_csv(detail_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("\n=== recall vs size, one row per configuration ===")
    cols = ["retriever", "dial", "dial_value", "nodes", "edges", "node_fraction",
            "answer_node_recall", "n_scored", "always_connected"]
    print(swept[[c for c in cols if c in swept.columns]].to_string(index=False))

    print("\n=== per question, at comparable size (PCST cost_e=0.5 vs top-10) ===")
    dcols = ["qid", "retriever", "nodes", "answer_node_recall", "connected"]
    print(
        detail[[c for c in dcols if c in detail.columns]]
        .sort_values(["qid", "retriever"])
        .to_string(index=False)
    )

    print("\n=== recall by question kind ===")
    detail = detail.copy()
    # qid is "<kind>-<n>"; strip the index to group the kinds.
    detail["qkind"] = detail["qid"].str.replace(r"-\d+$", "", regex=True)
    by_kind = detail.groupby(["retriever", "qkind"])["answer_node_recall"].agg(
        ["mean", "count"]
    )
    print(by_kind.round(4).to_string())

    print(f"\nwrote:\n  {swept_path}\n  {curve_path}\n  {detail_path}")

    # The headline numbers, restated so they end up in the terminal log too.
    published = swept[swept["dial"].str.contains("published")]
    if not published.empty:
        n = float(published["nodes"].iloc[0])
        print(
            f"\nF2 check: PCST with published defaults returned {n:.0f} of "
            f"{graph.num_nodes} nodes ({100 * n / graph.num_nodes:.1f}% of the graph)"
        )
    for kind in ("dx-named", "dx-codeonly"):
        subset = detail[detail["qkind"] == kind]
        if subset.empty:
            continue
        means = subset.groupby("retriever")["answer_node_recall"].mean()
        print(
            f"F6 check: mean recall on {kind} diagnoses is "
            + ", ".join(f"{r}={v:.3f}" for r, v in means.items())
        )
    print(
        "         dx-codeonly is expected to stay near 0 until the terminology "
        "cross-walk lands; that gap is the measurement the cross-walk has to move."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
