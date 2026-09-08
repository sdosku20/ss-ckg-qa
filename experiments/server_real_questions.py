"""
Run a handful of questions against the live clinical graph on CHIL.

SERVER ONLY. This opens a read-only session on the hospital Neo4j, so it runs on
chil.scicoreplus.unibas.ch and nowhere else.

    source ~/.ikgqa.env
    ~/venvs/ikgqa/bin/python experiments/server_real_questions.py --derive 5

Two question sources, and they measure different things:

  --derive N     Questions built from the graph's own `embed_text`. No clinical
                 input needed, so this runs today. It is a CEILING, not a
                 realistic score: the query is literally the target node's own
                 text, so nothing a real user types will ever do better. Its
                 value is diagnostic -- if recall is poor even here, the problem
                 is structural and no question wording will fix it.

  --gold FILE    Real questions you wrote, in the gold-set format
                 (ikgqa.eval.goldset). This is the honest measurement, and it
                 needs you to name the answer nodes by `node_key`. Template:
                     ~/venvs/ikgqa/bin/python -m ikgqa.eval.goldset template --out q.jsonl

Data handling. Two files come out:

    <out>.full.csv    question text and per-node detail. Mode 600. STAYS HERE.
    <out>.safe.csv    an explicit allowlist of aggregate columns only.
                      No text, no values, no keys, no patient identifiers.
                      This is the one that may leave the server.

Everything printed to the terminal is drawn from the safe columns plus SPHN
label names and counts, so the console output is quotable in the thesis.

One thing you control: `qid` appears in the safe file. The derived ones are
built from SPHN label names, which are schema and publishable. If you write a
gold set, keep its qids opaque (q1, q2, ...) -- do not put the question text or
a patient reference in the identifier.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ikgqa.data.sphn import default_encoder, load_patient_graph  # noqa: E402
from ikgqa.eval.metrics import KIND_ENTITY, Question, measure  # noqa: E402
from ikgqa.retrieval import (  # noqa: E402
    BFSExpansion,
    PCST,
    SeedExpansion,
    ShortestPaths,
    TopKNodesPlusNeighbors,
    TopKTriples,
    TwoStage,
    assert_valid,
)

# Columns allowed to leave the server. Allowlist, not denylist: a new column
# added upstream is excluded by default rather than leaking by default.
SAFE_COLUMNS = (
    "qid",
    "source",
    "kind",
    "answer_nodes",
    "retriever",
    "nodes",
    "edges",
    "chars",
    "node_fraction",
    "answer_node_recall",
    "connected",
    "seconds",
)

# What the experiments use. topk_e=0 because edge prizes collapse the effective
# edge cost on this graph (finding F2); see docs/pcst_from_the_papers.md section 8.
INNER = dict(topk=10, topk_e=0, cost_e=0.5, tie_break="stable")


def retrievers_for(graph):
    """Every strategy, each behind the same two-stage generator.

    A patient graph is far larger than the subgraphs G-Retriever was built for,
    so all five run on an induced region rather than the whole patient. Same
    generator for all of them, so the comparison is not confounded by it.
    """
    gen = SeedExpansion(k=10, hops=1, cap=2000)
    return [
        TwoStage(generator=gen, inner=PCST(**INNER)),
        TwoStage(generator=gen, inner=TopKTriples(k=INNER["topk"])),
        TwoStage(generator=gen, inner=TopKNodesPlusNeighbors(k=INNER["topk"])),
        TwoStage(generator=gen, inner=BFSExpansion(k=INNER["topk"], hops=1)),
        TwoStage(generator=gen, inner=ShortestPaths(k=INNER["topk"])),
    ]


def derive_questions(table: pd.DataFrame, limit: int, max_answers: int = 50) -> list:
    """Self-labelled questions: the query is a concept's own embed_text.

    A concept is one (sphn_label, embed_text) pair, and its answer set is every
    node carrying it -- which on this graph is usually more than one, because
    embed_text deduplicates heavily. Concepts with more than `max_answers` nodes
    are skipped: they are aggregations ("every creatinine value"), and recall
    against a budget of ten nodes measures the budget, not the retriever.
    """
    groups = table.groupby(["sphn_label", "embed_text"], sort=True).indices
    by_label: dict = {}
    for (label, text), idx in groups.items():
        if not text or not str(text).strip():
            continue
        if not (1 <= len(idx) <= max_answers):
            continue
        by_label.setdefault(label, []).append((text, np.asarray(idx, dtype=np.int64)))

    # Spread across labels rather than taking everything from the biggest one.
    picked: list = []
    round_no = 0
    while len(picked) < limit and any(len(v) > round_no for v in by_label.values()):
        for label in sorted(by_label):
            if len(picked) >= limit:
                break
            if len(by_label[label]) > round_no:
                text, idx = by_label[label][round_no]
                picked.append(
                    Question(
                        text=str(text),
                        answer_nodes=tuple(int(i) for i in idx),
                        qid=f"ceiling-{label}-{round_no}",
                        kind=KIND_ENTITY,
                        validated=False,
                    )
                )
        round_no += 1
    return picked


def load_gold(path: str, graph, table) -> list:
    """Real hand-written questions, resolved from node_key to node index."""
    from ikgqa.eval.goldset import GoldSet

    gold = GoldSet.from_jsonl(path)
    problems = gold.validate()
    for p in problems:
        print(f"  gold-set {p}")
    resolved, missing = gold.resolve(graph)
    if missing:
        print(f"  WARNING: {len(missing)} answer keys not found in this patient's graph")
    return resolved


def run(questions: list, graph, table, encoder, source: str) -> pd.DataFrame:
    rows: list = []
    for q in questions:
        q_emb = encoder.encode_one(q.text)
        for r in retrievers_for(graph):
            t0 = time.perf_counter()
            selection = r.retrieve(graph, q_emb)
            seconds = time.perf_counter() - t0
            assert_valid(selection, graph)
            row = measure(graph, selection, q)
            row.update(
                retriever=getattr(r.inner, "name", r.name),
                source=source,
                seconds=round(seconds, 3),
                answer_nodes=len(q.answer_nodes),
                # Safe, and the only "what did it actually return" we can print.
                labels=table.loc[selection.node_ids, "sphn_label"].value_counts().to_dict(),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, out: str) -> tuple:
    full = Path(f"{out}.full.csv")
    safe = Path(f"{out}.safe.csv")
    df.to_csv(full, index=False)
    os.chmod(full, stat.S_IRUSR | stat.S_IWUSR)  # 600: contains question text
    df[[c for c in SAFE_COLUMNS if c in df.columns]].to_csv(safe, index=False)
    return full, safe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="server_real_questions.py")
    ap.add_argument("--patient", type=int, default=0, help="patient index, not an identifier")
    ap.add_argument("--limit", type=int, default=20000, help="max rows per Cypher query")
    ap.add_argument("--derive", type=int, default=0, help="auto-derive N ceiling questions")
    ap.add_argument("--gold", default="", help="a gold-set .jsonl of real questions")
    ap.add_argument("--out", default="real_questions", help="output prefix")
    args = ap.parse_args(argv)

    if not args.derive and not args.gold:
        ap.error("give --derive N, or --gold FILE, or both")

    print("loading one patient from the clinical Neo4j (read-only) ...")
    graph, table, stats = load_patient_graph(patient_index=args.patient, limit=args.limit)
    print(stats.summary())
    print(f"\n{graph}")
    n_nodes, n_texts = len(table), stats.distinct_embed_texts
    print(
        f"text multiplicity: {n_nodes} nodes share {n_texts} distinct embed texts "
        f"({n_nodes / max(n_texts, 1):.1f} nodes per text)"
    )

    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    assert encoder.dim == graph.node_emb.shape[1], (
        f"encoder dim {encoder.dim} != graph dim {graph.node_emb.shape[1]}; "
        "build_patient_graph no longer uses default_encoder"
    )

    frames: list = []
    if args.derive:
        qs = derive_questions(table, args.derive)
        print(f"\nderived {len(qs)} ceiling questions (query = the node's own text):")
        for q in qs:
            print(f"  {q.qid:<44s} answer nodes {len(q.answer_nodes)}")
        if qs:
            frames.append(run(qs, graph, table, encoder, "ceiling"))
    if args.gold:
        print(f"\nreading gold questions from {args.gold}")
        qs = load_gold(args.gold, graph, table)
        print(f"  {len(qs)} usable")
        if qs:
            frames.append(run(qs, graph, table, encoder, "gold"))

    if not frames:
        print("\nno questions to run.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    full, safe = write_outputs(df, args.out)

    print(f"\n{'=' * 78}\nRESULTS  (recall is always read against the size beside it)\n{'=' * 78}")
    show = df[["qid", "source", "answer_nodes", "retriever", "nodes", "answer_node_recall", "connected", "seconds"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n{'=' * 78}\nBY RETRIEVER\n{'=' * 78}")
    agg = df.groupby("retriever").agg(
        questions=("qid", "count"),
        mean_recall=("answer_node_recall", "mean"),
        sd_recall=("answer_node_recall", "std"),
        mean_nodes=("nodes", "mean"),
        mean_seconds=("seconds", "mean"),
    )
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\n{'=' * 78}\nWHAT CAME BACK, BY SPHN LABEL  (no values, no text)\n{'=' * 78}")
    for qid, grp in df.groupby("qid", sort=True):
        first = grp.iloc[0]
        print(f"\n{qid}  (answer nodes {first['answer_nodes']})")
        for _, r in grp.iterrows():
            labels = ", ".join(f"{k} x{v}" for k, v in sorted(r["labels"].items()))
            print(f"    {r['retriever']:<28s} {int(r['nodes']):>5d} nodes  {labels}")

    print(f"\nfull results (mode 600, do not copy off the server): {full}")
    print(f"safe results (aggregates only, may be shared):         {safe}")
    print(
        "\nRead the ceiling rows as a diagnostic, not a score. The query is the target\n"
        "node's own text, so a real question can only do worse. If a ceiling row is\n"
        "already low, the cause is graph structure or text duplication, not wording."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
