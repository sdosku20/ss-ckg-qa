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
import re
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


CLINICAL_BANNER = """
################################################################################
#  The block below contains CLINICAL TEXT from a real patient record.          #
#  It is here so you can author gold questions. It must not be copied off the  #
#  server, pasted into chat, or included in the thesis. Only the node_key you  #
#  put in the .jsonl travels, and that file stays on BioMedIT too.             #
################################################################################
"""


# An ICD-10 / ATC-shaped token: a letter, two digits, optionally a dotted tail.
# Concepts whose whole text is one of these are the "no words to match" case
# that terminology resolution exists to fix.
BARE_CODE = re.compile(r"^[A-Z]\d{2}(\.\d+)?[A-Z]?$|^[A-Z]\d{2}$|^[A-Z]\d[A-Z]{2}\d{2}$")


def _is_bare_code(text: str) -> bool:
    tail = text.split(":", 1)[-1].strip()
    return bool(BARE_CODE.match(tail))


def show_concepts(table: pd.DataFrame, label: str, limit: int) -> None:
    """List the distinct concepts under one SPHN label, with a node_key each.

    This is the lookup you need to fill in `answer_keys`. Concepts are grouped
    by embed_text, so the count beside each one is how many nodes carry it --
    which is exactly the size of that question's answer set.
    """
    sel = table[table["sphn_label"].astype(str).str.contains(label, case=False, na=False)]
    if sel.empty:
        labels = ", ".join(sorted(table["sphn_label"].astype(str).unique()))
        print(f"no label matches {label!r}. Available: {labels}")
        return
    print(CLINICAL_BANNER)
    print(f"{len(sel)} nodes under labels matching {label!r}\n")
    groups = sel.groupby("embed_text", sort=True)
    rows = sorted(groups, key=lambda kv: (-len(kv[1]), str(kv[0])))[:limit]
    print(f"{'nodes':>5}  {'node_key (use this in answer_keys)':<44} embed_text")
    for text, grp in rows:
        key = str(grp["node_key"].iloc[0])
        mark = "  <- CODE ONLY, no words to match" if _is_bare_code(str(text)) else ""
        print(f"{len(grp):>5}  {key:<44} {str(text)[:60]}{mark}")
    print(
        f"\nShowing {len(rows)} of {groups.ngroups} distinct concepts."
        "\nPick a spread: some with words in embed_text, some that are bare codes."
        "\nFor a concept carried by N nodes, put ALL N keys in answer_keys, or the"
        "\nrecall denominator will be wrong. Use --keys-for to dump them."
    )


def show_keys_for(table: pd.DataFrame, text: str) -> None:
    """Every node_key whose embed_text matches, ready to paste into answer_keys."""
    sel = table[table["embed_text"].astype(str).str.contains(text, case=False, na=False)]
    if sel.empty:
        print(f"no embed_text contains {text!r}")
        return
    print(CLINICAL_BANNER)
    for et, grp in sel.groupby("embed_text", sort=True):
        keys = [str(k) for k in grp["node_key"]]
        print(f"\n{len(keys)} nodes, embed_text = {str(et)[:70]!r}")
        print("  answer_keys: " + json_list(keys))


def json_list(keys: list) -> str:
    import json

    return json.dumps(keys, ensure_ascii=False)


def draft_gold(table: pd.DataFrame, picks: list, out: str, patient: str) -> None:
    """Write a gold-set skeleton with answer_keys already filled in.

    Hand-writing JSON with forty node keys in it is the step most likely to go
    wrong, and a wrong key silently changes the recall denominator. So this
    fills every key for each picked concept and leaves exactly one thing for you
    to type: the question itself.

    Each `pick` is a substring of a concept's embed_text, from --concepts.
    """
    import json

    lines: list = []
    for i, pick in enumerate(picks, start=1):
        sel = table[table["embed_text"].astype(str).str.contains(pick, case=False, na=False, regex=False)]
        if sel.empty:
            print(f"  q{i}: NO MATCH for {pick!r} -- skipped")
            continue
        texts = sorted(set(str(t) for t in sel["embed_text"]))
        if len(texts) > 1:
            print(f"  q{i}: {pick!r} matches {len(texts)} different concepts; narrow it. Skipped.")
            continue
        keys = [str(k) for k in sel["node_key"]]
        coded = _is_bare_code(texts[0])
        lines.append(
            {
                "qid": f"q{i}",
                "text": "WRITE THE QUESTION HERE, as a clinician would ask it",
                "language": "de",
                "kind": "entity",
                "patient": patient,
                "answer_keys": keys,
                "validated_by": "",
                "validated_on": "",
                "source": "authored from --concepts",
                "notes": (
                    f"target concept: {texts[0]}"
                    + (
                        "  | CODE ONLY: ask by the disease NAME, never the code. "
                        "Expected recall 0.0 until terminology resolution runs."
                        if coded
                        else "  | has words: ask using words from this description."
                    )
                ),
                "skip_reason": "",
            }
        )
        print(f"  q{i}: {len(keys)} answer keys, {'CODE ONLY' if coded else 'named'}")

    path = Path(out)
    with path.open("w", encoding="utf-8") as fh:
        for row in lines:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    print(f"\nwrote {path} ({len(lines)} questions, mode 600, stays on the server)")
    print("Now edit ONLY the \"text\" field of each line. Everything else is done.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="server_real_questions.py")
    ap.add_argument("--patient", type=int, default=0, help="patient index, not an identifier")
    ap.add_argument("--limit", type=int, default=20000, help="max rows per Cypher query")
    ap.add_argument("--derive", type=int, default=0, help="auto-derive N ceiling questions")
    ap.add_argument("--gold", default="", help="a gold-set .jsonl of real questions")
    ap.add_argument("--out", default="real_questions", help="output prefix")
    ap.add_argument(
        "--concepts",
        default="",
        help="list distinct concepts under an SPHN label, with a node_key each (authoring aid)",
    )
    ap.add_argument("--top", type=int, default=30, help="how many concepts --concepts shows")
    ap.add_argument(
        "--keys-for",
        default="",
        help="dump every node_key whose embed_text contains this string (authoring aid)",
    )
    ap.add_argument(
        "--draft-gold",
        default="",
        help="write a gold-set skeleton to this path with answer_keys pre-filled",
    )
    ap.add_argument(
        "--pick",
        action="append",
        default=[],
        metavar="TEXT",
        help="a concept to build a question for; repeat once per question",
    )
    args = ap.parse_args(argv)

    if not any((args.derive, args.gold, args.concepts, args.keys_for, args.draft_gold)):
        ap.error(
            "give one of --derive N, --gold FILE, --concepts LABEL, --keys-for TEXT, "
            "or --draft-gold OUT --pick TEXT ..."
        )
    if args.draft_gold and not args.pick:
        ap.error("--draft-gold needs at least one --pick TEXT")

    print("loading one patient from the clinical Neo4j (read-only) ...")
    graph, table, stats = load_patient_graph(patient_index=args.patient, limit=args.limit)
    print(stats.summary())
    print(f"\n{graph}")
    n_nodes, n_texts = len(table), stats.distinct_embed_texts
    print(
        f"text multiplicity: {n_nodes} nodes share {n_texts} distinct embed texts "
        f"({n_nodes / max(n_texts, 1):.1f} nodes per text)"
    )

    if args.concepts or args.keys_for or args.draft_gold:
        # node_key lives on graph.nodes, the label/text columns on `table`.
        # Both are one row per node in the same order, so they join positionally.
        assert len(graph.nodes) == len(table), "node table and graph.nodes are misaligned"
        lookup = table.assign(node_key=graph.nodes["node_key"].to_numpy())
        if args.concepts:
            show_concepts(lookup, args.concepts, args.top)
        if args.keys_for:
            show_keys_for(lookup, args.keys_for)
        if args.draft_gold:
            draft_gold(lookup, args.pick, args.draft_gold, f"patient-{args.patient}")
    if not args.derive and not args.gold:
        return 0

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
