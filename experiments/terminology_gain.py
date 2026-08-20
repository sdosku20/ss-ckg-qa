"""
Does the terminology cross-walk make unreachable nodes reachable?

    python experiments/terminology_gain.py            # full patient
    python experiments/terminology_gain.py --small    # faster

Measures answer-node recall on diagnosis questions in four conditions:

  * no catalogue                 -- the baseline the graph gives you today
  * catalogue, exact tiers only   -- the honest gain
  * catalogue + hierarchy words   -- RQ3's schema signal, cheapest version
  * catalogue in the other language

The last condition is the point of the experiment. A cross-walk supplies
descriptions in whatever language its catalogue is written in, and a lexical or
English-only encoder cannot bridge a language gap. So the gain depends on a fact
nobody has confirmed yet: *what language do users ask questions in?* Running the
2x2 turns that open question into a number, which is a better thing to bring to a
supervisor than a guess.

Everything here runs on the synthetic replica. It measures mechanism -- whether
resolving a code changes what can be retrieved -- and not clinical quality.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from ikgqa.data.replica import (
    PATIENT_0,
    PATIENT_0_SMALL,
    build_replica,
    planted_questions,
    synthetic_catalogue,
)
from ikgqa.data.replica import generate_rows
from ikgqa.data.sphn import default_encoder
from ikgqa.eval.metrics import evaluate
from ikgqa.retrieval import PCST, TopKTriples

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def run_condition(
    profile,
    seed: int,
    question_language: str,
    catalogue_language: str | None,
    expand_hierarchy: bool,
    n_questions: int,
) -> tuple[pd.DataFrame, str]:
    """Build the graph under one condition and score the diagnosis questions."""
    rows_only = generate_rows(profile, seed=seed)
    catalogue = (
        synthetic_catalogue(rows_only, language=catalogue_language)
        if catalogue_language
        else None
    )
    graph, table, _, rows = build_replica(
        profile, seed=seed, terminology=catalogue, expand_hierarchy=expand_hierarchy
    )
    encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
    questions = [
        q
        for q in planted_questions(
            graph, table, rows, n_per_kind=n_questions, language=question_language
        )
        if q.qid.startswith("dx-")
    ]
    results = evaluate(
        graph,
        [
            PCST(topk=10, topk_e=0, cost_e=0.5, tie_break="stable"),
            TopKTriples(k=10, encoder=encoder),
        ],
        questions,
        encoder,
    )
    report = ""
    if isinstance(rows.get("_reports"), dict):
        report = rows["_reports"]["diagnoses"].summary()
    return results, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--questions", type=int, default=20, help="per kind")
    args = parser.parse_args(argv)

    profile = PATIENT_0_SMALL if args.small else PATIENT_0
    tag = "small" if args.small else "full"

    conditions = [
        ("no catalogue", "de", None, False),
        ("catalogue (de)", "de", "de", False),
        ("catalogue (de) + hierarchy", "de", "de", True),
        ("catalogue (en), question de", "de", "en", False),
        ("catalogue (en), question en", "en", "en", False),
        ("no catalogue, question en", "en", None, False),
    ]

    frames = []
    for name, q_lang, cat_lang, expand in conditions:
        print(f"\n=== {name} ===", flush=True)
        results, report = run_condition(
            profile, args.seed, q_lang, cat_lang, expand, args.questions
        )
        if report:
            print(report)
        results = results.assign(
            condition=name,
            question_language=q_lang,
            catalogue_language=cat_lang or "-",
            hierarchy=expand,
            qkind=results["qid"].str.replace(r"-\d+$", "", regex=True),
        )
        frames.append(results)
        summary = results.groupby(["retriever", "qkind"])["answer_node_recall"].mean()
        print(summary.round(4).to_string())

    everything = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"terminology_gain_{tag}.csv"
    everything.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    print("\n\n=== answer-node recall on diagnosis questions ===")
    pivot = everything.pivot_table(
        index="condition",
        columns=["retriever", "qkind"],
        values="answer_node_recall",
        aggfunc="mean",
        sort=False,
    )
    print(pivot.round(3).to_string())
    print(f"\nwrote {out}")

    # Per retriever, never averaged across them: mixing a method that recovers
    # the node with one that does not, and reporting the mean, would describe
    # neither.
    print("\ncode-only diagnoses, answer-node recall by retriever:")
    codeonly = everything[everything["qkind"] == "dx-codeonly"]
    named = everything[everything["qkind"] == "dx-named"]
    for retriever, group in codeonly.groupby("retriever"):
        def mean_for(frame, condition):
            return frame[
                (frame["retriever"] == retriever) & (frame["condition"] == condition)
            ]["answer_node_recall"].mean()

        print(f"  {retriever}")
        print(f"    no catalogue                    {mean_for(group, 'no catalogue'):.3f}")
        print(f"    catalogue, language matches     {mean_for(group, 'catalogue (de)'):.3f}")
        print(
            "    catalogue, language differs     "
            f"{mean_for(group, 'catalogue (en), question de'):.3f}"
        )
        print(
            "    named diagnoses, same condition "
            f"{mean_for(named, 'catalogue (en), question de'):.3f}"
            "   <- was 1.000 before enrichment"
        )
    print(
        "\nThe last line is the warning: applying a catalogue in the wrong language\n"
        "does not merely fail to help, it replaces text that was already matching\n"
        "and makes previously reachable nodes unreachable. So the language users\n"
        "actually ask in is a blocking question, not a detail."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
