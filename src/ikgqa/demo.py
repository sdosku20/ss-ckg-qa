#!/usr/bin/env python
"""
demo.py
=======

Run the PCST retrieval on a toy graph and print every stage.

    python demo.py                                   # clinical graph, default question
    python demo.py --graph chain --query "creatinine measured and tacrolimus drug"
    python demo.py --graph hub --query "lab result"  # see tie-splitting
    python demo.py --sweep                           # how cost_e and topk change the answer
    python demo.py --list                            # show the toy graphs

Everything is deterministic: the toy encoder is a bag of words over the graph's
own vocabulary, so cosine similarity is just word overlap and you can check the
prizes by hand.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ikgqa.pcst import core as P
from ikgqa.data import toy as T

DEFAULT_QUESTIONS = {
    "clinical": "what lab result and code does patient 001 have for creatinine",
    "chain": "creatinine measured and tacrolimus drug",
    "hub": "lab result",
    "two-components": "creatinine tacrolimus",
    "single-node": "creatinine",
}


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def show_stage_1_and_2(kg, result, topk, topk_e):
    t = result.trace
    rule(f"STAGE 1+2  score every node and edge against the question, turn scores into prizes")
    print(f"\nnode prizes (top {min(topk + 4, len(kg.node_texts))}), prize = topk - rank:\n")
    print(t.prize_table(kg.nodes_df, top=topk + 4).to_string(index=False))
    print(f"\nedge prizes (top {min(topk_e + 4, len(kg.triples))}):\n")
    print(t.edge_table(kg.edges_df, top=topk_e + 4).to_string(index=False))

    n_zero = int((t.node_similarity == 0.0).sum())
    if n_zero > 1 and topk > 0:
        print(
            f"\n  note: {n_zero} nodes share a similarity of exactly 0.00, so any prize given to\n"
            f"  one of them is arbitrary tie-breaking, not evidence of relevance."
        )

    n_tiers = np.unique(t.edge_similarity).size
    if n_tiers < topk_e:
        print(
            f"\n  note: only {n_tiers} distinct edge similarities exist, so topk_e was "
            f"clamped from {topk_e} to {n_tiers}; the top tier is worth {n_tiers}, not {topk_e}."
        )


def show_stage_3(result):
    t = result.trace
    inst = t.instance
    rule("STAGE 3  make the problem solvable: lower the edge cost, add virtual nodes")
    print(f"\nedge cost requested: {t.cost_e_requested}")
    print(
        f"edge cost actually used: {t.cost_e_used:.4f}  "
        f"(= min(requested, max_edge_prize * (1 - c/2)), so at least one edge always wins)"
    )
    if t.cost_e_used < 0.05 * t.cost_e_requested:
        print(
            "  WARNING: the cost collapsed to nearly zero, which happens when no edge\n"
            "  matches the question at all. Every edge then becomes nearly free and you\n"
            "  retrieve the entire connected component. Try a question that shares words\n"
            "  with the relation names."
        )
    print(
        f"\nsolver sees: {inst.num_real_nodes} real nodes + {inst.num_virtual_nodes} virtual nodes, "
        f"{inst.num_real_edges} priced edges + {2 * inst.num_virtual_nodes} free half-edges"
    )
    if inst.virtual_node_map:
        vid, eid = next(iter(inst.virtual_node_map.items()))
        print(
            f"\nexample transform, original edge {eid} (prize "
            f"{t.edge_prizes[eid]:.4f} > cost {t.cost_e_used:.4f}):"
        )
        print(f"    before:  src --[cost {t.cost_e_used:.4f} - {t.edge_prizes[eid]:.4f} < 0  ILLEGAL]--> dst")
        print(f"    after:   src --[cost 0]--> v{vid} --[cost 0]--> dst")
        print(f"             prize(v{vid}) = {t.edge_prizes[eid]:.4f} - {t.cost_e_used:.4f} "
              f"= {inst.prizes[vid]:.4f}")


def show_stage_4(kg, result):
    t = result.trace
    rule("STAGE 4  solve, then translate the answer back to the real graph")
    print(f"\nsolver returned vertices: {sorted(t.solver_vertices.tolist())}")
    print(f"solver returned edges:    {sorted(t.solver_edges.tolist())}")
    virtual = [v for v in t.solver_vertices.tolist() if v >= t.instance.num_real_nodes]
    if virtual:
        print(f"  of those, {virtual} are virtual, i.e. they mean 'take original edge "
              f"{[t.instance.virtual_node_map[v] for v in virtual]}'")

    prized = set(np.flatnonzero(t.node_prizes > 0).tolist())
    chosen = set(t.selected_nodes.tolist())
    bridges = sorted(chosen - prized)
    print(f"\nnodes with a prize: {sorted(prized)}")
    print(f"nodes retrieved:    {sorted(chosen)}")
    if bridges:
        print(f"bridge nodes added purely to stay connected: {bridges}")
        for b in bridges[:5]:
            print(f"    node {b:>2} (prize 0.0): {kg.node_texts[b]}")

    print()
    print(P.summarise(result, kg.nodes_df, kg.edges_df))


def show_prompt(kg, result):
    rule("WHAT THE LLM ACTUALLY RECEIVES")
    full = kg.nodes_df.to_csv(index=False) + "\n" + kg.edges_df.to_csv(
        index=False, columns=["src", "edge_attr", "dst"]
    )
    print(f"\nwhole graph as text: {len(full):>5} characters")
    print(f"retrieved subgraph:  {len(result.desc):>5} characters "
          f"({100 * len(result.desc) / len(full):.0f}% of the original)\n")
    print(result.desc)


def sweep(kg, question):
    rule("SWEEP  the two dials that control what you get")
    q = kg.q(question)
    args = (kg.graph, q, kg.nodes_df, kg.edges_df)

    print("\ncost_e: how much every extra edge must earn to be worth including\n")
    rows = []
    for c in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        r = P.retrieval_via_pcst_traced(*args, topk=3, topk_e=3, cost_e=c)
        rows.append(
            {
                "cost_e asked": c,
                "cost_e used": round(r.trace.cost_e_used, 4),
                "nodes": len(r.selected_nodes),
                "edges": len(r.selected_edges),
                "chars": len(r.desc),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Note how 'cost_e used' stops rising: it is capped just below the best")
    print("  edge prize, so asking for a huge cost_e has no further effect.")

    print("\n\ntopk / topk_e: how many nodes and edges get a prize at all\n")
    rows = []
    for k in (0, 1, 3, 5, 10):
        r = P.retrieval_via_pcst_traced(*args, topk=k, topk_e=k, cost_e=0.5)
        rows.append(
            {
                "topk = topk_e": k,
                "nodes": len(r.selected_nodes),
                "edges": len(r.selected_edges),
                "chars": len(r.desc),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="clinical", choices=sorted(T.ALL_GRAPHS))
    ap.add_argument("--query", default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--topk-e", dest="topk_e", type=int, default=3)
    ap.add_argument("--cost-e", dest="cost_e", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true", help="show how the hyperparameters matter")
    ap.add_argument("--list", action="store_true", help="print the toy graphs and exit")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 44)

    if args.list:
        for name, factory in T.ALL_GRAPHS.items():
            rule(name)
            print(factory().pretty())
        return

    P.check_pcst_fast_sanity()

    kg = T.ALL_GRAPHS[args.graph]()
    question = args.query or DEFAULT_QUESTIONS[args.graph]

    rule(f"GRAPH: {kg.name}    QUESTION: {question!r}")
    print(kg.pretty())

    result = P.retrieval_via_pcst_traced(
        kg.graph, kg.q(question), kg.nodes_df, kg.edges_df,
        topk=args.topk, topk_e=args.topk_e, cost_e=args.cost_e,
    )

    if result.trace.short_circuited:
        rule("SHORT CIRCUIT")
        print(
            "\nThis graph has an empty node or edge table, so `retrieval_via_pcst` returns it\n"
            "untouched without scoring anything. The whole graph is the answer."
        )
        show_prompt(kg, result)
        return

    show_stage_1_and_2(kg, result, args.topk, args.topk_e)
    show_stage_3(result)
    show_stage_4(kg, result)
    show_prompt(kg, result)

    if args.sweep:
        sweep(kg, question)


if __name__ == "__main__":
    main()
