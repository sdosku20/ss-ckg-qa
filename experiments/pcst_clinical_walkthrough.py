"""
From clinical text to a selected subgraph: every stage, printed.

A fourteen-node patient graph with real SPHN-shaped labels goes in, a question
goes in, and each stage of the actual retriever prints what it produced. Nothing
here is a reimplementation: it calls `ikgqa.pcst.core.retrieval_via_pcst_traced`,
the same function the experiments use.

    PCST\\.venv\\Scripts\\python.exe experiments/pcst_clinical_walkthrough.py

Stages, in the order they run:

    1  the graph, as text
    2  vocabulary          words -> dimensions
    3  encoding            text -> unit vectors you can check by hand
    4  similarity          cosine of question against every node
    5  node prizes         rank -> prize, and what that discards
    6  edge costs          why one flat number here
    7  the solver instance what pcst_fast actually receives
    8  the answer          solver output, then decoding, then the subgraph
    9  optimality          brute force over every connected subtree
   10  identical text      what ties do to the answer
   11  terminology fix     the 0.00 -> 0.80 result, in miniature
   12  edge costs          why they were all equal, and when they differ
   13  the LLM's input

Companion to `docs/pcst_from_the_papers.md`. The four-node algorithmic trace is
in `experiments/pcst_by_hand.py`; this one is about where the numbers come from.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ikgqa.encoders import BagOfWordsEncoder, tokenize  # noqa: E402
from ikgqa.pcst.core import (  # noqa: E402
    SimpleGraph,
    adjust_edge_cost,
    compute_edge_prizes,
    retrieval_via_pcst_traced,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 52)

# ---------------------------------------------------------------------------
# The graph. Labels follow SPHN shapes: a Case ties everything together, drugs
# reach the case through an administration event, diagnoses hang off the case.
# Node 6 carries only an ICD-10 code, which is the 80%-of-diagnoses situation
# measured on the real graph.
# ---------------------------------------------------------------------------

NODES = [
    "Inpatient case cardiology ward",              # 0  the hub every path crosses
    "Chronic heart failure reduced ejection fraction",  # 1
    "Type 2 diabetes mellitus",                    # 2
    "E11.9",                                       # 3  code-only diagnosis
    "Bisoprolol 5 mg film-coated tablet",          # 4
    "Metformin 1000 mg film-coated tablet",        # 5
    "Furosemide 40 mg film-coated tablet",         # 6
    "Drug administration event",                   # 7  identical text ...
    "Drug administration event",                   # 8  ... to node 7 ...
    "Drug administration event",                   # 9  ... and node 8
    "Creatinine 118 umol/L",                       # 10
    "NT-proBNP 2840 ng/L",                         # 11
    "HbA1c 7.9 percent",                           # 12
    "Data provider University Hospital Basel",     # 13 provenance, like the real hubs
]

# (src, relation text, dst)
EDGE_LIST = [
    (0, "hasDiagnosis", 1),
    (0, "hasDiagnosis", 2),
    (0, "hasDiagnosis", 3),
    (0, "hasDrugAdministration", 7),
    (0, "hasDrugAdministration", 8),
    (0, "hasDrugAdministration", 9),
    (7, "administeredDrug", 4),
    (8, "administeredDrug", 5),
    (9, "administeredDrug", 6),
    (0, "hasLabResult", 10),
    (0, "hasLabResult", 11),
    (0, "hasLabResult", 12),
    (0, "hasDataProvider", 13),
    (1, "hasDataProvider", 13),
]

QUESTION = "Which drug was administered for chronic heart failure?"

TOPK = 4
COST_E = 0.5


# The same graph after a terminology cross-walk has resolved the drug codes to
# descriptions. Only nodes 4, 5 and 6 differ from NODES above.
RESOLVED_NODES = list(NODES)
RESOLVED_NODES[4] = "Bisoprolol beta blocker drug for chronic heart failure"
RESOLVED_NODES[5] = "Metformin biguanide drug for type 2 diabetes mellitus"
RESOLVED_NODES[6] = "Furosemide loop diuretic drug for oedema"


# SPHN stores relation names in camelCase, which the tokeniser sees as one
# opaque word. Humanising them is a graph-preparation choice, and section 12
# shows it decides which of two very different edge-prize paths you land on.
HUMAN_RELATIONS = {
    "hasDiagnosis": "has diagnosis",
    "hasDrugAdministration": "has drug administration",
    "administeredDrug": "administered drug",
    "hasLabResult": "has lab result",
    "hasDataProvider": "has data provider",
}


def build(nodes=None, drug_order=(4, 5, 6), relation_text=None):
    """Assemble the graph, the frames the retriever wants, and the encoder.

    `drug_order` decides which drug node hangs off which administration event.
    The three events are textually identical, so permuting this changes nothing
    a retriever could observe -- which is exactly what section 10 exploits.

    `relation_text` maps a relation name to the string that gets embedded. Pass
    HUMAN_RELATIONS to embed readable phrases instead of camelCase.
    """
    nodes = list(NODES if nodes is None else nodes)
    edges = []
    admin_seen = 0
    for src, rel, dst in EDGE_LIST:
        if rel == "administeredDrug":
            dst = drug_order[admin_seen]
            admin_seen += 1
        edges.append((src, rel, dst))

    relation_text = relation_text or {}
    edge_texts = [relation_text.get(rel, rel) for _, rel, _ in edges]
    encoder = BagOfWordsEncoder(nodes + edge_texts + [QUESTION])

    graph = SimpleGraph(
        x=encoder.encode(nodes),
        edge_index=np.array([[s for s, _, _ in edges], [d for _, _, d in edges]], dtype=np.int64),
        edge_attr=encoder.encode(edge_texts),
        num_nodes=len(nodes),
    )
    textual_nodes = pd.DataFrame({"node_id": range(len(nodes)), "node_attr": nodes})
    textual_edges = pd.DataFrame(
        {
            "src": [s for s, _, _ in edges],
            "edge_attr": edge_texts,
            "dst": [d for _, _, d in edges],
        }
    )
    return graph, textual_nodes, textual_edges, encoder, nodes


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def main() -> None:
    graph, tnodes, tedges, encoder, nodes = build()

    head(1, "THE GRAPH")
    print(f"question: {QUESTION!r}\n")
    for i, text in enumerate(nodes):
        print(f"  node {i:2d}  {text}")
    print()
    for i, (s, rel, d) in enumerate(EDGE_LIST):
        print(f"  edge {i:2d}  {s:2d} --{rel}--> {d}")
    print(
        "\nNote the shape: every drug reaches the diagnosis only THROUGH node 0.\n"
        "That is the clinical version of a Steiner node, and it is why a ranked\n"
        "list of nodes cannot answer this question."
    )

    head(2, "VOCABULARY  (text -> dimensions)")
    print(f"tokenising node texts + relation texts + the question -> {len(encoder.vocab)} dimensions")
    print(f"the question tokenises to: {tokenize(QUESTION)}")
    print("\nEach dimension IS a word, so every number below is checkable by hand.")

    head(3, "ENCODING  (one vector per text)")
    q_emb = encoder.encode_one(QUESTION)
    inv = {v: k for k, v in encoder.vocab.items()}
    print("The question vector, non-zero entries only:")
    for j in np.nonzero(q_emb)[0]:
        print(f"    {inv[j]:<14s} {q_emb[j]:.4f}")
    print(
        f"\n  {len(np.nonzero(q_emb)[0])} words, each 1/sqrt(n) after L2 normalisation.\n"
        "  Normalising is what makes cosine similarity a plain dot product."
    )
    print("\nNode 1, for comparison:")
    for j in np.nonzero(graph.x[1])[0]:
        print(f"    {inv[j]:<14s} {graph.x[1][j]:.4f}")

    # ---- run the real retriever, edge prizes off (as the experiments do) ----
    result = retrieval_via_pcst_traced(
        graph, q_emb, tnodes, tedges, topk=TOPK, topk_e=0, cost_e=COST_E, tie_break="stable"
    )
    tr = result.trace

    head(4, "SIMILARITY  (cosine of the question against every node)")
    sim = tr.node_similarity
    order = np.argsort(-sim, kind="stable")
    print(f"{'node':>5}  {'cos':>7}   text")
    for i in order:
        print(f"{i:5d}  {sim[i]:7.4f}   {nodes[i]}")
    shared = set(tokenize(QUESTION)) & set(tokenize(nodes[order[0]]))
    print(f"\n  Top node scores {sim[order[0]]:.4f} on the shared words {sorted(shared)}.")
    print("  Every node scoring 0.0000 shares no word at all with the question.")
    print("  Node 4 (Bisoprolol) is the true answer and scores 0.0000 -- no wording")
    print("  connects a drug's brand name to the phrase 'heart failure'. This is the")
    print("  effect the thesis measures as code-only questions scoring 0.00 recall.")

    head(5, "NODE PRIZES  (rank, not value)")
    print(f"prize(v) = k - i for the i-th best node, k = {TOPK}; everyone else gets 0\n")
    print(f"{'node':>5}  {'cos':>7}  {'prize':>6}   text")
    for i in order:
        if tr.node_prizes[i] > 0 or sim[i] > 0:
            print(f"{i:5d}  {sim[i]:7.4f}  {tr.node_prizes[i]:6.1f}   {nodes[i]}")
    print(
        "\n  What this throws away: the gap between rank 1 and rank 2 is always exactly\n"
        "  one prize unit, whether the encoder thought them near-identical or wildly\n"
        "  different. Only the ORDER survives."
    )

    head(6, "EDGE COSTS")
    print(f"topk_e = 0, so every edge prize is 0 and every edge costs the flat cost_e = {tr.cost_e_used}.")
    print("This is a CONSEQUENCE of topk_e = 0, not a demo simplification -- see section 12,")
    print("which turns edge prizes on and shows costs genuinely differing.")

    head(7, "THE SOLVER INSTANCE  (what pcst_fast receives)")
    inst = tr.instance
    print(f"  nodes           {len(inst.prizes)}  ({inst.num_real_nodes} real + {inst.num_virtual_nodes} virtual)")
    print(f"  edges           {inst.edges.shape[0]}")
    print(f"  prizes          {np.array2string(inst.prizes, precision=1)}")
    print(f"  costs           {np.array2string(np.unique(inst.costs), precision=3)}  (all identical here)")
    print("  root -1 (unrooted), num_clusters 1 (one connected piece), pruning 'gw'")

    head(8, "THE ANSWER")
    print(f"  solver returned vertices  {sorted(int(v) for v in tr.solver_vertices)}")
    print(f"  solver returned edges     {sorted(int(e) for e in tr.solver_edges)}")
    print(f"  after decoding, nodes     {sorted(int(v) for v in tr.selected_nodes)}")
    print(f"  after decoding, edges     {sorted(int(e) for e in tr.selected_edges)}")
    print("\nThe selected subgraph:")
    for i in sorted(int(v) for v in tr.selected_nodes):
        prize = tr.node_prizes[i]
        why = "prize" if prize > 0 else "STEINER NODE: prize 0, kept only to connect"
        print(f"    node {i:2d}  prize {prize:4.1f}  {nodes[i]:<48s} {why}")
    for e in sorted(int(e) for e in tr.selected_edges):
        s, rel, d = EDGE_LIST[e]
        print(f"    edge {e:2d}  {s:2d} --{rel}--> {d}")

    head(9, "IS IT OPTIMAL?  (brute force over every connected subtree)")
    best = brute_force(inst)
    got = frozenset(int(v) for v in tr.solver_vertices)
    print(f"  enumerated {best['count']} connected subtrees of the solver instance")
    print(f"  best objective        {best['value']:.3f}  on nodes {sorted(best['nodes'])}")
    print(f"  what the solver got   {objective(inst, got, best['edges_for'](got)):.3f}  on nodes {sorted(got)}")
    print(
        "\n  THIS is how we agreed {a,b,c} was best in the four-node example: exhaustive\n"
        "  enumeration. It works at 14 nodes and at 4. It does NOT work at 15,810 --\n"
        "  2^45562 subsets. On the real graph there is no optimum to compare against,\n"
        "  and the factor-2 guarantee is the only thing standing between the returned\n"
        "  subgraph and the unknown best one. That is exactly why the guarantee matters."
    )

    head(10, "IDENTICAL TEXT  (nodes 7, 8, 9 all read 'Drug administration event')")
    print(f"  cosine for node 7  {sim[7]:.10f}")
    print(f"  cosine for node 8  {sim[8]:.10f}")
    print(f"  cosine for node 9  {sim[9]:.10f}")
    print(f"  bit-identical?     {sim[7] == sim[8] == sim[9]}")
    print(
        f"\n  and yet they receive prizes {tr.node_prizes[7]:.0f}, {tr.node_prizes[8]:.0f} and "
        f"{tr.node_prizes[9]:.0f}.\n"
        "  The prize function demands a strict ranking, so it invents one. Nothing in\n"
        "  the data says node 7 beats node 9; the sort order does."
    )
    print("\n  With a smaller budget the invented order becomes a hard cut:\n")
    for k in (2, 3, 4):
        r = retrieval_via_pcst_traced(
            graph, q_emb, tnodes, tedges, topk=k, topk_e=0, cost_e=COST_E, tie_break="stable"
        )
        prized = [i for i in (7, 8, 9) if r.trace.node_prizes[i] > 0]
        print(
            f"    topk={k}   prized admin nodes {prized}   "
            f"subgraph {sorted(int(v) for v in r.trace.selected_nodes)}"
        )
    print(
        "\n  At topk=2 node 7 is in and nodes 8 and 9 are out, on no evidence whatever.\n"
        "  Both tie-breaks we ship happen to agree here (both take the lowest index),\n"
        "  so the danger is not disagreement between them -- it is that the answer\n"
        "  follows STORAGE ORDER. Re-attach the same three drugs to the same three\n"
        "  indistinguishable events in a different order and the winner changes:\n"
    )
    for perm in ((4, 5, 6), (6, 5, 4)):
        g2, tn2, te2, enc2, _ = build(drug_order=perm)
        r = retrieval_via_pcst_traced(
            g2, enc2.encode_one(QUESTION), tn2, te2, topk=2, topk_e=0, cost_e=COST_E, tie_break="stable"
        )
        chosen = [i for i in (7, 8, 9) if r.trace.node_prizes[i] > 0]
        behind = [d for s, _, d in te2.itertuples(index=False) if s in chosen]
        names = [nodes[b] for b in behind]
        print(f"    drug order {perm}  ->  admin node {chosen} selected, one hop behind it: {names}")
    print(
        "\n  Same graph, same question, same code. A different row order out of Neo4j\n"
        "  puts a different drug behind the selected event. On the real graph 15,810\n"
        "  nodes carry 614 distinct texts, about 26 nodes per text, so this is the\n"
        "  normal case and not an edge case. It is Prediction 2 in the thesis, and it\n"
        "  is why the loader stores a stable `node_key` instead of trusting the index."
    )

    head(11, "THE TERMINOLOGY FIX  (0.00 -> retrieved, in miniature)")
    print(
        "Above, Bisoprolol scored 0.0000 and never came back. That is not a PCST\n"
        "failure -- no retriever can select a node whose text shares nothing with the\n"
        "question. Now resolve the ATC codes to descriptions, as the cross-walk does,\n"
        "and change nothing else:\n"
    )
    g3, tn3, te3, enc3, nodes3 = build(nodes=RESOLVED_NODES)
    q3 = enc3.encode_one(QUESTION)
    r3 = retrieval_via_pcst_traced(g3, q3, tn3, te3, topk=TOPK, topk_e=0, cost_e=COST_E, tie_break="stable")
    for i in np.argsort(-r3.trace.node_similarity, kind="stable")[:5]:
        star = "  <-- the answer" if i == 4 else ""
        print(f"    node {i:2d}  cos {r3.trace.node_similarity[i]:.4f}  {nodes3[i]}{star}")
    got3 = sorted(int(v) for v in r3.trace.selected_nodes)
    print(f"\n  subgraph now {got3}")
    print(f"  contains Bisoprolol (node 4)?   {4 in got3}")
    print(
        "\n  Bisoprolol went from 0.0000 to a top rank and is now returned. The retrieval\n"
        "  strategy did not change; the TEXT did. That is the 0.00 -> 0.80 result in the\n"
        "  thesis, and why it is reported as graph preparation rather than retrieval."
    )

    head(12, "WHERE DIFFERENT EDGE COSTS COME FROM")
    print(
        "G-Retriever takes ONE scalar cost_e for the whole graph -- there is no\n"
        "per-edge cost input. An edge's effective cost is cost_e minus its PRIZE:\n\n"
        "    prize <= cost_e   ->  cost becomes cost_e - prize\n"
        "    prize >  cost_e   ->  edge becomes a free pair of half-edges plus a\n"
        "                          virtual node worth (prize - cost_e)\n\n"
        "So costs differ only where prizes differ. With topk_e = 0 every prize is 0,\n"
        "every cost is exactly cost_e, and that is why section 6 showed one number.\n"
        "It is a consequence of topk_e = 0, not a simplification for the demo.\n"
    )

    print("--- path A: readable relation text, the ordinary tier split ---\n")
    gA, tnA, teA, encA, _ = build(relation_text=HUMAN_RELATIONS)
    qA = encA.encode_one(QUESTION)
    simA, przA = compute_edge_prizes(qA, gA.edge_attr, topk_e=3)
    ceA = adjust_edge_cost(przA, COST_E)
    print(f"  distinct edge similarities: {np.unique(simA)[::-1].round(4)}  -> 3 tiers\n")
    print(f"  {'relation':<24s} {'uses':>4}  {'cos':>7}  {'prize':>7}   effective cost")
    seen: set = set()
    for i, (_, rel, _) in enumerate(EDGE_LIST):
        if rel in seen:
            continue
        seen.add(rel)
        uses = sum(1 for _, r2, _ in EDGE_LIST if r2 == rel)
        if przA[i] > ceA:
            eff = f"FREE  (+ virtual prize {przA[i] - ceA:.4f})"
        else:
            eff = f"{ceA - przA[i]:.4f}"
        print(f"  {HUMAN_RELATIONS[rel]:<24s} {uses:>4}  {simA[i]:7.4f}  {przA[i]:7.4f}   {eff}")
    print(
        f"\n  cost_e stayed {ceA:.4f}. Tier k gets budget (3 - k) SPLIT across its members:\n"
        "  the three 'administered drug' edges share a budget of 3, so 1.0 each; the\n"
        "  eight zero-similarity edges share a budget of 1, so 0.125 each.\n\n"
        "  THERE is your answer: costs now genuinely differ, 0.3750 against free. The\n"
        "  relations the question talks about got cheap, the rest stayed expensive.\n"
        "  And note the bias -- 'administered drug' is worth 1.0 each because it is used\n"
        "  three times. Used three thousand times it would be worth 0.001 each. A\n"
        "  relation's value falls as it becomes more common, whatever the question."
    )

    print("\n--- path B: camelCase relation text, as SPHN stores it ---\n")
    r = retrieval_via_pcst_traced(
        graph, q_emb, tnodes, tedges, topk=TOPK, topk_e=3, cost_e=COST_E, tie_break="stable"
    )
    ti = r.trace
    print(f"  distinct edge similarities: {np.unique(ti.edge_similarity)}  -> ONE tier, all zero")
    print(f"  every edge prize          {ti.edge_prizes[0]:.6f}   (= 1/14, the split of a budget of 1)")
    print(f"  cost_e requested {ti.cost_e_requested}  ->  used {ti.cost_e_used:.6f}   (the gamma cap)")
    print(f"  virtual nodes created     {ti.instance.num_virtual_nodes} of {len(EDGE_LIST)} edges")
    print(f"  nodes returned            {len(ti.selected_nodes)} of {len(nodes)}")
    print(f"  nodes returned at topk_e=0    {len(tr.selected_nodes)} of {len(nodes)}")
    print(
        "\n  5 nodes became all 14. But be precise about WHY, because it is not the same\n"
        "  trigger as the real graph. camelCase tokenises to one opaque word, so no\n"
        "  relation shares anything with the question and every similarity is exactly\n"
        "  0.0. That fires NOTE-A in compute_edge_prizes: the tier's value IS 0.0, so\n"
        "  the membership test `prize == 0.0` matches every edge, and all 14 collect\n"
        "  1/14. The cap then crushes cost_e below that, so every edge turns virtual.\n\n"
        "  On the real graph the route is different and duller: `hasCode` genuinely\n"
        "  occurs 4,883,809 times, so a real top tier splits its budget a few million\n"
        "  ways and lands near 1e-6. Different trigger, same endpoint -- every edge\n"
        "  ends up a virtual node with a tiny positive prize behind free half-edges,\n"
        "  and the decoder then closes the node set over every recovered edge.\n\n"
        "  Worth stating in the thesis: whether you hit the quirk or the tier path\n"
        "  depends on whether relation names were humanised, which is a preparation\n"
        "  decision, not a retrieval one."
    )

    head(13, "WHAT THE LLM WOULD RECEIVE")
    print(result.desc)


# ---------------------------------------------------------------------------
# Exhaustive optimality check on the solver instance.
# ---------------------------------------------------------------------------


def objective(inst, nodes: frozenset, edges: tuple) -> float:
    """c(T) + pi(complement of T), the quantity pcst_fast minimises."""
    cost = float(sum(inst.costs[e] for e in edges))
    forfeited = float(sum(p for i, p in enumerate(inst.prizes) if i not in nodes))
    return cost + forfeited


def brute_force(inst) -> dict:
    """Score every connected subtree. Only tractable because the graph is tiny."""
    m = inst.edges.shape[0]
    if m > 20:
        return {"count": 0, "value": float("nan"), "nodes": frozenset(), "edges_for": lambda s: ()}

    best_value, best_nodes, best_edges, count = float("inf"), frozenset(), (), 0
    for r in range(m + 1):
        for chosen in itertools.combinations(range(m), r):
            nodes = frozenset(int(v) for e in chosen for v in inst.edges[e])
            if r and not _connected(inst, nodes, chosen):
                continue
            count += 1
            value = objective(inst, nodes, chosen)
            if value < best_value:
                best_value, best_nodes, best_edges = value, nodes, chosen
    # single nodes are legal trees too
    for v in range(len(inst.prizes)):
        value = objective(inst, frozenset({v}), ())
        count += 1
        if value < best_value:
            best_value, best_nodes, best_edges = value, frozenset({v}), ()

    def edges_for(sel: frozenset) -> tuple:
        return tuple(e for e in range(m) if all(int(x) in sel for x in inst.edges[e]))

    return {"count": count, "value": best_value, "nodes": best_nodes, "edges_for": edges_for}


def _connected(inst, nodes: frozenset, chosen: tuple) -> bool:
    if not nodes:
        return True
    adj: dict = {v: set() for v in nodes}
    for e in chosen:
        u, v = (int(x) for x in inst.edges[e])
        adj[u].add(v)
        adj[v].add(u)
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == set(nodes)


if __name__ == "__main__":
    main()
