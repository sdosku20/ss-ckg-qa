"""
Does G-Retriever's PCST transfer to a graph shaped like the STCS clinical KG?

Run:  python -m experiments.synthetic_clinical

This uses NO clinical data. It builds a synthetic graph that reproduces three
structural properties measured from the real STCS graph on 19 Aug 2026, and
tests whether the published method still behaves as intended on that shape.

Measured properties being reproduced
------------------------------------
1. Text is shared at enormous multiplicity. 4,832,744 LabTest nodes resolve to
   only 544 distinct code descriptions -- so cosine similarity can take just 544
   values across millions of nodes.
2. Only 29 relation types exist across 72,218,890 edges, and they are extremely
   skewed (hasSourceSystem 11.7M, hasInstitutionCode 1).
3. Administrative hubs dominate connectivity: SourceSystem averages degree
   69,525 over 168 nodes; BodySite averages 575,814 over 6.

Hypotheses
----------
H1  Edge prizes collapse. compute_edge_prizes splits each tier's budget across
    all edges tied in that tier, so a relation shared by millions of edges is
    worth ~0 each. adjust_edge_cost then lowers cost_e to just under that
    maximum, making every edge nearly free and destroying size control.
H2  Node selection is decided by tie-breaking. With 544 distinct texts, the
    top-k is chosen among thousands of exactly-tied nodes, so which ones win is
    an artefact of the sort, not of relevance.
H3  Hub pruning changes the retrieved subgraph materially, because hubs give
    PCST a cheap route between otherwise distant nodes.

Each hypothesis prints VERDICT: SUPPORTED or NOT SUPPORTED with the numbers
behind it. A refuted hypothesis is as useful as a confirmed one -- the point is
to know before building the real pipeline on the assumption.
"""

from __future__ import annotations

import numpy as np

from ikgqa.encoders import BagOfWordsEncoder
from ikgqa.graph import TextualGraph
from ikgqa.retrieval import PCST

# Scaled-down but ratio-faithful: one patient's neighbourhood is ~18,000 nodes
# in the real graph (21,537,305 nodes / 1,197 patients).
N_LAB_EVENTS = 1500
N_LAB_CODES = 40          # real: 544 distinct codes; scaled to keep the ratio brutal
N_DIAGNOSES = 12
N_DRUGS = 20

LAB_NAMES = [f"lab test long name {i}" for i in range(N_LAB_CODES)]
DIAG_NAMES = [f"diagnosis long name {i}" for i in range(N_DIAGNOSES)]
DRUG_NAMES = [f"drug generic name {i}" for i in range(N_DRUGS)]

# The real relation-count skew, normalised. Only the ratios matter here.
RELATION_SKEW = {
    "hasSourceSystem": 11_680_076,
    "hasSubjectPseudoIdentifier": 11_600_854,
    "hasSample": 4_832_744,
    "hasLabTest": 4_832_744,
    "hasResult": 4_832_744,
    "hasCode": 4_883_809,
    "hasDrug": 661_808,
    "hasInstitutionCode": 1,
}


def build_graph(*, with_hubs: bool, seed: int = 0):
    """A single patient's neighbourhood, in the real graph's shape.

    Returns (graph, meta) where meta records which node ids are which kind, so
    the experiment can ask questions like "was the right lab test found".
    """
    rng = np.random.default_rng(seed)
    texts: list = []
    triples: list = []
    kind: dict = {}

    def add(text: str, what: str) -> int:
        texts.append(text)
        kind[len(texts) - 1] = what
        return len(texts) - 1

    patient = add("subject pseudo identifier patient", "patient")
    source_system = add("source system provenance record", "hub") if with_hubs else None
    body_site = add("body site", "hub") if with_hubs else None

    lab_tests: list = []
    for i in range(N_LAB_EVENTS):
        code_ix = int(rng.integers(0, N_LAB_CODES))
        # LabTest text comes from its Code: identical for every test of a type.
        test = add(LAB_NAMES[code_ix], "lab_test")
        # LabResult text carries a value, so it is nearly unique.
        result = add(f"lab result value {rng.integers(1, 400) / 10} mg per dl", "lab_result")
        event = add(f"lab test event {i}", "lab_event")
        sample = add(f"sample {i}", "sample")
        lab_tests.append((test, code_ix))

        triples += [
            (event, "hasLabTest", test),
            (event, "hasSample", sample),
            (test, "hasResult", result),
            (event, "hasSubjectPseudoIdentifier", patient),
        ]
        if with_hubs:
            triples += [
                (event, "hasSourceSystem", source_system),
                (sample, "hasSourceSystem", source_system),
                (sample, "hasBodySite", body_site),
            ]

    for i in range(N_DIAGNOSES):
        dx = add(DIAG_NAMES[i], "diagnosis")
        triples.append((dx, "hasSubjectPseudoIdentifier", patient))
        if with_hubs:
            triples.append((dx, "hasSourceSystem", source_system))

    for i in range(N_DRUGS):
        drug = add(DRUG_NAMES[i], "drug")
        event = add(f"drug administration event {i}", "drug_event")
        triples += [
            (event, "hasDrug", drug),
            (event, "hasSubjectPseudoIdentifier", patient),
        ]

    encoder = BagOfWordsEncoder(texts + [r for _, r, _ in triples])
    graph = TextualGraph.from_texts(texts, triples, encoder,
                                    name=f"synthetic({'hubs' if with_hubs else 'pruned'})")
    return graph, {"encoder": encoder, "kind": kind, "lab_tests": lab_tests, "patient": patient}


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def h1_edge_prizes(graph, meta) -> None:
    rule("H1  Do edge prizes collapse when relation types are few and skewed?")
    q = meta["encoder"].encode_one("lab test long name 7")

    retriever = PCST(topk=5, topk_e=5, cost_e=0.5)
    result = retriever.trace(graph, q)
    t = result.trace

    distinct_edge_sims = len(np.unique(t.edge_similarity))
    print(f"distinct edge similarity values : {distinct_edge_sims}  "
          f"(over {graph.num_edges} edges)")
    print(f"largest edge prize              : {t.edge_prizes.max():.3e}")
    print(f"cost_e requested                : {t.cost_e_requested}")
    print(f"cost_e actually used            : {t.cost_e_used:.3e}")
    print(f"ratio used/requested            : {t.cost_e_used / t.cost_e_requested:.3e}")
    print(f"retrieved                       : {len(result.selected_nodes)} nodes, "
          f"{len(result.selected_edges)} edges")

    collapsed = t.cost_e_used < t.cost_e_requested / 100
    print(f"\nVERDICT: {'SUPPORTED' if collapsed else 'NOT SUPPORTED'} -- "
          f"cost_e was driven {'to a small fraction of' if collapsed else 'close to'} "
          "the requested value by adjust_edge_cost.")
    if collapsed:
        print("Consequence: every edge costs almost nothing, so the size dial stops")
        print("controlling size. Use topk_e=0 and set cost_e explicitly, or give edges")
        print("composed (distinguishable) text.")


def h2_ties(graph, meta) -> None:
    rule("H2  Is the top-k decided by tie-breaking rather than by similarity?")
    q = meta["encoder"].encode_one("lab test long name 7")
    from ikgqa.pcst import core as P

    sim = P.cosine_similarity(q, graph.node_emb)
    top = sim.max()
    tied = int((sim == top).sum())
    print(f"nodes sharing the single highest similarity : {tied}")
    print(f"distinct similarity values across the graph : {len(np.unique(sim))} "
          f"(over {graph.num_nodes} nodes)")

    a = PCST(topk=5, topk_e=0, cost_e=0.5, tie_break="stable").retrieve(graph, q)
    b = PCST(topk=5, topk_e=0, cost_e=0.5, tie_break="auto").retrieve(graph, q)
    same = set(a.node_ids.tolist()) == set(b.node_ids.tolist())
    print(f"stable vs torch tie-break give same nodes   : {same}")

    print(f"\nVERDICT: {'SUPPORTED' if tied > 5 else 'NOT SUPPORTED'} -- "
          f"{tied} nodes are exactly tied for first place, so a top-5 must pick "
          "5 of them arbitrarily.")
    if tied > 5:
        print("Consequence: on this graph shape, similarity identifies the concept TYPE.")
        print("Which instance is returned is not a semantic decision at all, so instance")
        print("selection has to come from structure or time (candidate generation).")


def h3_hubs() -> None:
    rule("H3  Do administrative hubs change what PCST retrieves?")
    rows = []
    for with_hubs in (True, False):
        graph, meta = build_graph(with_hubs=with_hubs)
        q = meta["encoder"].encode_one("lab test long name 7")
        r = PCST(topk=5, topk_e=0, cost_e=0.5).retrieve(graph, q)
        hub_ids = {i for i, k in meta["kind"].items() if k == "hub"}
        hubs_in = len(set(r.node_ids.tolist()) & hub_ids)
        rows.append((with_hubs, graph.num_nodes, graph.num_edges,
                     r.num_nodes, r.num_edges, hubs_in))

    print(f"{'hubs':>6} | {'graph nodes':>11} | {'graph edges':>11} | "
          f"{'got nodes':>9} | {'got edges':>9} | {'hubs kept':>9}")
    for with_hubs, gn, ge, rn, re_, hi in rows:
        print(f"{str(with_hubs):>6} | {gn:>11} | {ge:>11} | {rn:>9} | {re_:>9} | {hi:>9}")

    differs = rows[0][3] != rows[1][3] or rows[0][4] != rows[1][4]
    print(f"\nVERDICT: {'SUPPORTED' if differs else 'NOT SUPPORTED'} -- "
          f"pruning hubs {'changes' if differs else 'does not change'} the retrieved "
          "subgraph.")


def main() -> None:
    print(__doc__.split("Hypotheses")[0].strip())
    graph, meta = build_graph(with_hubs=True)
    print(f"\nsynthetic graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{len(set(graph.node_texts))} distinct node texts, "
          f"{len(set(graph.edge_texts))} distinct relation types")

    h1_edge_prizes(graph, meta)
    h2_ties(graph, meta)
    h3_hubs()


if __name__ == "__main__":
    main()
