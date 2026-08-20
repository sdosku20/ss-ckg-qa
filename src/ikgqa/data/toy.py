"""
ikgqa.data.toy
==============

Small hand-built graphs to experiment on, plus a deterministic toy text
encoder so the playground needs no model download and gives the same answer
every run.

The real pipeline encodes node/edge text with SentenceBERT. Here `encode()`
builds an L2-normalised bag-of-words vector over the vocabulary of the graph,
so cosine similarity is literally "how many words does this share with the
question". That makes every prize hand-checkable, which is the point.

Swapping in the real thing is one line:

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    x = m.encode(node_texts, normalize_embeddings=True)
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ikgqa.pcst.core import SimpleGraph

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


class ToyEncoder:
    """Bag-of-words encoder over a fixed vocabulary. Deterministic, tiny."""

    def __init__(self, corpus: Iterable[str]):
        vocab = {}
        for text in corpus:
            for tok in tokenize(text):
                vocab.setdefault(tok, len(vocab))
        self.vocab = vocab

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), max(len(self.vocab), 1)), dtype=np.float32)
        for r, text in enumerate(texts):
            for tok in tokenize(text):
                if tok in self.vocab:
                    out[r, self.vocab[tok]] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-8)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class ToyKG:
    """A tiny textual graph: node texts, triples, embeddings and DataFrames."""

    def __init__(self, node_texts: Sequence[str], triples: Sequence[tuple], name: str = "toy"):
        self.name = name
        self.node_texts = list(node_texts)
        self.triples = [(int(s), str(r), int(d)) for s, r, d in triples]
        self.edge_texts = [r for _, r, _ in self.triples]

        self.encoder = ToyEncoder(self.node_texts + self.edge_texts)
        self.x = self.encoder.encode(self.node_texts)
        self.edge_attr = (
            self.encoder.encode(self.edge_texts)
            if self.edge_texts
            else np.zeros((0, max(len(self.encoder.vocab), 1)), dtype=np.float32)
        )
        self.edge_index = np.array(
            [[s for s, _, _ in self.triples], [d for _, _, d in self.triples]], dtype=np.int64
        ).reshape(2, -1)

        self.nodes_df = pd.DataFrame(
            {"node_id": np.arange(len(self.node_texts)), "node_attr": self.node_texts}
        )
        self.edges_df = pd.DataFrame(
            {
                "src": [s for s, _, _ in self.triples],
                "edge_attr": self.edge_texts,
                "dst": [d for _, _, d in self.triples],
            }
        )

    @property
    def graph(self) -> SimpleGraph:
        return SimpleGraph(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=len(self.node_texts),
        )

    def q(self, question: str) -> np.ndarray:
        """Embed a question in the same space as the graph."""
        return self.encoder.encode_one(question)

    def pretty(self) -> str:
        lines = [f"# {self.name}: {len(self.node_texts)} nodes, {len(self.triples)} edges"]
        for i, t in enumerate(self.node_texts):
            lines.append(f"  node {i:>2}: {t}")
        for i, (s, r, d) in enumerate(self.triples):
            lines.append(f"  edge {i:>2}: ({self.node_texts[s]}) -[{r}]-> ({self.node_texts[d]})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. A chain, to show that PCST pays for bridges
# ---------------------------------------------------------------------------


def chain_kg() -> ToyKG:
    """A path graph whose two ends both match the query.

    Query "creatinine tacrolimus" matches node 0 and node 6 strongly and
    nothing in between. Top-k retrieval would return two disconnected hits.
    PCST has to buy the boring middle nodes to connect them, or drop one end.
    Raising `cost_e` is what flips that decision.
    """
    node_texts = [
        "creatinine",
        "laboratory result",
        "specimen",
        "encounter",
        "medication statement",
        "dose",
        "tacrolimus",
        "unrelated allergy record",
    ]
    triples = [
        (0, "measured in", 1),
        (1, "derived from", 2),
        (2, "collected during", 3),
        (3, "includes", 4),
        (4, "has", 5),
        (5, "of drug", 6),
        (7, "documented during", 3),
    ]
    return ToyKG(node_texts, triples, name="chain")


# ---------------------------------------------------------------------------
# 2. A hub, to show what happens with a very common relation
# ---------------------------------------------------------------------------


def hub_kg() -> ToyKG:
    """One central patient node with many identical `has lab result` edges.

    Because all those edges share one relation string they share one embedding,
    so they tie, so the tier budget is split ten ways and each edge is worth
    almost nothing. Meanwhile the single rare `has kidney transplant` edge
    keeps its whole tier. This is the tie-splitting rule from
    `compute_edge_prizes` doing its job.
    """
    node_texts = ["patient"] + [f"lab value {i}" for i in range(10)] + [
        "kidney transplant",
        "hepatitis c virus",
    ]
    triples = [(0, "has lab result", i + 1) for i in range(10)]
    triples += [(0, "has kidney transplant", 11), (0, "has viral infection", 12)]
    return ToyKG(node_texts, triples, name="hub")


# ---------------------------------------------------------------------------
# 3. A small clinical KG in the shape of an SPHN-style transplant cohort
# ---------------------------------------------------------------------------


def clinical_kg() -> ToyKG:
    """A 20-node stand-in for a clinical knowledge graph.

    Two patients, each with a diagnosis, a lab result with a LOINC-style code,
    a medication with an ATC-style code, and an encounter. Deliberately
    includes near-duplicate relation names so tie-splitting is exercised, and
    a second patient's subgraph so retrieval has a wrong answer available.
    """
    node_texts = [
        "patient 001",                        # 0
        "patient 002",                        # 1
        "kidney transplant 2019",             # 2
        "hepatitis c virus infection",        # 3
        "creatinine serum 2 1 mg per dl",     # 4
        "creatinine serum 0 9 mg per dl",     # 5
        "tacrolimus 5 mg oral",               # 6
        "mycophenolate 500 mg oral",          # 7
        "loinc 2160 0 creatinine",            # 8
        "loinc 1975 2 bilirubin",             # 9
        "atc l04ad02 tacrolimus",             # 10
        "atc l04aa06 mycophenolic acid",      # 11
        "inpatient encounter 2020 03",        # 12
        "outpatient encounter 2021 07",       # 13
        "icd10 n18 5 chronic kidney disease stage 5",  # 14
        "icd10 b18 2 chronic viral hepatitis c",       # 15
        "bilirubin total 0 4 mg per dl",      # 16
        "graft rejection episode",            # 17
        "sofosbuvir velpatasvir 12 weeks",    # 18
        "sustained virologic response",       # 19
    ]
    triples = [
        (0, "has diagnosis", 2),
        (0, "has diagnosis", 3),
        (0, "has lab result", 4),
        (0, "has medication statement", 6),
        (0, "has encounter", 12),
        (4, "has code", 8),
        (6, "has code", 10),
        (2, "coded as", 14),
        (3, "coded as", 15),
        (0, "has complication", 17),
        (3, "treated with", 18),
        (18, "resulted in", 19),
        (1, "has diagnosis", 3),
        (1, "has lab result", 5),
        (1, "has lab result", 16),
        (1, "has medication statement", 7),
        (1, "has encounter", 13),
        (5, "has code", 8),
        (16, "has code", 9),
        (7, "has code", 11),
    ]
    return ToyKG(node_texts, triples, name="clinical")


def clinical_gold_set() -> dict:
    """Hand-written gold triples for the clinical graph: question -> edge ids.

    A miniature version of a human-validated evaluation set. Each entry lists
    the triples a retriever must return for the question to be answerable at
    all, which lets you score retrieval on its own without an LLM in the loop.
    Edge ids refer to `clinical_kg().triples`.
    """
    return {
        "what lab result and code does patient 001 have for creatinine": {2, 5},
        "which complication did patient 001 have": {9},
        "which medication statement was given to patient 001 and what is its code": {3, 6},
        "how was the hepatitis c virus infection treated and what resulted": {10, 11},
        "which diagnosis of patient 001 is coded as icd10 n18 5": {0, 7},
    }


# ---------------------------------------------------------------------------
# 4. Degenerate graphs, for edge-case tests
# ---------------------------------------------------------------------------


def single_node_kg() -> ToyKG:
    return ToyKG(["lonely creatinine node"], [], name="single-node")


def two_components_kg() -> ToyKG:
    """Two disconnected triangles. `num_clusters=1` means only one can win."""
    node_texts = [
        "creatinine",
        "serum",
        "milligram per decilitre",
        "tacrolimus",
        "trough level",
        "nanogram per millilitre",
    ]
    triples = [
        (0, "measured in", 1),
        (1, "unit", 2),
        (2, "reported for", 0),
        (3, "measured as", 4),
        (4, "unit", 5),
        (5, "reported for", 3),
    ]
    return ToyKG(node_texts, triples, name="two-components")


def random_kg(seed: int = 0, num_nodes: int = 40, num_edges: int = 90, dim: int = 16,
              num_relations: int = 6) -> ToyKG:
    """A random graph with a small relation vocabulary, so edge ties occur.

    Used by the equivalence test against the official implementation.
    """
    rng = np.random.default_rng(seed)
    node_texts = [f"node {i} thing {rng.integers(0, 5)}" for i in range(num_nodes)]
    rels = [f"relation {r}" for r in range(num_relations)]
    triples = []
    # spanning path first, so the graph is connected
    for i in range(num_nodes - 1):
        triples.append((i, rels[int(rng.integers(0, num_relations))], i + 1))
    while len(triples) < num_edges:
        s, d = rng.integers(0, num_nodes, size=2)
        if s == d:
            continue
        triples.append((int(s), rels[int(rng.integers(0, num_relations))], int(d)))
    kg = ToyKG(node_texts, triples, name=f"random-{seed}")
    # replace bag-of-words vectors with random ones for a harder test
    kg.x = rng.standard_normal((num_nodes, dim)).astype(np.float32)
    rel_vecs = rng.standard_normal((num_relations, dim)).astype(np.float32)
    kg.edge_attr = np.stack([rel_vecs[rels.index(r)] for _, r, _ in kg.triples]).astype(np.float32)
    kg._rng = rng
    kg._dim = dim
    return kg


ALL_GRAPHS = {
    "chain": chain_kg,
    "hub": hub_kg,
    "clinical": clinical_kg,
    "single-node": single_node_kg,
    "two-components": two_components_kg,
}
