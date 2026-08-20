"""
ikgqa.graph
===========

One container for "a textual graph plus its embeddings", used by every
retriever, metric and experiment in this package.

Why a dedicated type instead of passing four loose arrays around: the shapes
have to agree (one embedding row per node, one per edge, edge endpoints inside
range) and a mismatch produces a *plausible but wrong* subgraph rather than a
crash. Validating once at construction turns a silent research bug into an
immediate error message.

Two representations are kept deliberately:

  * node_emb / edge_emb -- float32 matrices, what similarity is computed on
  * nodes / edges       -- pandas DataFrames of text, what the LLM reads

They must stay row-aligned: node i is nodes.iloc[i] and node_emb[i]. The
DataFrame column names (node_id, node_attr, src, edge_attr, dst) match
G-Retriever's, so ikgqa.pcst.core.build_description produces byte-identical
prompts to upstream.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ikgqa.pcst.core import SimpleGraph

NODE_COLUMNS = ("node_id", "node_attr")
EDGE_COLUMNS = ("src", "edge_attr", "dst")


@dataclasses.dataclass(frozen=True)
class TextualGraph:
    """A graph whose nodes and edges carry text, with embeddings attached.

    Args:
        nodes: DataFrame with columns node_id, node_attr. Row order is the node
            id order; row i is node i.
        edges: DataFrame with columns src, edge_attr, dst, where src/dst are
            node ids (ints).
        node_emb: [num_nodes, d] float32.
        edge_emb: [num_edges, d] float32.
        name: free-text label used in reports.
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_emb: np.ndarray
    edge_emb: np.ndarray
    name: str = "graph"

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        missing_n = [c for c in NODE_COLUMNS if c not in self.nodes.columns]
        missing_e = [c for c in EDGE_COLUMNS if c not in self.edges.columns]
        if missing_n:
            raise ValueError(f"nodes DataFrame is missing columns {missing_n}")
        if missing_e:
            raise ValueError(f"edges DataFrame is missing columns {missing_e}")

        object.__setattr__(self, "node_emb", np.ascontiguousarray(self.node_emb, dtype=np.float32))
        object.__setattr__(self, "edge_emb", np.ascontiguousarray(self.edge_emb, dtype=np.float32))

        if self.node_emb.ndim != 2:
            raise ValueError(f"node_emb must be 2-D, got shape {self.node_emb.shape}")
        if self.node_emb.shape[0] != len(self.nodes):
            raise ValueError(
                f"{len(self.nodes)} nodes but {self.node_emb.shape[0]} node embeddings"
            )
        if len(self.edges) and self.edge_emb.shape[0] != len(self.edges):
            raise ValueError(
                f"{len(self.edges)} edges but {self.edge_emb.shape[0]} edge embeddings"
            )
        if len(self.edges) and self.edge_emb.shape[-1] != self.node_emb.shape[-1]:
            raise ValueError(
                "node and edge embeddings must share a dimension "
                f"({self.node_emb.shape[-1]} vs {self.edge_emb.shape[-1]}); "
                "cosine similarity against one query vector requires it"
            )

        if len(self.edges):
            ends = np.concatenate(
                [
                    self.edges["src"].to_numpy(dtype=np.int64),
                    self.edges["dst"].to_numpy(dtype=np.int64),
                ]
            )
            if ends.min() < 0 or ends.max() >= len(self.nodes):
                raise ValueError(
                    f"edge endpoints must be node ids in [0, {len(self.nodes)}), "
                    f"got range [{ends.min()}, {ends.max()}]"
                )

    @classmethod
    def from_texts(
        cls,
        node_texts: Sequence[str],
        triples: Sequence[tuple],
        encoder: Any,
        name: str = "graph",
    ) -> "TextualGraph":
        """Build from raw text, embedding it with encoder.

        Args:
            node_texts: one string per node, in node id order.
            triples: (src_id, relation_text, dst_id) tuples.
            encoder: anything with encode(list[str]) -> [n, d]; see
                ikgqa.encoders.
        """
        texts = [str(t) for t in node_texts]
        rows = [(int(s), str(r), int(d)) for s, r, d in triples]
        edge_texts = [r for _, r, _ in rows]

        node_emb = encoder.encode(texts)
        dim = int(np.asarray(node_emb).shape[-1])
        edge_emb = encoder.encode(edge_texts) if edge_texts else np.zeros((0, dim), dtype=np.float32)

        return cls(
            nodes=pd.DataFrame({"node_id": np.arange(len(texts)), "node_attr": texts}),
            edges=pd.DataFrame(
                {
                    "src": [s for s, _, _ in rows],
                    "edge_attr": edge_texts,
                    "dst": [d for _, _, d in rows],
                }
            ),
            node_emb=node_emb,
            edge_emb=edge_emb,
            name=name,
        )

    # -- views -------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def dim(self) -> int:
        return int(self.node_emb.shape[-1])

    @property
    def edge_index(self) -> np.ndarray:
        """[2, num_edges] int64 array of (src row, dst row)."""
        if not len(self.edges):
            return np.zeros((2, 0), dtype=np.int64)
        return np.stack(
            [
                self.edges["src"].to_numpy(dtype=np.int64),
                self.edges["dst"].to_numpy(dtype=np.int64),
            ]
        )

    @property
    def node_texts(self) -> list:
        """What a reader sees: the full display text, values and dates included.

        This is what goes into a prompt and what prompt_chars measures. It is
        *not* what similarity is computed on when the two differ -- see
        embed_texts.
        """
        return self.nodes["node_attr"].astype(str).tolist()

    @property
    def embed_texts(self) -> list:
        """The text ``node_emb`` was computed from.

        For a graph with one text per node this is the same as node_texts. The
        clinical loader keeps them apart: display text carries the measured
        value and timestamp so an LLM can answer from it, while embed text
        carries only the semantic core so that repeated measurements share one
        embedding.

        Any retriever that scores text rather than using node_emb directly must
        use *this*, or it is ranking against a different graph than the one the
        embeddings describe. That is not a subtlety: scoring KAPING on display
        text while scoring PCST on embed text made a terminology fix invisible
        to the baseline and produced a 1.000-versus-0.000 result that was purely
        an artefact of the mismatch.
        """
        if "embed_attr" in self.nodes.columns:
            return self.nodes["embed_attr"].astype(str).tolist()
        return self.node_texts

    @property
    def edge_texts(self) -> list:
        return self.edges["edge_attr"].astype(str).tolist()

    def as_simple_graph(self) -> SimpleGraph:
        """The shape ikgqa.pcst.core and the PCST solver expect."""
        return SimpleGraph(
            x=self.node_emb,
            edge_index=self.edge_index,
            edge_attr=self.edge_emb,
            num_nodes=self.num_nodes,
        )

    def triple_texts(self) -> list:
        """"head relation tail" per edge -- what KAPING scores against.

        KAPING ranks whole triples, not relations, so giving it only edge_attr
        would handicap the baseline unfairly.

        Built from embed_texts, not node_texts, so the baseline scores against
        the same text the embeddings were computed from. See embed_texts.
        """
        texts = self.embed_texts
        return [
            f"{texts[int(s)]} {r} {texts[int(d)]}"
            for s, r, d in zip(self.edges["src"], self.edges["edge_attr"], self.edges["dst"])
        ]

    def pretty(self, limit: Optional[int] = None) -> str:
        texts = self.node_texts
        lines = [f"# {self.name}: {self.num_nodes} nodes, {self.num_edges} edges, dim={self.dim}"]
        for i, t in enumerate(texts if limit is None else texts[:limit]):
            lines.append(f"  node {i:>3}: {t}")
        rows = list(zip(self.edges["src"], self.edges["edge_attr"], self.edges["dst"]))
        for i, (s, r, d) in enumerate(rows if limit is None else rows[:limit]):
            lines.append(f"  edge {i:>3}: ({texts[int(s)]}) -[{r}]-> ({texts[int(d)]})")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TextualGraph(name={self.name!r}, num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, dim={self.dim})"
        )
