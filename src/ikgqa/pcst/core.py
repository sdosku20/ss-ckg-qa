"""
ikgqa.pcst.core
================

A step-by-step, instrumented re-implementation of G-Retriever's PCST subgraph
retrieval (He et al., NeurIPS 2024, Section 5.3).

The original implementation is one 90-line function:
    G-Retriever/src/dataset/utils/retrieval.py::retrieval_via_pcst (verified byte-identical to upstream main, 18 Aug 2026)
A verbatim copy sits beside it in `ikgqa/pcst/reference.py` (MIT, (c) 2024
Xiaoxin He). `tests/test_pcst.py::test_matches_official_implementation` asserts that
this module returns byte-identical output to that verbatim copy.

The only reason this file exists is that the original is a monolith: prizes,
the virtual-node transform, the solver call and the decoding all happen in one
scope, so you cannot inspect or unit-test any single step. Here each step is a
named function that returns plain numpy, and `retrieval_via_pcst_traced` hands
you a `RetrievalTrace` recording every intermediate quantity.

Numerics are kept identical to the original on purpose, including two quirks
that are documented inline (see NOTE-A and NOTE-B).

Dependencies: numpy and pandas. The PCST solver itself is ours -- see
`ikgqa.pcst.gw`, which implements Goemans-Williamson from the papers -- so
nothing in the retrieval path needs a compiled external solver. `pcst_fast` is
still an optional dependency, used only to cross-check our solver
(`tests/test_gw.py`) and to keep the byte-equivalence test against the published
code a controlled comparison (`tests/test_pcst.py`). torch / torch_geometric
are optional too, used only to mirror the original's output types when the input
is a PyG Data.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# NOTE: pcst_fast is imported lazily (see `_pcst_fast()` below), not here.
# Everything except the actual solver call (`solve_pcst`, and the sanity check
# that guards it) is pure numpy/pandas and works without pcst_fast installed at
# all. That matters in practice: pcst_fast ships no Windows build, official or
# conda (bioconda's pcst-fast package lists linux-64 and osx-64 only), so on a
# native Windows machine without WSL you may be stuck without a working solver
# for a while. This way you can still read, run and test every stage up to
# "hand the problem to the solver" in the meantime.

# ---------------------------------------------------------------------------
# Constants from the original implementation
# ---------------------------------------------------------------------------

#: `c` in the original code. Used for two things: to shrink each successive
#: edge-prize tier slightly (so tiers stay strictly ordered), and to shave the
#: edge cost just below the top edge prize.
C = 0.01

#: PCST solver settings hard-coded in the original.
ROOT = -1  # -1 => unrooted problem
NUM_CLUSTERS = 1  # force a single connected component
PRUNING = "gw"  # Goemans-Williamson pruning, as the published code passes

#: Pruning used by our own solver. Strong pruning (Johnson, Minkoff & Phillips,
#: SODA 2000) solves PCST exactly on the tree the growth stage produced, and
#: their result is that it is never worse than GW's rule. Measured here: on the
#: twelve random graphs in the equivalence test it reproduces `pcst_fast`'s `gw`
#: answer exactly, while our own reading of the GW pruning rule is weaker (see
#: `ikgqa.pcst.gw._prune_gw`). So this is both the faithful choice and the
#: better one, and it is what keeps the byte-equivalence test passing.
OWN_PRUNING = "strong"
VERBOSITY = 0

#: torch.nn.CosineSimilarity's default epsilon.
COS_EPS = 1e-8


# ---------------------------------------------------------------------------
# Small containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimpleGraph:
    """Minimal stand-in for torch_geometric.data.Data.

    Any object exposing `x`, `edge_index`, `edge_attr`, `num_nodes` and
    `num_edges` works with this module, so a real PyG `Data` can be passed
    straight in.
    """

    x: np.ndarray  # [num_nodes, d] node embeddings
    edge_index: np.ndarray  # [2, num_edges] (src row, dst row)
    edge_attr: np.ndarray  # [num_edges, d] edge embeddings
    num_nodes: int

    @property
    def num_edges(self) -> int:
        return int(np.asarray(self.edge_index).shape[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SimpleGraph(num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"dim={np.asarray(self.x).shape[-1]})"
        )


@dataclasses.dataclass
class PCSTInstance:
    """The graph that is actually handed to the PCST solver.

    It is *not* the input graph: every edge whose prize exceeds the edge cost
    has been replaced by a "virtual node" (paper, Section 5.3), because the
    solver only accepts prizes on nodes and non-negative costs on edges.
    """

    edges: np.ndarray  # [num_real_edges + 2*num_virtual, 2]
    prizes: np.ndarray  # [num_nodes + num_virtual]
    costs: np.ndarray  # [num_real_edges + 2*num_virtual]
    num_real_edges: int  # solver edge ids < this are real graph edges
    num_real_nodes: int  # solver node ids < this are real graph nodes
    edge_map: dict  # solver edge id -> original edge id
    virtual_node_map: dict  # solver node id -> original edge id it replaces
    cost_e: float  # the (possibly lowered) per-edge cost actually used

    @property
    def num_virtual_nodes(self) -> int:
        return len(self.virtual_node_map)


@dataclasses.dataclass
class RetrievalTrace:
    """Every intermediate quantity, for inspection and teaching."""

    node_similarity: np.ndarray
    node_prizes: np.ndarray
    edge_similarity: np.ndarray
    edge_prizes: np.ndarray
    cost_e_requested: float
    cost_e_used: float
    instance: Optional[PCSTInstance]
    solver_vertices: np.ndarray
    solver_edges: np.ndarray
    selected_nodes: np.ndarray
    selected_edges: np.ndarray
    short_circuited: bool = False

    # -- convenience reporting -------------------------------------------------
    def _padded(self, arr: np.ndarray, n: int) -> np.ndarray:
        """On the short-circuit path no prizes were computed, so report zeros."""
        return arr if arr.size == n else np.zeros(n, dtype=np.float32)

    def prize_table(self, textual_nodes: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        n = len(textual_nodes)
        self.node_similarity = self._padded(self.node_similarity, n)
        self.node_prizes = self._padded(self.node_prizes, n)
        df = pd.DataFrame(
            {
                "node_id": np.arange(len(self.node_prizes)),
                "text": list(textual_nodes["node_attr"]),
                "cos_sim": self.node_similarity,
                "prize": self.node_prizes,
                "selected": np.isin(np.arange(len(self.node_prizes)), self.selected_nodes),
            }
        )
        return df.sort_values("cos_sim", ascending=False).head(top)

    def edge_table(self, textual_edges: pd.DataFrame, top: int = 10) -> pd.DataFrame:
        m = len(textual_edges)
        self.edge_similarity = self._padded(self.edge_similarity, m)
        self.edge_prizes = self._padded(self.edge_prizes, m)
        df = pd.DataFrame(
            {
                "edge_id": np.arange(len(self.edge_prizes)),
                "src": list(textual_edges["src"]),
                "text": list(textual_edges["edge_attr"]),
                "dst": list(textual_edges["dst"]),
                "cos_sim": self.edge_similarity,
                "prize": self.edge_prizes,
                "cost": np.maximum(self.cost_e_used - self.edge_prizes, 0.0),
                "virtualised": self.edge_prizes > self.cost_e_used,
                "selected": np.isin(np.arange(len(self.edge_prizes)), self.selected_edges),
            }
        )
        return df.sort_values("cos_sim", ascending=False).head(top)


@dataclasses.dataclass
class RetrievalResult:
    subgraph: Any  # PyG Data if torch is available, else SimpleGraph
    desc: str  # the textualised subgraph fed to the LLM
    selected_nodes: np.ndarray  # ids into the ORIGINAL graph
    selected_edges: np.ndarray  # ids into the ORIGINAL graph
    trace: RetrievalTrace


# ---------------------------------------------------------------------------
# Step 0: environment guard
# ---------------------------------------------------------------------------

_SANITY_CHECKED = False


def _pcst_fast():
    """Import pcst_fast on first use, with an actionable error if it's missing.

    Deferred so that everything up to `solve_pcst` works in an environment
    that has numpy/pandas but no compiled solver, e.g. while you are still
    setting up WSL on Windows.
    """
    try:
        from pcst_fast import pcst_fast as _fn

        return _fn
    except ImportError as e:
        raise ImportError(
            "pcst_fast is not installed, so the solver step cannot run (everything "
            "before it, prizes and the virtual-node transform, works fine without it).\n"
            "  pip install pcst_fast\n"
            "pcst_fast ships no official Windows build (its own bioconda package lists "
            "linux-64 and osx-64 only). On native Windows this will try to compile from "
            "C++ source and needs a compiler that was never tested against MSVC. The "
            "reliable fix is WSL: `wsl --install`, then run this project inside the "
            "Ubuntu shell it gives you, where `pip install pcst_fast` just works."
        ) from e


def check_pcst_fast_sanity() -> None:
    """Verify that the installed `pcst_fast` binary actually works.

    Why this exists: the pcst_fast 1.0.10 manylinux wheel is built against the
    NumPy 1.x C ABI. Under NumPy >= 2 it still imports and still returns arrays
    of the *correct length*, but the values are garbage (every entry identical).
    Nothing raises, so a broken environment silently produces nonsense
    subgraphs. Pin `numpy<2` (G-Retriever's own env uses numpy 1.x) or build
    pcst_fast from source.
    """
    # Path 0-1-2-3, prize only on the two endpoints, cheap edges => take all.
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    prizes = np.array([10.0, 0.0, 0.0, 10.0])
    costs = np.array([1.0, 1.0, 1.0])
    vertices, chosen = _pcst_fast()(edges, prizes, costs, ROOT, NUM_CLUSTERS, PRUNING, VERBOSITY)
    ok = set(vertices.tolist()) == {0, 1, 2, 3} and set(chosen.tolist()) == {0, 1, 2}
    if not ok:
        raise RuntimeError(
            "pcst_fast is returning incorrect results in this environment "
            f"(got vertices={vertices.tolist()}, edges={chosen.tolist()}; "
            "expected [0,1,2,3] and [0,1,2]).\n"
            f"numpy version = {np.__version__}. The pcst_fast wheel needs numpy<2; "
            "run `pip install 'numpy<2'` or build pcst_fast from source."
        )


def _ensure_sane() -> None:
    global _SANITY_CHECKED
    if not _SANITY_CHECKED:
        check_pcst_fast_sanity()
        _SANITY_CHECKED = True


# ---------------------------------------------------------------------------
# Step 1: similarity
# ---------------------------------------------------------------------------


def _to_numpy(a: Any, dtype=np.float32) -> np.ndarray:
    """Accept numpy arrays, torch tensors or lists."""
    if hasattr(a, "detach"):  # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


def cosine_similarity(q_emb: Any, mat: Any) -> np.ndarray:
    """Cosine similarity between one query vector and each row of `mat`.

    Mirrors `torch.nn.CosineSimilarity(dim=-1)(q_emb, mat)`: float32
    throughout, product of norms clamped from below by eps.
    """
    q = _to_numpy(q_emb).reshape(-1)
    m = _to_numpy(mat)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    numer = m @ q
    denom = np.maximum(
        np.linalg.norm(m, axis=1) * np.linalg.norm(q), np.float32(COS_EPS)
    )
    return (numer / denom).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: node prizes  (paper Eq. 6)
# ---------------------------------------------------------------------------


def _topk_indices(sim: np.ndarray, k: int, tie_break: str) -> np.ndarray:
    """Indices of the k largest values, best first.

    Tie-breaking matters more than you would expect. When several nodes have
    *identical* similarity (very common: duplicate node text, or a query that
    shares no vocabulary with most nodes, so everything scores exactly 0), the
    choice of which tied node enters the top-k changes the retrieved subgraph.

    The original code uses `torch.topk`, whose tie order is an artefact of its
    quickselect and is not documented or index-ordered. So:

      * tie_break="auto"   -> use torch.topk if torch is installed, giving
                              bit-identical results to upstream G-Retriever;
                              otherwise fall back to "stable".
      * tie_break="stable" -> lowest index wins. Deterministic, torch-free, and
                              reproducible across versions, but it can disagree
                              with upstream on tied inputs.
    """
    if tie_break == "auto":
        try:
            import torch

            tie_break = "torch"
        except ImportError:  # pragma: no cover
            tie_break = "stable"

    if tie_break == "torch":
        import torch

        _, idx = torch.topk(torch.from_numpy(np.ascontiguousarray(sim)), k, largest=True)
        return idx.numpy()
    if tie_break == "stable":
        return np.argsort(-sim, kind="stable")[:k]
    raise ValueError(f"unknown tie_break {tie_break!r}")


def compute_node_prizes(q_emb: Any, x: Any, topk: int, tie_break: str = "auto") -> tuple:
    """prize(n) = k - i for the i-th most similar node (0-indexed), else 0.

    So the best node gets prize `topk`, the second `topk - 1`, ..., the k-th
    gets 1, and everybody else gets 0 (paper Eq. 6). Returns (similarity, prizes).

    Note what this throws away: only the *rank* survives, not the similarity
    value. A node that matches the question perfectly and one that matches it
    weakly get prizes 3 and 2 if they happen to be ranked first and second.
    """
    sim = cosine_similarity(q_emb, x)
    num_nodes = sim.shape[0]
    if topk <= 0:
        return sim, np.zeros(num_nodes, dtype=np.float32)

    k = min(topk, num_nodes)
    order = _topk_indices(sim, k, tie_break)
    prizes = np.zeros(num_nodes, dtype=np.float32)
    prizes[order] = np.arange(k, 0, -1, dtype=np.float32)
    return sim, prizes


# ---------------------------------------------------------------------------
# Step 3: edge prizes  (the part the paper glosses over)
# ---------------------------------------------------------------------------


def compute_edge_prizes(q_emb: Any, edge_attr: Any, topk_e: int, c: float = C) -> tuple:
    """Edge prizes, replicating the original code exactly.

    The paper says "edge prizes are assigned similarly" to nodes, but the code
    does something more careful, because in a knowledge graph the *same*
    relation string ("people.person.parents") appears on hundreds of edges and
    therefore has an identical embedding and an identical similarity.

    The procedure:
      0. `topk_e` is first clamped to the number of *distinct* similarity
         values, so on a graph with only 2 distinct edge similarities a request
         for topk_e=5 becomes topk_e=2 and the top tier is worth 2, not 5.
         Easy to miss, and it changes the prize scale (and therefore how
         `cost_e` behaves) from graph to graph;
      1. rank the distinct similarity values, keep the top `topk_e` tiers;
      2. zero out every edge below the last surviving tier;
      3. tier k (0-indexed, best first) would nominally be worth `topk_e - k`,
         but that budget is *split* across all edges tied in that tier:
         value = (topk_e - k) / n_tied;
      4. clamp so each tier is worth strictly less than the previous one
         (`last * (1 - c)`), keeping the ranking intact after splitting.

    Consequence worth knowing: a relation shared by 100 edges is individually
    worth almost nothing, so PCST will not pay to reach it. Rare relations
    dominate. That is a real retrieval bias, not an implementation detail.

    NOTE-A (faithful quirk): step 3 selects tier members with
    `e_prizes == tier_value` on the array it is simultaneously overwriting. If
    a tier value is exactly 0.0 it also matches every edge zeroed in step 2,
    so those edges get a prize. This only triggers when a top-k similarity is
    exactly zero. Reproduced here for equivalence with the original.
    """
    sim = cosine_similarity(q_emb, edge_attr)
    num_edges = sim.shape[0]
    if topk_e <= 0:
        return sim, np.zeros(num_edges, dtype=np.float32)

    prizes = sim.copy()
    unique_vals = np.unique(prizes)  # ascending
    k = min(topk_e, unique_vals.size)
    tier_values = unique_vals[::-1][:k]  # descending == torch.topk

    prizes[prizes < tier_values[-1]] = 0.0

    last = float(k)
    for i in range(k):
        members = prizes == tier_values[i]
        n_tied = int(members.sum())
        value = min((k - i) / n_tied, last)
        prizes[members] = value
        last = value * (1 - c)

    return sim, prizes.astype(np.float32)


def adjust_edge_cost(edge_prizes: np.ndarray, cost_e: float, c: float = C) -> float:
    """Lower the per-edge cost so that at least one edge is guaranteed a prize
    strictly above its cost.

        cost_e <- min(cost_e, max_prize * (1 - c/2))

    Without this, a caller-supplied `cost_e` of 0.5 could exceed every edge
    prize, no virtual node would ever be created, and edge semantics would be
    ignored entirely. With it, the top edge always becomes a virtual node with
    a positive prize, so the retrieved subgraph always contains at least one
    query-relevant edge (assuming the graph has one).
    """
    if edge_prizes.size == 0:
        return float(cost_e)
    return float(min(cost_e, float(edge_prizes.max()) * (1 - c / 2)))


# ---------------------------------------------------------------------------
# Step 4: the virtual-node transform  (paper Section 5.3)
# ---------------------------------------------------------------------------


def build_pcst_instance(
    edge_index: Any,
    node_prizes: np.ndarray,
    edge_prizes: np.ndarray,
    cost_e: float,
    num_nodes: int,
) -> PCSTInstance:
    """Turn "prizes on edges" into a problem the PCST solver understands.

    The classic PCST problem allows prizes on nodes only, and requires
    non-negative edge costs. G-Retriever wants prizes on edges too, and handles
    it in two cases:

      * prize <= cost: keep the edge, just make it cheaper.
            new_cost = cost_e - prize        (>= 0, so still legal)

      * prize >  cost: `cost_e - prize` would be negative, which the solver
        rejects. So the edge is *deleted* and replaced by a virtual node
        carrying the surplus:
            src --(cost 0)-- v_e --(cost 0)-- dst,   prize(v_e) = prize - cost_e
        Buying the virtual node is now exactly as attractive as buying the
        original edge would have been, and it is still only reachable by
        passing through both endpoints.

    Virtual nodes are appended after the real ones, so solver node id
    `num_nodes + j` is virtual. Two bookkeeping dicts let us undo all of this
    afterwards.
    """
    ei = np.asarray(_to_numpy(edge_index, dtype=np.int64))
    real_edges: list = []
    real_costs: list = []
    virtual_prizes: list = []
    virtual_edges: list = []
    virtual_costs: list = []
    edge_map: dict = {}
    virtual_node_map: dict = {}

    for i, (src, dst) in enumerate(ei.T):
        prize = float(edge_prizes[i])
        if prize <= cost_e:
            edge_map[len(real_edges)] = i
            real_edges.append((int(src), int(dst)))
            real_costs.append(cost_e - prize)
        else:
            virtual_id = num_nodes + len(virtual_prizes)
            virtual_node_map[virtual_id] = i
            virtual_edges.append((int(src), virtual_id))
            virtual_edges.append((virtual_id, int(dst)))
            virtual_costs.extend([0.0, 0.0])
            virtual_prizes.append(prize - cost_e)

    # float32 on purpose: the original concatenates a float32 torch tensor with
    # float32 scalars, so the prize vector it hands the solver is float32. Using
    # float64 here would differ in the last bits and could, on an exact tie,
    # change which subgraph is returned.
    prizes = np.concatenate(
        [np.asarray(node_prizes, dtype=np.float32), np.asarray(virtual_prizes, dtype=np.float32)]
    )
    all_edges = np.asarray(real_edges + virtual_edges, dtype=np.int64).reshape(-1, 2)
    all_costs = np.asarray(real_costs + virtual_costs, dtype=np.float64)

    return PCSTInstance(
        edges=all_edges,
        prizes=prizes,
        costs=all_costs,
        num_real_edges=len(real_edges),
        num_real_nodes=int(num_nodes),
        edge_map=edge_map,
        virtual_node_map=virtual_node_map,
        cost_e=float(cost_e),
    )


# ---------------------------------------------------------------------------
# Step 5: solve
# ---------------------------------------------------------------------------


SOLVERS = ("own", "pcst_fast")
DEFAULT_SOLVER = "own"


def solve_pcst(
    instance: PCSTInstance,
    root: int = ROOT,
    num_clusters: int = NUM_CLUSTERS,
    pruning: str = PRUNING,
    verbosity: int = VERBOSITY,
    solver: str = DEFAULT_SOLVER,
) -> tuple:
    """Solve the PCST instance.

    Maximising (prizes collected - edge costs) is the same as minimising
    (prizes forgone + edge costs), which is the standard PCST objective.
    `root=-1` means unrooted, `num_clusters=1` forces one connected component,
    `pruning='gw'` applies the Goemans-Williamson pruning pass.

    Args:
        solver: "own" runs `ikgqa.pcst.gw`, the primal-dual algorithm written out
            from the papers, and is the default: the retrieval pipeline depends
            on no external solver. "pcst_fast" calls the C++ library, and exists
            so the two can be cross-checked -- see `tests/test_gw.py`, which
            requires our objective never to be worse on random instances.

    Returns (vertices, edges) as ids into the *instance*, not the input graph.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if instance.edges.shape[0] == 0:
        # With no edges only isolated prized nodes exist, and a single connected
        # component can hold at most one of them.
        best = int(np.argmax(instance.prizes)) if instance.prizes.size else 0
        keep = np.array([best], dtype=np.int64) if instance.prizes.size else np.array([], dtype=np.int64)
        return keep, np.array([], dtype=np.int64)
    if solver == "own":
        from ikgqa.pcst import gw

        return gw.solve(
            instance.edges, instance.prizes, instance.costs, root, num_clusters, pruning, verbosity
        )
    _ensure_sane()
    return _pcst_fast()(
        instance.edges, instance.prizes, instance.costs, root, num_clusters, pruning, verbosity
    )


# ---------------------------------------------------------------------------
# Step 6: decode back to the original graph
# ---------------------------------------------------------------------------


def decode_solution(
    instance: PCSTInstance,
    vertices: np.ndarray,
    solver_edges: np.ndarray,
    edge_index: Any,
) -> tuple:
    """Map the solver's answer back onto original node and edge ids.

    Three things happen:
      * solver vertices below `num_real_nodes` are real nodes;
      * solver edges below `num_real_edges` are real edges;
      * every *selected virtual node* stands for an original edge, so it is
        converted back into an edge id (its two zero-cost half-edges are
        ignored).

    Finally the node set is closed under the selected edges: any endpoint of a
    selected edge is added even if the solver never listed it. This is what
    pulls in zero-prize "bridge" nodes and is the reason the returned subgraph
    is connected and contextual rather than a bag of top-k hits.
    """
    vertices = np.asarray(vertices)
    solver_edges = np.asarray(solver_edges)

    selected_nodes = vertices[vertices < instance.num_real_nodes]
    selected_edges = [instance.edge_map[int(e)] for e in solver_edges if int(e) < instance.num_real_edges]

    virtual_vertices = vertices[vertices >= instance.num_real_nodes]
    if virtual_vertices.size > 0:
        recovered = [instance.virtual_node_map[int(v)] for v in virtual_vertices]
        selected_edges = np.asarray(selected_edges + recovered, dtype=np.int64)
    else:
        selected_edges = np.asarray(selected_edges, dtype=np.int64)

    ei = np.asarray(_to_numpy(edge_index, dtype=np.int64))
    endpoints = ei[:, selected_edges] if selected_edges.size else np.zeros((2, 0), dtype=np.int64)
    selected_nodes = np.unique(
        np.concatenate([selected_nodes.astype(np.int64), endpoints[0], endpoints[1]])
    )
    return selected_nodes, selected_edges


# ---------------------------------------------------------------------------
# Step 7: rebuild the subgraph and its textual description
# ---------------------------------------------------------------------------


def _index_rows(arr: Any, idx: np.ndarray) -> Any:
    """Row-index a numpy array or torch tensor, preserving its type."""
    if hasattr(arr, "detach"):  # torch tensor
        import torch

        return arr[torch.as_tensor(np.asarray(idx), dtype=torch.long)]
    return np.asarray(arr)[idx]


def _make_edge_index(src: Sequence[int], dst: Sequence[int], like: Any) -> Any:
    if hasattr(like, "detach"):  # torch tensor
        import torch

        return torch.LongTensor([list(src), list(dst)])
    return np.asarray([list(src), list(dst)], dtype=np.int64)


def _make_graph(x, edge_index, edge_attr, num_nodes):
    """Return a PyG Data if torch_geometric is importable, else a SimpleGraph."""
    if hasattr(x, "detach"):
        try:
            from torch_geometric.data.data import Data

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        except ImportError:  # pragma: no cover
            pass
    return SimpleGraph(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def build_description(
    textual_nodes: pd.DataFrame,
    textual_edges: pd.DataFrame,
    selected_nodes: np.ndarray,
    selected_edges: np.ndarray,
) -> str:
    """The exact string the LLM sees: two CSV blocks separated by a blank line.

    Careful: the ids inside this text are the ORIGINAL node ids, while the
    returned tensor subgraph is re-indexed 0..n-1. The LLM and the GNN
    therefore see different numbering. Harmless for G-Retriever (the LLM only
    reads text) but a trap if you try to align the two.
    """
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    return n.to_csv(index=False) + "\n" + e.to_csv(index=False, columns=["src", "edge_attr", "dst"])


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


def retrieval_via_pcst_traced(
    graph: Any,
    q_emb: Any,
    textual_nodes: pd.DataFrame,
    textual_edges: pd.DataFrame,
    topk: int = 3,
    topk_e: int = 3,
    cost_e: float = 0.5,
    tie_break: str = "auto",
    solver: str = DEFAULT_SOLVER,
) -> RetrievalResult:
    """Full G-Retriever PCST retrieval, with a trace of every intermediate step.

    `solver="own"` (the default) uses our own primal-dual implementation;
    `solver="pcst_fast"` calls the external library, which is what the
    byte-equivalence test against the published code holds fixed.
    """
    num_nodes = int(graph.num_nodes)

    # Short circuit, exactly as in the original: with no text there is nothing
    # to score, so the whole graph is returned untouched.
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + "\n" + textual_edges.to_csv(
            index=False, columns=["src", "edge_attr", "dst"]
        )
        sub = _make_graph(graph.x, graph.edge_index, graph.edge_attr, num_nodes)
        trace = RetrievalTrace(
            node_similarity=np.zeros(0, dtype=np.float32),
            node_prizes=np.zeros(0, dtype=np.float32),
            edge_similarity=np.zeros(0, dtype=np.float32),
            edge_prizes=np.zeros(0, dtype=np.float32),
            cost_e_requested=float(cost_e),
            cost_e_used=float(cost_e),
            instance=None,
            solver_vertices=np.arange(num_nodes),
            solver_edges=np.arange(int(graph.num_edges)),
            selected_nodes=np.arange(num_nodes),
            selected_edges=np.arange(int(graph.num_edges)),
            short_circuited=True,
        )
        return RetrievalResult(sub, desc, trace.selected_nodes, trace.selected_edges, trace)

    n_sim, n_prizes = compute_node_prizes(q_emb, graph.x, topk, tie_break=tie_break)
    e_sim, e_prizes = compute_edge_prizes(q_emb, graph.edge_attr, topk_e)
    cost_used = adjust_edge_cost(e_prizes, cost_e) if topk_e > 0 else float(cost_e)

    instance = build_pcst_instance(graph.edge_index, n_prizes, e_prizes, cost_used, num_nodes)
    vertices, solver_edges = solve_pcst(
        instance, pruning=OWN_PRUNING if solver == "own" else PRUNING, solver=solver
    )
    selected_nodes, selected_edges = decode_solution(instance, vertices, solver_edges, graph.edge_index)

    desc = build_description(textual_nodes, textual_edges, selected_nodes, selected_edges)

    # Re-index the subgraph so node ids run 0..len(selected_nodes)-1.
    ei = np.asarray(_to_numpy(graph.edge_index, dtype=np.int64))
    sub_ei = ei[:, selected_edges] if selected_edges.size else np.zeros((2, 0), dtype=np.int64)
    remap = {int(n): i for i, n in enumerate(selected_nodes.tolist())}
    new_edge_index = _make_edge_index(
        [remap[int(i)] for i in sub_ei[0]], [remap[int(i)] for i in sub_ei[1]], graph.edge_index
    )
    sub = _make_graph(
        _index_rows(graph.x, selected_nodes),
        new_edge_index,
        _index_rows(graph.edge_attr, selected_edges),
        len(selected_nodes),
    )

    trace = RetrievalTrace(
        node_similarity=n_sim,
        node_prizes=n_prizes,
        edge_similarity=e_sim,
        edge_prizes=e_prizes,
        cost_e_requested=float(cost_e),
        cost_e_used=float(cost_used),
        instance=instance,
        solver_vertices=np.asarray(vertices),
        solver_edges=np.asarray(solver_edges),
        selected_nodes=selected_nodes,
        selected_edges=selected_edges,
    )
    return RetrievalResult(sub, desc, selected_nodes, selected_edges, trace)


def retrieval_via_pcst(
    graph: Any,
    q_emb: Any,
    textual_nodes: pd.DataFrame,
    textual_edges: pd.DataFrame,
    topk: int = 3,
    topk_e: int = 3,
    cost_e: float = 0.5,
    tie_break: str = "auto",
    solver: str = DEFAULT_SOLVER,
) -> tuple:
    """Drop-in replacement for the original function: returns (subgraph, desc)."""
    r = retrieval_via_pcst_traced(
        graph, q_emb, textual_nodes, textual_edges, topk, topk_e, cost_e, tie_break, solver
    )
    return r.subgraph, r.desc


# ---------------------------------------------------------------------------
# Reporting helper used by demo.py
# ---------------------------------------------------------------------------


def summarise(result: RetrievalResult, textual_nodes: pd.DataFrame, textual_edges: pd.DataFrame) -> str:
    t = result.trace
    lines = []
    lines.append(f"edge cost: requested {t.cost_e_requested:.3f} -> used {t.cost_e_used:.4f}")
    if t.instance is not None:
        lines.append(
            f"solver instance: {t.instance.num_real_nodes} real nodes "
            f"+ {t.instance.num_virtual_nodes} virtual, "
            f"{t.instance.num_real_edges} real edges "
            f"+ {2 * t.instance.num_virtual_nodes} zero-cost half-edges"
        )
    lines.append(
        f"retrieved: {len(result.selected_nodes)}/{len(textual_nodes)} nodes, "
        f"{len(result.selected_edges)}/{len(textual_edges)} edges"
    )
    lines.append("")
    lines.append("triples in the retrieved subgraph:")
    for eid in result.selected_edges.tolist():
        row = textual_edges.iloc[eid]
        s = textual_nodes.iloc[int(row["src"])]["node_attr"]
        d = textual_nodes.iloc[int(row["dst"])]["node_attr"]
        lines.append(f"  ({s}) -[{row['edge_attr']}]-> ({d})")
    return "\n".join(lines)
