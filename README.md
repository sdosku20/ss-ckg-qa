# Intent-Aware Subgraph Selection for Clinical KGQA

Code for the MSc thesis of the same name (Serxhio Dosku, University of Basel, 2026).

Given a clinical knowledge graph and a natural-language question, which part of the
graph should an LLM see? This repository implements G-Retriever's Prize-Collecting
Steiner Tree (PCST) selection, the baselines it is compared against, and the
evaluation that decides between them.

---

## Quick start

```bash
pip install -e ".[dev]"      # add ",oracle" for the upstream-equivalence test
pytest                       # 122 tests, ~4 seconds
ikgqa-demo                   # walk through PCST one stage at a time
ikgqa-report                 # compare all retrievers on the clinical toy graph
```

No model download and no database are needed for any of the above: the toy graphs
ship with a deterministic bag-of-words encoder, so every number is reproducible and
hand-checkable.

Thirty seconds of API:

```python
from ikgqa.data import toy
from ikgqa.graph import TextualGraph
from ikgqa.retrieval import PCST

kg = toy.clinical_kg()
graph = TextualGraph.from_texts(kg.node_texts, kg.triples, kg.encoder, name="clinical")

selection = PCST(cost_e=0.5).retrieve(graph, kg.q("which complication did patient 001 have"))
print(selection.node_ids, selection.edge_ids)
```

When a result looks wrong, ask why rather than guessing. The prize table explains
almost every surprise:

```python
result = PCST().trace(graph, kg.q("..."))
print(result.trace.prize_table(graph.nodes))
```

---

## Layout

| Module | What lives there |
|---|---|
| `ikgqa.graph` | `TextualGraph`: nodes, edges and embeddings in one validated object |
| `ikgqa.encoders` | text to vectors — bag-of-words for development, SentenceBERT for real runs |
| `ikgqa.pcst` | the PCST algorithm, held byte-equivalent to the published G-Retriever code |
| `ikgqa.retrieval` | every selection strategy behind one interface |
| `ikgqa.eval` | metrics, parameter sweeps, reports |
| `ikgqa.data` | graphs to run on (toy graphs today, the STCS graph next) |

Each retriever answers exactly one question — *which nodes and edges should the LLM
see?* — and nothing else. It does not embed text, score itself, or call a model.
That narrowness is what makes PCST and its baselines interchangeable, which is the
whole premise of the comparison.

| Retriever | Family (thesis §2.3) | Size dial |
|---|---|---|
| `TopKTriples` | independent scoring, no connectivity guarantee | `k` |
| `TopKNodesPlusNeighbors` | neighbourhood expansion | none — degree decides |
| `BFSExpansion` | neighbourhood expansion, explicit hops | `hops` |
| `ShortestPaths` | path-based, connected but unbounded length | none |
| `PCST` | global objective, connectivity **and** size priced in | `cost_e` |

"Has no size dial" is a finding, not a gap. A recall-versus-size curve needs an axis
to sweep, and a retriever without one contributes a single point.

---

## Faithfulness to the published method

`ikgqa/pcst/reference.py` is a byte-for-byte copy of upstream
`G-Retriever/src/dataset/utils/retrieval.py` (verified against the repository's `main`
branch on 18 August 2026). `ikgqa/pcst/core.py` is the same algorithm decomposed into
named, individually testable stages, and `tests/test_pcst.py::test_matches_official_implementation`
asserts the two produce identical output — on hand-picked cases and on random graphs.

Two upstream quirks are reproduced on purpose rather than fixed, and are documented
inline as NOTE-A and NOTE-B. Deviating would make the comparison a comparison with
something other than G-Retriever.

**Environment note.** `pcst_fast`'s wheel is built against the NumPy 1.x C ABI. Under
NumPy 2 it imports fine and returns arrays of the right shape filled with garbage —
nothing raises. Hence `numpy<2` in `pyproject.toml`, and
`check_pcst_fast_sanity()`, which runs once before the first solve and fails loudly
rather than letting a broken environment produce plausible nonsense.

---

## How the evaluation stays honest

The thesis metric (§4.3) is answer-node recall at a given subgraph size. Three
decisions in `ikgqa.eval.metrics` guard against ways that could quietly lie:

1. **Aggregate questions are refused, not scored zero.** "How many patients had graft
   rejection in 2023" has no answer node — the count exists nowhere in the graph.
   Scoring it as a miss would punish every retriever and make the headline number
   meaningless. `Question.kind` marks these; results record `NaN` plus a reason. If
   most production questions turn out to be aggregates, that is a finding to report.
2. **Undefined is `NaN`, never `0.0`.** "No gold answer defined" and "retrieved
   nothing relevant" are different facts, and pandas skips `NaN` when averaging, so
   an undefined item cannot drag a mean down. Every mean is reported with `n_scored`.
3. **Size is measured three ways** — nodes, edges, prompt characters. They disagree.
   Characters is what actually consumes the context window, so it is the fairest axis
   when comparing dials that are not the same quantity (`cost_e` against `k`).

Numbers from `ikgqa.eval.toy_report` are **not** citable results. The toy graph and
its gold set were authored together, so any ranking they produce is a demonstration.
Real results come from the human-validated gold set on the STCS graph.

---

## Status

Done: PCST implementation with upstream equivalence, four baselines behind one
interface, the metric layer, parameter sweeps, 122 tests.

Next: the STCS graph loader (Neo4j), real SentenceBERT embeddings over SPHN node
text, the SnapQuery baseline harness, and the intent-augmented prize function.

An early empirical note from the sweep: above the top edge prize, `cost_e` stops
shrinking the subgraph, because `adjust_edge_cost` caps it at
`max_edge_prize * (1 - c/2)`. The size dial therefore saturates upward, which matters
for how the recall-versus-size curve is swept — sweep `cost_e` *downward* from small
values, and vary `topk`/`topk_e` to reach larger subgraphs.

## Development notes

The virtual environment currently lives at `PCST/.venv` for historical reasons
(that folder was the original playground). It works from anywhere; to relocate it,
create a fresh one at the repository root — moving a venv breaks its absolute paths.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e ".[dev,oracle]"
```

Licence: MIT. `ikgqa/pcst/reference.py` is (c) 2024 Xiaoxin He, MIT; see
`docs/LICENSE_G-Retriever`.
