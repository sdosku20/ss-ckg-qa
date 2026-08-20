# PCST Playground

A working, testable reconstruction of the subgraph retrieval step in **G-Retriever**
(He et al., *Retrieval-Augmented Generation for Textual Graph Understanding and Question
Answering*, NeurIPS 2024), built for poking at rather than for running experiments.

The original is a single 90-line function, `src/dataset/utils/retrieval.py::retrieval_via_pcst`.
An unmodified copy sits here as `reference_original.py`, and the test suite asserts that this
reimplementation produces **byte-identical output** to it on 23 cases.

## Files

| file | what it is |
|---|---|
| `pcst_retrieval.py` | the algorithm, split into one function per stage, plus a trace object that records every intermediate value |
| `toy_graphs.py` | four small graphs (a chain, a hub, a clinical KG, two disconnected triangles) and a deterministic toy text encoder |
| `baselines.py` | the three competing retrievers from Appendix D.1, plus a scoring harness |
| `test_pcst.py` | 55 tests, each named after a claim about the algorithm |
| `test_baselines.py` | 16 tests pinning down what separates PCST from the alternatives |
| `demo.py` | prints every stage for a graph and question of your choosing |
| `reference_original.py` | verbatim copy of the official implementation, used as the oracle |
| `LICENSE_G-Retriever` | MIT licence for that copy, (c) 2024 Xiaoxin He |

## Setup

```bash
pip install "numpy<2" pandas pcst_fast pytest      # minimum
pip install torch torch_geometric                  # optional, enables the equivalence tests
```

**`numpy<2` is not optional.** The `pcst_fast` 1.0.10 wheel is compiled against the NumPy 1.x
C ABI. Under NumPy 2 it still imports, still returns arrays of the correct *length*, and
returns **garbage values** without raising anything. `check_pcst_fast_sanity()` runs
automatically the first time the solver is called and will tell you if this has happened.
G-Retriever's own environment file pins Python 3.9 and NumPy 1.x, so upstream never hit it.

**On Windows, use WSL.** Two independent problems show up on native Windows, not one:

1. `numpy==1.26.4` ships wheels only through Python 3.12. If you are on a newer Python
   (3.13, 3.14, ...), pip has nothing to download and tries to compile numpy from source,
   which needs a C compiler Windows does not ship with. This is the error you get first,
   and it makes the *second* problem look like the only one.
2. `pcst_fast` has **no Windows build at all**, official or via conda. Its own bioconda
   package (`anaconda.org/bioconda/pcst-fast`) lists `linux-64` and `macOS-64` as supported
   platforms; Windows was never built, not just currently broken. Fixing (1) by installing
   Python 3.11 or 3.12 gets you past the numpy error and straight into this one: pip will try
   to compile pcst_fast's C++ source, which was only ever tested against g++/clang, using a
   compiler Windows does not have by default either.

The reliable fix for both at once is **WSL**: open PowerShell as administrator,
`wsl --install`, reboot, open the Ubuntu shell it gives you, and run every command in this
README there instead. That is a real Linux userspace, so `pip install pcst_fast` behaves
exactly as it did when this playground was built and tested. Docker is an equally valid
alternative if you already use it.

**You do not have to wait for WSL to start reading and running the code.** `pcst_fast` is
imported lazily inside `solve_pcst`, not at the top of the file, so `import pcst_retrieval`
always works with just numpy and pandas installed. `compute_node_prizes`, `compute_edge_prizes`,
`adjust_edge_cost` and `build_pcst_instance`, which is to say every stage up to "hand the
finished problem to the solver", run and can be tested with no solver at all. Only the very
last step, actually solving the Steiner tree, raises, and it raises with the paragraph above
rather than an import error, so the fix is in the message.
`test_prize_stages_work_without_pcst_fast_installed` in `test_pcst.py` checks this directly.

```bash
python -m pytest -v                  # 71 passed
python demo.py --graph chain
python demo.py --graph clinical --sweep
python demo.py --list
python baselines.py                  # PCST vs the three Appendix D.1 baselines
```

---

# The explanation, as if you were a kid

## The setup

Imagine a big map of **islands** joined by **bridges**. Each island has something written on
it, like *"creatinine 2.1 mg/dl"*, and each bridge has something written on it too, like
*"has lab result"*. That map is the knowledge graph.

Now someone asks a question: *"what was patient 001's creatinine?"*

You want to hand a robot a **small piece of the map** that contains the answer. Not the whole
map, because the robot can only read a little bit at a time. Not just the single best island,
because a lonely island with no bridges does not tell the robot how anything is connected.

So you need to choose a small, **joined-up** piece of the map. That is the whole problem.

## Turning it into a game with coins

Here is the trick the paper uses. Turn it into a game.

**Step 1: put coins on the good islands.**
Read the question, then look at every island and ask *"how much does this island sound like
the question?"* Rank them. The best-sounding island gets 3 coins, the next gets 2, the next
gets 1, and every other island gets nothing. (3 because we chose `topk=3`.)

Notice something sneaky: only the *ranking* matters. If the best island is a perfect match and
the second is a terrible match, they still get 3 coins and 2 coins. The numbers say nothing
about how good the match really was.

**Step 2: put coins on the good bridges too.**
Same idea, for the writing on the bridges. But bridges have a problem that islands do not: in
a real knowledge graph, hundreds of bridges have the *exact same* writing on them, like *"has
lab result"*. They all sound equally like the question, so they all tie. The code decides they
have to **share** the coins for their place in the ranking. If ten bridges tie for first place
and first place is worth 2 coins, each bridge gets 0.2 coins.

That has a real consequence. A boring bridge label that appears everywhere is worth almost
nothing each. A rare, unusual bridge label keeps its whole pile. The algorithm quietly prefers
rare relations.

**Step 3: charge a toll for every bridge you use.**
Every bridge costs you `cost_e` coins to walk across. This is the only thing stopping you from
grabbing the entire map: if you take a bridge, it has to be worth the toll.

**Step 4: play the game.**
Choose a set of islands and bridges that is all **one connected piece**, so that you collect as
many coins as possible while paying as little toll as possible. Maximise
*(coins collected minus tolls paid)*.

That game already has a name in mathematics, from decades before anyone thought about language
models: the **Prize-Collecting Steiner Tree**. Prizes are the coins, the Steiner tree is the
connected piece, and a fast solver for it already exists (`pcst_fast`). This is the actual
contribution of the paper: noticing that "find me a small relevant connected piece of a graph"
*is* this old problem in disguise.

## The clever bit: coins on bridges are not allowed

The old solver has a rule: **coins go on islands, tolls go on bridges.** You cannot put coins
on a bridge.

So what do you do with a bridge worth 3 coins whose toll is only 0.5?

* **Easy case, small coins.** If the bridge is worth 0.2 coins and the toll is 0.5, just make
  the bridge cheaper: charge 0.3 instead of 0.5. Same thing, and the solver is happy.
* **Hard case, big coins.** If the bridge is worth 3 coins and the toll is 0.5, the same trick
  would mean charging *minus* 2.5 coins, and the solver refuses negative tolls.

  So: **knock the bridge down and build a tiny island in the middle of the gap.**
  ```
  before:   island A ====[worth 3 coins, toll 0.5]====> island B      not allowed

  after:    island A --[toll 0]--> tiny island --[toll 0]--> island B
                                    (2.5 coins)
  ```
  Walking across both halves is free, and stepping onto the tiny island in the middle gives you
  2.5 coins, exactly the profit the original bridge would have given. And you can only reach
  the tiny island from A or B, so it cannot be collected without also joining A and B, just
  like a real bridge. The paper calls these **virtual nodes**.

Afterwards you have to remember what you did, so a little notebook records *"tiny island #8
really means bridge #0"*. When the solver hands back its answer, any tiny island in the answer
gets translated back into a real bridge.

## The best part

When the solver plays the game, it sometimes buys islands with **zero coins on them**. Why
would it pay tolls for a worthless island?

Because coins are only collectable if everything is one connected piece. If *creatinine* is at
one end of the map and *tacrolimus* is at the other, the only way to have both is to walk
across all the boring islands in between. So the boring islands come along for the ride.

**That is the entire reason to use this instead of ordinary top-k search.** Top-k gives you
{creatinine, tacrolimus} and no idea how they relate. PCST gives you the path between them,
which is exactly the context the language model needs to answer the question. Run
`python demo.py --graph chain` to watch it happen, and see
`test_zero_prize_bridge_nodes_get_pulled_in`.

## The code, piece by piece

Each stage is one function in `pcst_retrieval.py`.

| function | in kid words | in paper words |
|---|---|---|
| `cosine_similarity` | how much do these two bits of writing sound alike | cosine between SBERT embeddings, Eq. 4-5 |
| `compute_node_prizes` | put 3, 2, 1 coins on the three best islands | Eq. 6 |
| `compute_edge_prizes` | same for bridges, but tied bridges share their coins | "edge prizes are assigned similarly", plus tie handling only in the code |
| `adjust_edge_cost` | make sure the toll is cheap enough that at least one good bridge is worth crossing | not in the paper, only in the code |
| `build_pcst_instance` | cheapen the small-coin bridges, replace the big-coin bridges with tiny islands | Section 5.3, virtual nodes |
| `solve_pcst` | play the game | Eq. 7, solved by `pcst_fast` |
| `decode_solution` | translate tiny islands back into bridges, then add any island that a chosen bridge touches | the connectivity guarantee |
| `build_description` | write the chosen piece of map out as two CSV tables for the language model | `textualize(S*)`, Eq. 11 |

`retrieval_via_pcst_traced` runs all of them and hands back a `RetrievalTrace` holding every
intermediate number, so you can print `trace.prize_table(...)` and `trace.edge_table(...)` and
see exactly why each thing was chosen.

---

# Comparing PCST against something

`baselines.py` implements the three retrievers the paper compares against in
Appendix D.1, all with the same signature so they are directly swappable:

* **top-k triples (KAPING)**, rank triples by similarity, keep the best k;
* **top-k nodes plus neighbours**, take the best k nodes and everything one hop away;
* **shortest paths**, take the best k nodes and the shortest path between every pair.

Table 10 in the paper reports end-to-end Hit@1 on WebQSP (PCST 66.17, KAPING 52.64,
neighbours 49.82, shortest path 55.20), which needs an LLM. What you can measure without one
is retrieval quality against a gold set. `toy_graphs.clinical_gold_set()` has five questions
with hand-written answer triples, and `baselines.score_all` scores all four retrievers on them.

```
                          recall  precision  nodes  edges  chars  always_connected
PCST                         0.9      0.313    6.2    5.2  299.4              True
top-k triples (KAPING)       0.9      0.320    6.4    5.0  300.6             False
top-k nodes + neighbours     0.8      0.100   15.0   14.2  747.8              True
shortest paths               0.4      0.133    5.6    4.8  261.8              True
```

**Read this honestly.** On a 20-edge graph with a bag-of-words encoder, PCST does *not* beat
KAPING on recall or precision. It ties on recall and loses precision by 0.007. Do not cite this
table as a reproduction of Table 10, because it is not one: the paper's gap comes from WebQSP,
where graphs average 1370 nodes and 4252 edges and independently chosen triples fragment badly.
Twenty edges is not enough room for that failure to show up.

What *is* already visible at this scale is worth more than a score, because it is structural
rather than empirical:

1. **PCST is connected by construction, KAPING is not.** Two of the five questions give KAPING a
   result in more than one piece. No amount of tuning fixes that, because nothing in top-k
   selection refers to the graph.
   (`test_topk_triples_can_return_a_disconnected_bag_of_facts`)

2. **PCST can find an answer whose node text does not match the question.** Ask "which
   complication did patient 001 have". The answer is *graft rejection episode*, which shares no
   word with the question, so it scores exactly 0.0 and never becomes a seed. Both node-seeded
   baselines miss it. PCST finds it, because it prizes *edges* too, and the relation
   *has complication* does match. That edge prize becomes a virtual node, and buying the virtual
   node drags in both endpoints. This is the mechanism behind the shortest-path baseline's 0.4
   recall, and it is the single most useful thing in the comparison.
   (`test_shortest_paths_needs_the_answer_node_to_match_the_question`)

3. **Only PCST has a size dial.** For the neighbourhood baseline, `k` chooses how many seeds to
   expand, not how much comes back, so output size jumps in whatever increments the node degrees
   happen to be (6, then 11, then 12 edges). `cost_e` gives PCST smooth control over the same
   quantity.
   (`test_topk_nodes_plus_neighbors_has_no_size_dial`)

Two implementation choices, since the paper describes each baseline in one sentence and the
released code does not include them: KAPING scores whole triples, so it gets embeddings of the
concatenated head/relation/tail text rather than the relation alone, which would handicap it
unfairly; and "top-k nodes plus neighbours" is read as every edge incident to a top-k node.

---

# Things that will bite you

Found by writing the tests. Each one has a test naming it.

1. **`cost_e` is a ceiling, not a setting.** `adjust_edge_cost` silently lowers it to
   `max_edge_prize * (1 - c/2)`. Asking for `cost_e=5` on a graph whose best edge prize is 1.0
   gives you 0.995. So `cost_e` is not comparable across graphs or across questions, and a
   hyperparameter sweep above the cap does nothing.
   (`test_cost_floor_caps_cost_e_so_raising_it_stops_helping`)

2. **If no edge matches the question, you retrieve the entire connected component.** With no
   overlap at all, every edge similarity is identical, so every edge lands in one tier, so
   `max_edge_prize` is tiny, so `cost_e` collapses to nearly zero, so every edge becomes almost
   free. The "retrieval" step returns the whole graph. Watch for it with
   `python demo.py --graph clinical --query "what was the creatinine value for patient 001"`,
   where none of the relation names share a word with the question.

3. **`topk_e` gets clamped to the number of *distinct* edge similarities**, and the top tier is
   worth the clamped value. On a graph with 2 distinct edge similarities, `topk_e=5` behaves
   like `topk_e=2` and the top tier is worth 2, not 5. This changes the prize scale, and
   therefore the meaning of `cost_e`, from graph to graph.
   (`test_edge_prizes_split_the_tier_budget_between_tied_edges`)

4. **Ties make the top-k arbitrary.** When a question shares no vocabulary with most of the
   graph, dozens of nodes score exactly 0.0, and whichever ones `torch.topk` happens to return
   get real prizes. The original inherits `torch.topk`'s undocumented tie order. This module
   defaults to `tie_break="auto"`, which delegates to `torch.topk` for exact parity, and offers
   `tie_break="stable"` for a reproducible torch-free alternative.
   (`test_tied_similarities_make_the_top_k_arbitrary`)

5. **`num_clusters=1` means one connected component, always.** If the answer genuinely lives in
   two disconnected parts of the graph, you can only get one of them. Relevant for clinical
   graphs assembled from separate sources where the join may be missing.
   (`test_only_one_component_survives_when_the_graph_is_disconnected`)

6. **The text and the tensors use different node numbering.** The CSV description keeps the
   *original* node ids, while the returned `Data` object is renumbered `0..n-1`. Fine for
   G-Retriever, because the LLM only reads the text and the GNN only reads the tensors, but a
   trap if you try to align them, for example to attribute an answer back to a source triple.
   `RetrievalResult.selected_nodes` is the map between them.
   (`test_subgraph_is_reindexed_but_topologically_identical`)

7. **Prizes are ranks, not scores.** A node ranked first gets `topk` coins whether its
   similarity was 0.99 or 0.01. Two questions with very different retrieval quality produce
   identical prize vectors. Any analysis of *why* a subgraph was chosen has to look at
   `trace.node_similarity`, not `trace.node_prizes`.
   (`test_node_prizes_ignore_how_similar_a_node_actually_is`)

8. **`pcst_fast` fails silently under NumPy 2.** See Setup.

---

# Swapping in real data

The toy encoder exists so the playground needs no downloads and every prize is checkable by
hand. It builds an L2-normalised bag-of-words vector over the graph's own vocabulary, so
cosine similarity is literally word overlap.

To use the real thing, replace the embeddings and keep everything else:

```python
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data.data import Data
import pcst_retrieval as P

enc = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

graph = Data(
    x=torch.tensor(enc.encode(nodes_df.node_attr.tolist())),
    edge_index=torch.tensor([nodes_src, nodes_dst], dtype=torch.long),
    edge_attr=torch.tensor(enc.encode(edges_df.edge_attr.tolist())),
    num_nodes=len(nodes_df),
)
q_emb = torch.tensor(enc.encode(question))

result = P.retrieval_via_pcst_traced(graph, q_emb, nodes_df, edges_df,
                                     topk=3, topk_e=5, cost_e=0.5)   # WebQSP settings
print(result.desc)
print(result.trace.prize_table(nodes_df))
```

`retrieval_via_pcst` accepts a PyG `Data` or a plain-numpy `SimpleGraph` interchangeably, and
returns whichever type it was given. `textual_nodes` needs a `node_attr` column, `textual_edges`
needs `src`, `edge_attr`, `dst`, matching the CSV format G-Retriever's preprocessing produces.

Published hyperparameters, from Appendix B.1 of the paper: `topk=3, topk_e=3, cost_e=1.0` for
SceneGraphs, `topk=3, topk_e=5, cost_e=0.5` for WebQSP, and `topk=0` for ExplaGraphs because it already
fits in the context window. Two things the appendix and the released code disagree about, worth
knowing if you cite either: the code passes `cost_e=0.5` for SceneGraphs where the appendix says
1.0, and for ExplaGraphs the code never calls `retrieval_via_pcst` at all rather than calling it
with `topk=0`.

# Verification

`test_matches_official_implementation` and `test_matches_official_implementation_on_random_graphs`
run `reference_original.py` and this module on the same inputs and require identical
descriptions and identical tensors. 23 cases: four toy graphs across a range of
`topk`/`topk_e`/`cost_e` including the disabled-prize paths, plus twelve random 40-node graphs
with a 6-relation vocabulary so that edge-similarity ties are exercised. The two deliberate
deviations from the original are both documented in `pcst_retrieval.py`: array conversion is
unconditional rather than only when virtual nodes exist, and the zero-edge case returns a
single node instead of reaching the solver. Neither changes any output the original produces
without crashing. The prize vector is deliberately kept in float32 to match the dtype the
original hands to the solver.
