# PCST from the five papers

Written 3 September 2026, after the supervision meeting where I could not explain
how PCST works. The point of this document is to be able to derive it on a
whiteboard, not to have read about it.

Notation is plain text on purpose, so this reads the same in a text editor as in
a browser. "sum over v in S of p(v)" means what it says. Everything below is
checked against the sources or against a run. The runnable companion is
`experiments/pcst_by_hand.py`, which traces the algorithm event by event on a
four-node graph and then calls the real solver on the same instance.

---

## 1. What the five references are, and the order to read them

They are not five views of one paper. They are the lineage of the algorithm our
code calls, and each one fixes a specific shortcoming of the previous.

| # | Reference | What it contributes | Read it for |
|---|---|---|---|
| 1 | Bienstock, Goemans, Simchi-Levi, Williamson, *A note on the prize collecting traveling salesman problem*, Math. Prog. 59 (1993) 413-420 | **Introduces PCST.** LP relaxation + rounding. Factor 2.5 for PC-TSP (Thm 3.1); factor 3 for the Steiner tree version (section 4) | Where the problem comes from, and why a *combinatorial* algorithm was wanted |
| 2 | Goemans & Williamson, *A general approximation technique for constrained forest problems*, SIAM J. Comput. 24(2) (1995) 296-317 | The **primal-dual factor-2 algorithm**. This is the algorithm we run | The actual mechanism, the LP/dual pair, the analysis |
| 3 | Goemans & Williamson, *The primal-dual method for approximation algorithms...* (Hochbaum, ed., 1997; the Waterloo PDF) | Textbook exposition of the same method, with the **moat** picture | Intuition. Read this *before* #2 if LP duality is rusty |
| 4 | Johnson, Minkoff, Phillips, *The prize collecting Steiner tree problem: theory and practice*, SODA 2000, 760-769 | **Strong pruning** (exact DP on the growth-stage tree) and direct handling of the unrooted variant | Why there is a `pruning=` argument at all |
| 5 | Hegde, Indyk, Schmidt, *A fast, adaptive variant of the GW scheme for PCST*, DIMACS 11th Impl. Challenge (2014) | `pcst_fast`. O(d*m*log n), keeps the factor-2 guarantee | **This is the binary our code calls** |

Paper 1 solves an LP with the ellipsoid method (the separation problem is min-cut)
and rounds: scale x-hat = (5/3) * x-bar, then set y-hat(i) = 1 exactly when
y-bar(i) >= 3/5, then run Christofides. That is why paper 2 was a breakthrough:
same problem, better factor, no LP solver at all.

---

## 2. The problem, in the form every one of those papers uses

Graph G = (V, E). Every node v has a prize p(v) >= 0. Every edge e has a cost
c(e) >= 0.

> **PCST (unrooted).** Find a subtree T of G that minimises
>
>     c(T)  +  pi(complement of T)
>
> where c(T) is the total cost of the edges in T, and pi(complement of T) is the
> total prize of the nodes that are NOT in T.

Buy edges, or forfeit prizes. Hegde et al. state exactly this (their Definition 1).

**Rooted vs unrooted.** GW's original algorithm is *rooted*: one designated node
must be in T. Unrooted reduces to rooted by trying all n roots, at a factor n in
running time, which is why GW is O(n^2 log n) rooted but O(n^3 log n) unrooted.
Papers 4 and 5 handle unrooted directly. We pass `root = -1`, which is the
unrooted case, and that is the right choice: rooting means naming a node the tree
must contain, i.e. assuming where the answer is.

---

## 3. The bridge to G-Retriever: the derivation to have ready

This is the question I could not answer.

The thesis states the objective as a **maximisation** over connected subgraphs S:

    F(S)  =  (sum over v in S of p(v))  -  (sum over e in S of c(e))

The literature states a **minimisation**. They are the same problem.

Let PI be the total prize on the whole graph:

    PI  =  sum over all v in V of p(v)

PI is a constant. It depends on the graph and the question, never on which
subgraph you pick. Every node is either in S or outside it, and nowhere else, so
the total prize splits in two with nothing left over:

    PI  =  (sum over v in S of p(v))  +  (sum over v not in S of p(v))
        =  collected(S)  +  forfeited(S)

Rearranged:

    collected(S)  =  PI  -  forfeited(S)

Substitute that into F:

    F(S)  =  collected(S)          -  cost(S)
          =  PI - forfeited(S)     -  cost(S)
          =  PI  -  [ cost(S) + forfeited(S) ]

The bracket is exactly the PCST minimisation objective. PI is a constant, so
**maximising F(S) is the same as minimising cost(S) + forfeited(S)**. Not merely
similar: the subgraph that maximises one is the identical subgraph that minimises
the other, and the two optimal values sum to PI.

Numerical check on the worked example in section 6. Prizes 10, 0, 10, 1, so
PI = 21. The best tree scores 9 on the minimisation. So the maximisation value
should be 21 - 9 = 12. Check it directly: collected = 10 + 0 + 10 = 20, cost
= 4 + 4 = 8, F = 20 - 8 = 12. Agrees.

Say it in that order: the maximisation is the thesis's presentation, the
minimisation is what the solver implements, the offset is the total prize.

---

## 4. Where the moats come from: the LP and its dual

For the rooted problem with root r, the primal relaxation (GW95 section 4;
confirm the exact form there, this is the standard statement).

Variables: x(e) for each edge, meaning "buy this edge". z(T) for each node set T
not containing the root, meaning "give up on everything in T".

    minimise    sum over e of c(e)*x(e)  +  sum over T of pi(T)*z(T)

    subject to  for every set S not containing the root:

                  sum of x(e) over edges e crossing out of S
                + sum of z(T) over sets T that contain S
                >= 1

                x, z >= 0

Read the constraint as a forced choice: for every set S cut off from the root,
either buy an edge that crosses out of S and connect it, or pay the penalty for
some set that swallows S and give up on it. Doing neither is infeasible.

The dual has one variable y(S) per node set S:

    maximise    sum over S of y(S)

    subject to  for every edge e:
                  sum of y(S) over sets S whose boundary contains e  <=  c(e)
                                                          [edge constraint]

                for every set T:
                  sum of y(S) over sets S contained in T  <=  pi(T)
                                                       [cluster constraint]

                y >= 0

Those two dual constraints are the whole algorithm. Picture y(S) as the **width
of a moat** drawn around the set S. Then:

- **edge constraint**: the moats an edge crosses never total more than the edge
  costs, i.e. moats never spill over an edge;
- **cluster constraint**: the moats drawn inside a set never total more than the
  prizes in it, i.e. a cluster cannot spend more than it is worth.

Hegde et al. section 2 state both invariants in exactly this language, and call an
inequality *tight* when it holds with equality.

Why growing the dual solves the primal. Any feasible dual value is a lower bound
on the primal optimum (weak duality), so the total moat width grown is a
certificate. Complementary slackness says a primal variable may go positive only
when its dual constraint is tight. So "edge constraint tight" is the algorithm
earning the right to buy that edge, and "cluster constraint tight" is a cluster
having spent its entire worth and therefore stopping. The dual growth schedule
*is* the primal construction rule. That is what primal-dual means here, and it is
where the factor 2 comes from.

---

## 5. The algorithm

Two stages: **growth**, then **pruning**.

### Growth stage

Clusters form a *laminar family*: any two clusters are disjoint or one contains
the other, so the maximal ones always partition V. Each cluster is active or
inactive, and only maximal clusters can be active. Every node starts as its own
active singleton with moat 0. Alongside, keep an edge set F which, restricted to
any cluster, spans it.

```
while some cluster is active:
    grow y(C) for every active cluster C at the same unit rate,
      until an edge constraint or a cluster constraint goes tight

    if edge e = (u,v) went tight:
        deactivate the maximal clusters containing u and v
        add their union to the family as a new ACTIVE cluster, with moat 0
        drop every edge with both endpoints inside that union
        add e to F

    if the cluster constraint for C went tight:
        mark C inactive

run a pruning scheme on F restricted to the last active cluster
```
(after Hegde et al., Algorithm 1)

Three things worth noticing, because they are what gets asked:

1. **The loop runs at most 2n times.** Every iteration either reduces the cluster
   count by one, or deactivates a cluster while creating none.
2. **A merged cluster is active again with a fresh moat of zero.** So a node whose
   own prize was long exhausted can start growing again as part of something
   bigger. This is why the analysis is not a simple greedy argument.
3. **A prize-zero node is never active.** Its cluster constraint says
   sum of moats <= 0, which is tight before the clock starts. Such a node can only
   enter the answer by being swallowed in a merge, which is exactly what a Steiner
   node is.

### Rates, and why speeding this up is hard

Slack on an edge with **two** active endpoints drains at rate 2; with **one**
active endpoint, at rate 1; with none, not at all. Worse, an edge's type changes
over time, because a deactivated cluster becomes active again when it merges. So
one cluster changing state alters the predicted tight-time of many edges at once.

The fix is **dynamic edge splitting** (Cole, Hariharan, Lewenstein, Porat, SODA
2001): put an inactive sentinel node at the midpoint of every two-active-endpoint
edge, so every edge has at most one active endpoint and all live slacks drain at
the same rate. Each merge involving a sentinel halves the remaining distance.
Hegde et al.'s contribution is to move sentinels *adaptively*, which drops the
O(k*m*log^2 n) of Cole et al. to O(d*m*log n), or O(m log n) at constant
precision, and unlike Cole et al. without the extra 2/(n*k) term in the
guarantee.

### Pruning

The growth stage over-collects: F can span nodes that do not pay for themselves.
Two schemes:

- **GW pruning** (paper 2) depends on the laminar family built during growth, and
  removes branches lying in clusters that were deactivated.
- **Strong pruning** (paper 4) ignores the laminar family and solves PCST
  *exactly* on the growth-stage tree with a linear-time dynamic program. Its ratio
  is at least as good as GW pruning's.

`pcst_fast` exposes `none`, `simple`, `gw`, `strong`. We pass `gw`, because that is
what G-Retriever passes and we reproduce it exactly. **Note for the thesis:
`strong` is documented as never worse, so "why gw?" has only a reproduction
answer, not a quality one.**

### Guarantees

GW95 gives the **Lagrangian-preserving** bound

    c(T) + 2*pi(complement of T)  <=  2*c(T_opt) + 2*pi(complement of T_opt)

which is stronger than a plain factor 2 because of the 2 multiplying the penalty
term on the left. Best known ratio is **1.9672** (Archer, Bateni, Hajiaghayi,
Karloff 2011; Hajiaghayi, Khandekar 2013). Approximating Steiner tree, and hence
PCST, within **96/95** is NP-hard (Chlebik, Chlebikova 2002). So the gap left to
close is narrow, and the factor 2 we inherit is close to what anyone gets.

---

## 6. Worked example: trace this before the next meeting

    a(10) --4-- b(0) --4-- c(10) --5-- d(1)

Node prizes in brackets, edge costs on the lines.

The optimum is {a, b, c}: cost 8, forfeited prize 1, objective 9. Check the
alternatives yourself. Take everything: cost 13, forfeit 0, total 13. Take only a:
cost 0, forfeit 11, total 11. Take nothing: 21.

The trace, verified by `experiments/pcst_by_hand.py`:

| time | event |
|---|---|
| 0 | {b} deactivates. Prize 0, so its cluster constraint is tight already |
| 1 | {d} deactivates. Its moat reached its prize of 1 |
| 4 | edge a-b tight, merge into {a,b}, fresh moat 0 |
| 4 | edge b-c tight, merge into {a,b,c} |
| 4 | edge c-d tight, merge into {a,b,c,d} |
| 16 | the single remaining cluster goes tight, nothing active, stop |

Why c-d goes tight at time 4 although its cost is 5: both c and d were active
until time 1, so slack drained at rate 2 down to 3, and then at rate 1 once d had
stopped. 1 + 3 = 4.

F then spans **all four nodes**. The growth stage does not exclude d.
**Pruning removes d**, and the solver returns {a, b, c}, which is optimal here
even though only a factor 2 is promised. Two lessons. The answer to "how does it
decide what to leave out" is *the pruning stage*. And b is in the answer with
prize 0 purely because it sits between two valuable nodes.

---

## 7. What our own code does with all this

`src/ikgqa/pcst/core.py`, with `reference.py` beside it holding the verbatim
published routine used as a test oracle.

- `root = -1` (unrooted), `num_clusters = 1` (one connected component),
  `pruning = "gw"`, `verbosity = 0`.
- **Edge prizes are not part of PCST.** The solver accepts node prizes and edge
  costs only. G-Retriever wants prizes on edges too, and handles it in two
  branches. Write p for the edge's prize and c_e for the standard edge cost:
  - If p <= c_e: keep the edge, and lower its cost to c_e - p, which is still
    non-negative and therefore legal. Taking it changes the objective by
    -(c_e - p) = p - c_e.
  - If p > c_e: the cost c_e - p would be negative, which the solver rejects. So
    delete the edge, and insert a **virtual node** carrying prize p - c_e, joined
    to the two former endpoints by **two zero-cost half-edges**. Taking that
    two-step path collects p - c_e and pays 0, so it changes the objective by
    p - c_e as well. Same value, legal shape. And the virtual node is still only
    reachable by passing through both original endpoints.

  > **Correction to what I wrote before:** the half-edges cost **0**, not c_e/2.
  > See `reference.py:63-64`, which appends 0 twice. At c_e/2 each, the net would
  > be p - 2*c_e and the transform would not preserve the objective.

- The constant 0.01 (called gamma in the thesis) enters as
  `c_e <- min(c_e, max_over_edges(p) * (1 - gamma/2))`, which guarantees the
  highest-prize edge always becomes a virtual node.

---

## 8. The whole-graph result, explained mechanically

This is the thesis's headline finding, that PCST with published defaults returned
15,810 of 15,810 nodes. Chapter 3 currently explains it through the objective
degenerating. The algorithm gives a sharper account. **Verified by running
`pcst_fast` directly, not reasoned about:**

1. Edge prizes are tier-split, p(e) = b(tier) / n(tier). With few, heavily shared
   relation types, one enormous tier hands every member the same tiny positive
   prize.
2. The gamma cap drags c_e down to just under the largest edge prize, so c_e lands
   on the order of 10^-6.
3. So nearly every edge takes the *second* branch above: it becomes a virtual node
   with **positive** prize reached by **zero-cost** half-edges.
4. A free path to a positive prize is always worth taking. The solver takes those
   virtual nodes.
5. Decoding turns each selected virtual node back into an edge, and then
   `decode_solution` **closes the node set under the selected edges**, adding both
   endpoints regardless of their prize (`core.py:565-570`).
6. Every node is an endpoint of some such edge. Hence essentially the whole graph.

Two things this corrects. First, **pruning is not the failure point**: on a star
of zero-prize leaves, `gw` pruning correctly returns one single node. But it
prunes the solver's *node* set, and the decoder re-adds the endpoints afterwards,
so pruning cannot undo the closure. Second, cost collapse alone does not do it:
zero-prize leaves at cost 10^-6 are still pruned. It is the **virtual nodes plus
the closure**, not the small costs on their own.

Worth raising as a finding rather than defending. The closure step is what makes
the returned subgraph connected and contextual, and it is also what makes the size
dial fail.

---

## 9. Self-test: answer without notes

1. State PCST as a minimisation. Now show it is the thesis's maximisation.
2. Write the dual. Which constraint is a moat not spilling over an edge?
3. Why is a prize-zero node never active, and how does it get into the answer?
4. In the worked example, why does c-d go tight at time 4 and not 5?
5. Which stage decides what to leave out?
6. Two active endpoints versus one: what differs, and why does that make a fast
   implementation hard?
7. What do the sentinel nodes achieve?
8. GW pruning versus strong pruning: which is better, and why do we use the other?
9. State the Lagrangian-preserving guarantee. Why is it stronger than factor 2?
10. Edge prizes are not part of PCST. Where do they go, and what does a virtual
    node's half-edge cost?
11. Why unrooted, and what does rooting cost in running time?
12. Trace the whole-graph result from tie tiers to the final node set.

If 1, 3, 5 and 12 are fluent, the rest is recoverable in the room.
