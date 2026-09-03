# PCST from the five papers

Written 3 September 2026, after the supervision meeting where I could not explain
how PCST works. The point of this document is to be able to derive it on a
whiteboard, not to have read about it.

Everything below is checked against the sources or against a run. The runnable
companion is `experiments/pcst_by_hand.py`, which traces the algorithm event by
event on a four-node graph and then calls the real solver on the same instance.

---

## 1. What the five references are, and the order to read them

They are not five views of one paper. They are the lineage of the algorithm our
code calls, and each one fixes a specific shortcoming of the previous.

| # | Reference | What it contributes | Read it for |
|---|---|---|---|
| 1 | Bienstock, Goemans, Simchi-Levi, Williamson, *A note on the prize collecting traveling salesman problem*, Math. Prog. 59 (1993) 413–420 | **Introduces PCST.** LP relaxation + rounding. Factor 2.5 for PC-TSP (Thm 3.1); factor 3 for the Steiner tree version (§4) | Where the problem comes from, and why a *combinatorial* algorithm was wanted |
| 2 | Goemans & Williamson, *A general approximation technique for constrained forest problems*, SIAM J. Comput. 24(2) (1995) 296–317 | The **primal-dual factor-2 algorithm**. This is the algorithm we run | The actual mechanism, the LP/dual pair, the analysis |
| 3 | Goemans & Williamson, *The primal-dual method for approximation algorithms…* (Hochbaum, ed., 1997; the Waterloo PDF) | Textbook exposition of the same method, with the **moat** picture | Intuition. Read this *before* #2 if LP duality is rusty |
| 4 | Johnson, Minkoff, Phillips, *The prize collecting Steiner tree problem: theory and practice*, SODA 2000, 760–769 | **Strong pruning** (exact DP on the growth-stage tree) and direct handling of the unrooted variant | Why there is a `pruning=` argument at all |
| 5 | Hegde, Indyk, Schmidt, *A fast, adaptive variant of the GW scheme for PCST*, DIMACS 11th Impl. Challenge (2014) | `pcst_fast`. $O(dm \log n)$, keeps the factor-2 guarantee | **This is the binary our code calls** |

Paper 1 solves an LP with the ellipsoid method (the separation problem is min-cut)
and rounds: scale $\hat{x} = \tfrac{5}{3}\bar{x}$, threshold $\hat{y}_i = 1$ iff
$\bar{y}_i \ge \tfrac{3}{5}$, then run Christofides. That is why paper 2 was a
breakthrough: same problem, better factor, no LP solver at all.

---

## 2. The problem, in the form every one of those papers uses

Graph $G=(V,E)$, edge costs $c: E \to \mathbb{R}^+_0$, node prizes
$\pi: V \to \mathbb{R}^+_0$.

> **PCST (unrooted).** Find a subtree $T$ of $G$ minimising
> $c(T) + \pi(\bar{T})$.

Buy edges, or forfeit prizes. $\bar T$ is the complement, so $\pi(\bar T)$ is the
prize you walked away from. Hegde et al. state exactly this (their Definition 1).

**Rooted vs unrooted.** GW's original algorithm is *rooted*: a designated node
must be in $T$. Unrooted reduces to rooted by trying all $n$ roots, at a factor
$n$ in running time — which is why GW is $O(n^2 \log n)$ rooted but
$O(n^3 \log n)$ unrooted. Papers 4 and 5 handle unrooted directly. We pass
`root = -1`, which is the unrooted case, and it is the right choice: rooting means
naming a node the tree must contain, i.e. assuming where the answer is.

---

## 3. The bridge to G-Retriever — the derivation to have ready

This is the question I could not answer, and it is one line of algebra.

The thesis states the objective as a **maximisation** over connected subgraphs $S$:

$$F(S) = \sum_{v \in S} p(v) - \sum_{e \in S} c(e)$$

The literature states a **minimisation**. They are the same problem. Let
$\Pi = \sum_{v \in V} p(v)$, a constant fixed by the instance. Then

$$\sum_{v \in S} p(v) = \Pi - \sum_{v \notin S} p(v) = \Pi - \pi(\bar{S})$$

so

$$F(S) = \Pi - \pi(\bar{S}) - c(S) = \Pi - \bigl[\,c(S) + \pi(\bar{S})\,\bigr]$$

$\Pi$ does not depend on $S$, so **maximising $F(S)$ is exactly minimising
$c(S) + \pi(\bar S)$**, the standard PCST objective. Collecting a prize and
declining to forfeit it are the same act, counted from opposite ends.

Say it in that order: the maximisation is the thesis's presentation, the
minimisation is what the solver implements, the offset is the total prize.

---

## 4. Where the moats come from: the LP and its dual

For the rooted problem, the primal relaxation (GW95 §4 — confirm the exact form
there, this is the standard statement):

$$
\begin{aligned}
\min \quad & \sum_{e \in E} c_e x_e + \sum_{T \subseteq V \setminus \{r\}} \pi(T)\, z_T \\
\text{s.t.} \quad & \sum_{e \in \delta(S)} x_e + \sum_{T \supseteq S} z_T \ \ge\ 1
  \qquad \forall\, S \subseteq V \setminus \{r\} \\
& x, z \ \ge\ 0
\end{aligned}
$$

Read the constraint as: for every set $S$ not containing the root, either buy an
edge crossing out of $S$, or pay to give up on a set containing $S$.

The dual has one variable $y_S$ per set:

$$
\begin{aligned}
\max \quad & \sum_{S} y_S \\
\text{s.t.} \quad & \sum_{S \,:\, e \in \delta(S)} y_S \ \le\ c_e \qquad \forall\, e \in E
  &&\text{(edge constraint)}\\
& \sum_{S' \subseteq T} y_{S'} \ \le\ \pi(T) \qquad \forall\, T
  &&\text{(cluster constraint)}\\
& y \ \ge\ 0
\end{aligned}
$$

Those two dual constraints are the whole algorithm. Picture $y_S$ as the **width
of a moat** drawn around the set $S$. Then:

- *edge constraint*: the moats crossing an edge never sum to more than the edge
  costs — **moats never spill over an edge**;
- *cluster constraint*: the moats inside a set never sum to more than the prizes
  in it — **a cluster cannot spend more than it is worth**.

Hegde et al. §2 state both invariants in exactly this language, and call an
inequality *tight* when it holds with equality. The algorithm grows moats until
something goes tight, and every tight constraint is a complementary-slackness
condition being satisfied. That is what "primal-dual" means here: the dual growth
schedule *is* the primal construction rule.

---

## 5. The algorithm

Two stages: **growth**, then **pruning**.

### Growth stage

Clusters form a *laminar family* $\mathcal{L}$ — any two clusters are disjoint or
nested, so the maximal ones always partition $V$. Each cluster is active or
inactive; only maximal clusters can be active. Every node starts as its own
active singleton with moat $0$. Alongside, keep an edge set $F$ that restricted to
any cluster spans it.

```
while some cluster is active:
    grow y_C for every active cluster C at unit rate,
      until an edge constraint or a cluster constraint goes tight

    if edge e = (u,v) went tight:
        deactivate the maximal clusters C_u, C_v
        add C' = C_u u C_v to L as active, with y_C' = 0
        drop every edge with both endpoints inside C'
        add e to F

    if the cluster constraint for C went tight:
        mark C inactive

run a pruning scheme on F restricted to the last active cluster
```
(after Hegde et al., Algorithm 1)

Three things worth noticing, because they are what gets asked:

1. **The loop runs at most $2n$ times.** Every iteration either reduces the
   cluster count by one or deactivates a cluster without creating one.
2. **A merged cluster is active again with a fresh moat of zero.** So a node whose
   own prize was long exhausted can start growing again as part of something
   bigger. This is why the analysis is not a simple greedy argument.
3. **A prize-zero node is never active.** Its cluster constraint
   $\sum y \le \pi(C) = 0$ is tight before the clock starts. Such a node can only
   enter the answer by being swallowed in a merge — which is precisely a Steiner
   node.

### Rates, and why speeding this up is hard

Slack on an edge with **two** active endpoints drains at rate $2$; with **one**,
at rate $1$; with none, not at all. Worse, an edge's type changes over time,
because a deactivated cluster becomes active again when it merges. So one cluster
changing state alters the predicted tight-time of many edges.

The fix in papers 4/5 lineage is **dynamic edge splitting** (Cole, Hariharan,
Lewenstein, Porat, SODA 2001): put an inactive sentinel at the midpoint of every
two-active-endpoint edge, so *every* edge has at most one active endpoint and all
live slacks drain at the same rate. Each merge involving a sentinel halves the
remaining distance. Hegde et al.'s contribution is to move sentinels
*adaptively*, which drops the $O(km\log^2 n)$ of Cole et al. to
$O(dm \log n)$ — $O(m\log n)$ at constant precision — and, unlike Cole et al.,
without the extra $\tfrac{2}{nk}$ term in the guarantee.

### Pruning

The growth stage over-collects: $F$ can span nodes that do not pay for themselves.
Two schemes:

- **GW pruning** (paper 2) — depends on the laminar family built during growth;
  removes branches lying in clusters that were deactivated.
- **Strong pruning** (paper 4) — ignores the laminar family and solves PCST
  *exactly* on the growth-stage tree by a linear-time DP. Its ratio is at least as
  good as GW pruning's.

`pcst_fast` exposes `none`, `simple`, `gw`, `strong`. We pass `gw`, because that is
what G-Retriever passes and we reproduce it exactly. **Note for the thesis:
`strong` is documented as never worse, so "why `gw`?" has only a reproduction
answer, not a quality one.**

### Guarantees

GW95 gives the **Lagrangian-preserving** bound

$$c(T) + 2\pi(\bar{T}) \ \le\ 2\,c(T_{\mathrm{OPT}}) + 2\,\pi(\bar{T}_{\mathrm{OPT}})$$

which is stronger than plain factor-2 because of the $2$ on the left. Best known
ratio is **1.9672** (Archer–Bateni–Hajiaghayi–Karloff 2011; Hajiaghayi–Khandekar
2013). Approximating Steiner tree, hence PCST, within **96/95** is NP-hard
(Chlebík–Chlebíková 2002). So the gap to close is narrow, and the factor 2 we
inherit is close to what anyone gets in practice.

---

## 6. Worked example — trace this before the next meeting

$$a(10) \ \overset{4}{-\!-} \ b(0) \ \overset{4}{-\!-} \ c(10) \ \overset{5}{-\!-} \ d(1)$$

Optimum is $\{a,b,c\}$: cost $8$, forfeited prize $1$, objective $9$. Check the
alternatives — take everything, $13$; take only $a$, $11$; take nothing, $21$.

The trace, verified by `experiments/pcst_by_hand.py`:

| time | event |
|---|---|
| $0$ | $\{b\}$ deactivates — prize $0$, cluster constraint tight already |
| $1$ | $\{d\}$ deactivates — its moat reached its prize of $1$ |
| $4$ | $a\!-\!b$ tight, merge $\to \{a,b\}$, fresh moat $0$ |
| $4$ | $b\!-\!c$ tight, merge $\to \{a,b,c\}$ |
| $4$ | $c\!-\!d$ tight, merge $\to \{a,b,c,d\}$ |
| $16$ | the single cluster's constraint goes tight, nothing active, stop |

Why $t=4$ for $c\!-\!d$ although its cost is $5$: both $c$ and $d$ were active
until $t=1$, so slack drained at rate $2$ down to $3$, then at rate $1$.

$F$ then spans **all four nodes** — the growth stage does not exclude $d$.
**Pruning removes $d$**, and the solver returns $\{a,b,c\}$, which is optimal here
even though only a factor 2 is promised. Two lessons: the answer to "how does it
decide what to leave out" is *the pruning stage*, and $b$ is in the answer with
prize $0$ purely because it sits between two valuable nodes.

---

## 7. What our own code does with all this

`src/ikgqa/pcst/core.py`, and `reference.py` beside it is the verbatim published
routine used as a test oracle.

- `root = -1` (unrooted), `num_clusters = 1` (one connected component),
  `pruning = "gw"`, `verbosity = 0`.
- **Edge prizes are not part of PCST.** The solver takes node prizes and edge
  costs only. G-Retriever handles edge prizes in two branches:
  - $p(e) \le c_e$: keep the edge, cost becomes $c_e - p(e) \ge 0$.
  - $p(e) > c_e$: a negative cost is illegal, so delete the edge and insert a
    **virtual node** of prize $p(e) - c_e$ joined to both endpoints by **two
    zero-cost half-edges**.

  Buying the virtual node then nets $p(e) - c_e$, matching the original, and it is
  still reachable only through both endpoints.

  > **Correction to what I had written before:** the half-edges cost **0**, not
  > $c_e/2$. `reference.py:63-64` appends `0` twice. With $c_e/2$ each the net
  > would be $p(e) - 2c_e$ and the transform would not preserve the objective.

- $\gamma = 0.01$ enters as
  $c_e \leftarrow \min\{c_e,\ (\max_e p(e))(1 - \gamma/2)\}$, which guarantees the
  top edge always becomes a virtual node.

---

## 8. The whole-graph result, explained mechanically

This is the thesis's headline finding — PCST with published defaults returned
15,810 of 15,810 nodes — and Chapter 3 currently explains it through the
objective degenerating. The algorithm gives a sharper account. **Verified by
running `pcst_fast` directly, not reasoned about:**

1. Edge prizes are tier-split, $p(e) = b(\tau)/n_\tau$. With few, heavily shared
   relation types, one enormous tier gives every member the same tiny positive
   prize.
2. The $\gamma$ cap drags $c_e$ to just under $\max_e p(e)$, i.e. to the $10^{-6}$
   scale.
3. So nearly every edge takes the *second* branch: it becomes a virtual node with
   **positive** prize and **zero-cost** half-edges.
4. A zero-cost path to a positive prize is always worth taking. The solver takes
   those virtual nodes.
5. Decoding turns each selected virtual node back into an edge — and then
   `decode_solution` **closes the node set under the selected edges**, adding both
   endpoints whatever their prize (`core.py:565-570`).
6. Every node is an endpoint of some such edge. Hence essentially the whole graph.

Two things this corrects. First, **pruning is not the failure point**: on a star
of zero-prize leaves, `gw` pruning correctly returns one node — but it prunes the
solver's *node* set, and the decoder re-adds the endpoints afterwards, so pruning
cannot undo the closure. Second, cost collapse alone does not do it: zero-prize
leaves at cost $10^{-6}$ are still pruned. It is the **virtual nodes plus the
closure**, not the small costs on their own.

Worth raising as a finding rather than defending: the closure step is what makes
the returned subgraph connected and contextual, and it is also what makes the size
dial fail.

---

## 9. Self-test — answer without notes

1. State PCST as a minimisation. Now show it is the thesis's maximisation.
2. Write the dual. Which constraint is a moat not spilling over an edge?
3. Why is a prize-zero node never active, and how does it get into the answer?
4. In the worked example, why does $c\!-\!d$ go tight at $t=4$, not $t=5$?
5. Which stage decides what to leave out?
6. Two active endpoints vs one: what differs, and why does that make a fast
   implementation hard?
7. What do the sentinel nodes achieve?
8. GW pruning vs strong pruning — which is better, and why do we use the other?
9. State the Lagrangian-preserving guarantee. Why is it stronger than factor 2?
10. Edge prizes are not part of PCST. Where do they go, and what does a virtual
    node's half-edge cost?
11. Why unrooted, and what does rooting cost in running time?
12. Trace the whole-graph result from tie tiers to the final node set.

If 1, 3, 5 and 12 are fluent, the rest is recoverable in the room.
