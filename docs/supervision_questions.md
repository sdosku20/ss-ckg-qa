# Likely supervision questions on Chapters 1–3, with answers

Prepared 2 September 2026. Every number here is traceable to
`docs/server_measurements.md` or to a named script in `experiments/`.

Read the derivations, not just the numbers. The follow-up question is always
"where did that come from", and the derivation is the answer to both.

---

## The two questions already asked

### 1. How is recall measured in our case?

**Answer-node recall.** For one question we fix a set of gold answer nodes $A$:
the graph nodes that *are* the answer. The retriever returns a node set $V_S$.
Then

$$\text{recall} = \frac{|A \cap V_S|}{|A|}$$

It is always reported together with the size of $V_S$, because returning the
whole graph would score 1.0 trivially.

**For the 0.80 figure specifically.** Those are code-only diagnosis questions.
Each asks about one diagnosis, so $|A| = 1$ and per-question recall is binary:
either that one node came back or it did not. Ten such questions were run; eight
returned their answer node and two did not, giving a mean of 0.80. Before
terminology resolution all ten returned nothing, hence 0.00.

**Why not 1.00.** The synthetic catalogue used in that experiment deliberately
covers only 85% of the codes, so a minority of diagnoses stay unresolvable. That
is a property of the fixture, not a limit of the method.

**If asked why recall and not precision or F1:** precision is captured by the
size axis instead. A retriever with high recall and terrible precision returns a
large subgraph, and that shows up directly on the x axis of every plot. Reporting
recall against measured size makes the trade-off visible rather than collapsing
it into one number that hides which side it came from.

---

### 2. What connectivity constraint do we have?

**The returned subgraph must be connected**: every returned node reachable from
every other using only returned edges. In PCST it is a constraint on the feasible
set, not a term in the objective (Chapter 3, after Eq. 3.3), so no node however
relevant can buy its way in while disconnected.

**Why that could help.** It admits *Steiner nodes*: a node with prize zero is
worth including if it lies on a path between two nodes that jointly forfeit more
than the connecting edges cost. Such a node is invisible to any method scoring
nodes independently. Clinically that is the bridging entity — a case or an
encounter that links a drug to a diagnosis — which is exactly the kind of thing a
top-$k$ scorer cannot see.

**What RQ2 does with it.** RQ2 asks how much of any difference comes from the
connectivity constraint rather than from the similarity scores. The ablations
hold the encoder, the graph and the question embedding fixed and vary only how
connectivity is handled:

| Strategy | Connectivity |
|---|---|
| Top-$k$ triples (KAPING) | none; independently scored, can be disconnected |
| Top-$k$ nodes + neighbours | connected by construction |
| BFS expansion from seeds | connected by construction |
| Shortest paths | connected by construction |
| PCST | connected, and priced against size in one objective |

**The measured answer, and it is negative.** At about eleven nodes PCST, shortest
paths, top-$k$ nodes with neighbours and KAPING all scored exactly 0.164. Over
ten patients of differing size, three strategies agreed to four decimal places
(0.1461, SD 0.0081). On this graph the connectivity constraint bought nothing;
the similarity signal decided the outcome. Chapter 3 Prediction 3 says why: 168
provenance nodes of average degree 69,525 put almost any two nodes within two or
three hops, so connectivity is satisfiable at negligible cost.

---

## The question he is most likely to ask next

### 3. 0.164 recall is very low. Is the method failing?

**No, and this is the most important thing to be able to explain.** 0.164 is a
mean over fourteen questions whose answer-set sizes differ by three orders of
magnitude. The per-question values at 11 retrieved nodes:

| Question kind | Answer nodes $|A|$ | Recall | Why |
|---|---|---|---|
| Named diagnosis (×2) | 1 | **1.000** | one node, retrievable by its words |
| Code-only diagnosis (×2) | 1 | **0.000** | no words at all to match |
| Drug (×5) | ~10–60 | 0.017–0.093 | partial by arithmetic |
| Analyte (×5) | 1,000–2,407 | 0.0004–0.0016 | **arithmetically capped** |

An analyte question asks for every measurement of one substance. With 2,407
answer nodes and 11 retrieved, the *maximum possible* recall is
$11/2407 = 0.0046$. No retriever could do better at that size. So the analyte
questions are not measuring retrieval quality; they are measuring the ratio of
budget to answer size.

**The honest concession, and make it before he does.** Averaging over questions
with $|A| = 1$ and $|A| = 2407$ mixes incommensurable things, and 0.164 is
therefore not a meaningful single number. What is meaningful is that *every
strategy produced the identical 0.164*, which is a statement about the strategies
being indistinguishable, not about their quality.

**What follows.** Those analyte questions are really aggregations ("all
creatinine values"), which the metric section already says recall cannot score
properly. Either they get reported separately by answer-set size, or they are
reclassified as aggregate questions. This is a real methodological point to raise
with him rather than defend.

---

## Chapter 1

### 4. Where do 21,537,305 and 15,810 come from?

The graph-wide figures come from a read-only schema survey run on the CHIL server
on 19 August (`~/recon.py`, output committed as `docs/recon.txt`). The
per-patient figures come from running the loader on patient #0 on 20 August:
`python -m ikgqa.data.sphn --patient 0`. Both are in
`docs/server_measurements.md` with the command and date beside them. Nothing goes
into the thesis without a command that reproduces it.

### 5. What is an "answer node" for a lab question versus a diagnosis?

For a diagnosis question it is the `BilledDiagnosis` node. For a drug question it
is the `Drug` node, or the administration events, depending on how the question
is phrased. For an analyte question it is every `LabObservation` of that analyte.
The loader collapses SPHN's four-node laboratory construct into one
`LabObservation` node per result, so "the answer" is one node per measurement
rather than four.

### 6. Why exclude query latency?

It is a property of an implementation and a deployment, not of a selection
strategy. Including it would reward engineering effort rather than retrieval
behaviour. Wall-clock time *is* recorded where it affects feasibility — candidate
generation is reported as three to four times faster — but it is never part of the
comparison metric.

### 7. Why does the 8,192-token context matter if you are not using that model?

Because it is the context of the model behind the system we compare against, so
it is the budget the comparison actually operates under. It also converts the
context argument from a general claim about language models into a measured
constraint: one prepared patient serialises to roughly $4.5\times10^5$ tokens, so
under one per cent of a single patient fits. Caveat to state: the ten-tokens-per
-triple figure is an estimate. Port 8001 exposes a `/tokenize` endpoint, so it can
be measured exactly, and that is on the list.

### 8. Does "graph preparation" bias the comparison in your favour?

Every strategy receives the identical prepared graph, so no strategy gains from
it relative to another. Each of the five steps is independently switchable and
the unprepared graph is the baseline every gain is measured against. The one
choice needing care is that a text-scoring baseline must be given the *embedding*
text and not the display text — scoring KAPING on display text while scoring PCST
on embedding text would understate it, and a test pins that.

### 9. Contributions claims the implementation reproduces the published one exactly. How?

A verbatim copy of the published G-Retriever routine is kept in the package,
confirmed byte-identical to upstream, and never edited. It is used as a test
oracle: our instrumented implementation must return the same node set, edge set
and serialised output on curated graphs chosen to hit each branch, and on
randomly generated graphs. Any divergence is a test failure. 32 tests cover PCST,
including that equivalence.

### 10. Is it acceptable that RQ3 happens "only if on schedule"?

It is scoped as an optional extension in the preparation document, and RQ1 and
RQ2 stand alone as a contribution. Worth saying: the negative result on RQ2 makes
RQ3 more interesting, not less. If the similarity signal rather than the
structure decides the outcome, then improving the prize function is the
indicated response, which is precisely what RQ3 proposes.

---

## Chapter 2

### 11. Why is PCST the only family with both connectivity and size control?

Top-$k$ has exact size control via $k$ but no connectivity guarantee. Expansion
and path-based methods guarantee connectivity but lose size control: hops are
integers, and on a dense graph the neighbourhood grows by orders of magnitude per
hop. Measured here: BFS goes from 0.05% to 97% of the graph between one and two
hops, with nothing available in between. PCST prices connectivity and size in one
objective, so the cost per edge is a continuous dial.

**But** Chapter 6 shows the continuous dial does not work on this graph: sweeping
$c_e$ across its full published range moved the result only between 8 and 11
nodes, because of the coupling in Eq. 3.8. So the table's claim is true of the
formulation and false of this instance, and the thesis says so.

### 12. Sentence encoders are multilingual now. Why the terminology cross-walk?

Three reasons. G-Retriever specifically uses `all-roberta-large-v1`, which is
English-only, so reproducing the published method faithfully means inheriting
that. More fundamentally, 80% of one patient's diagnoses carry **no description
at all** — only a code — so there is nothing for any encoder, multilingual or
not, to embed. And a cross-walk is deterministic and auditable, which matters
clinically in a way that a learned multilingual embedding does not.

A multilingual encoder is the right comparison to run, and it is fair to say so:
it would separate "the labels are in the wrong language" from "the labels are
absent". The second is the larger effect and only the cross-walk addresses it.

### 13. Is comparing against SnapQuery fair, given it works differently?

It is a comparison between a schema-reading query generator and a text-similarity
retriever, not between two variants of one mechanism, and Chapter 2 states that
explicitly. The comparison is on the shared metric: what fraction of the answer
entities each ends up putting in front of the model. Two limitations are stated
rather than discovered later: SnapQuery exposes no size parameter so it gives a
single point rather than a curve, and treating the entities its rows touch as its
subgraph is an interpretation this thesis imposes, not something SnapQuery
asserts.

### 14. Why not compare against Think-on-Graph or an LLM-guided traversal?

It pays several LLM calls per question where the others pay one encoding pass, so
it is not comparable on cost, and it would confound the retrieval question with
the quality of whichever model drives the traversal. It is in the table as a
family, and named as excluded.

---

## Chapter 3

### 15. Why prizes by rank rather than by similarity value?

Rank makes the prize scale independent of how tightly the similarity scores are
clustered, which is a sensible defence against an encoder whose scores sit in a
narrow band. The cost is that magnitude is discarded entirely: the gap between
the best and second-best node is always exactly one prize unit whether the
encoder thought them near-identical or wildly different. On this graph that
matters, because many nodes are genuinely tied.

### 16. Walk me through the virtual-node transformation.

PCST places prizes on nodes and costs on edges, but G-Retriever also wants prizes
on edges, which no solver accepts. Two cases. If $p(e) \le c_e$ the prize is
absorbed into the cost: keep the edge with cost $c_e - p(e) \ge 0$. If
$p(e) > c_e$ that would need a negative cost, so a virtual node carrying prize
$p(e) - c_e$ is inserted and the edge becomes two edges of cost $c_e/2$ each.
Selecting both costs $c_e$ and collects $p(e) - c_e$, a net gain of $p(e) - c_e$,
matching the original. The transformation is objective-preserving in both
branches. The price is paid at decoding: virtual nodes in the solution must be
mapped back to the edges they stand for.

### 17. Where does $\gamma = 0.01$ come from?

It is the published implementation's constant, not ours. It caps the per-edge cost
just below the largest edge prize so the second branch of the transformation
stays reachable and edge prizes keep influencing the solution. The point
Chapter 3 makes is that this reads as a safety rail but acts as a coupling: the
effective cost can never exceed $\max_e p(e)$, so if that maximum is tiny the
size dial collapses regardless of the value requested.

### 18. Talk me through the $10^{-6}$ in Prediction 1.

Edge text comes from the relation type, so every use of a relation gets identical
similarity and all of them land in one tie tier. The tier's budget is divided
among its members, so $p(e) = b/n_\tau$. The relation `hasCode` is used
4,883,809 times in the measured graph, and $b \le k_e$, so
$p(e) \le k_e / 4{,}883{,}809 \approx 10^{-6} k_e$. Eq. 3.8 then drags $c_e$ down
to just below that. Every edge becomes almost free, the cost term becomes
negligible beside the prize term, and nothing reachable is worth excluding.
Predicted: the method returns essentially its whole input. Measured: 15,810 of
15,810 nodes.

### 19. If PCST returns the entire graph, isn't your implementation broken?

That was the first hypothesis and it was tested rather than assumed. The
equivalence oracle shows our implementation and the published one agree exactly
on shared inputs, so both return the whole graph on this input. The behaviour
follows from the published parameter defaults meeting a graph with few, heavily
shared relation types. Chapter 3 derives it before Chapter 6 measures it, which
is what distinguishes a prediction from an excuse.

### 20. PCST is NP-hard. What does the approximation give you?

We use the Goemans–Williamson primal-dual algorithm in the near-linear-time
formulation of Hegde et al., via `pcst_fast`. It is a constant-factor
approximation, not exact. It is invoked unrooted, for a single connected
component, with GW pruning. Unrooted is the right choice because no node is
distinguished in advance; a rooted formulation would require naming a node the
tree must contain, which amounts to assuming where the answer is.

### 21. Prediction 2 says two correct implementations return different subgraphs. Is that a bug?

Not in either implementation. When many nodes share identical text their
similarities are exactly equal, so the rank $\pi(v)$ is decided by whatever order
the sorting routine produces among ties. Two implementations sorting differently
are both correct and disagree. It is a property of the input: 4,832,744
laboratory tests resolve to 544 distinct descriptions, and one patient's 15,810
nodes carry 614 texts, about 26 nodes per text. Our runs pin the tie-break
explicitly so results are reproducible, and the effect is reported rather than
hidden.

### 22. Prediction 3 concerns hubs, but you prune them. So why does it matter?

Two reasons. The prediction is about the raw graph and explains why the
unqualified connectivity requirement is nearly vacuous there, which justifies
pruning as a methodological choice rather than tidying. And pruning does not
rescue connectivity's value: even after it, every strategy scored identically at
matched size. The prediction was that connectivity contributes little until such
nodes are excluded; what was measured is that it contributes little afterwards
too, for the separate reason in Prediction 2.

---

## Things to volunteer rather than be caught on

1. **Every experimental number is from the synthetic replica**, which reproduces
   patient #0's structure exactly on nine of ten measured counts but has invented
   content. Valid for mechanism, invalid for clinical retrieval quality.
2. **No real embeddings have been run.** All experiments used a deterministic
   bag-of-words encoder. Absolute recall values are not clinical numbers.
3. **The SnapQuery comparison has not run.** Its graph path is broken on CHIL
   (the deployed planner is Qwen3.8-27B-FP8, not the documented Qwen2.5), so the
   client is written and tested against fixtures but the comparison is pending.
4. **The wrong-language result (1.00 to 0.00) is overstated by the fixture.** The
   synthetic German and English vocabularies are deliberately token-disjoint. Real
   clinical German and English share cognates, so the real effect would be
   milder.
5. **The per-analyte distribution is a Zipf assumption**, not a measurement. Only
   the mean, 62 measurements per analyte, was measured.
