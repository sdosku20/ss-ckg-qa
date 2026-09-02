# Supervision questions on Chapters 1–3

Prepared 2 September 2026. He has read the chapters, so each answer is the number
and where it came from — not a restatement of the text. Every figure is traceable
to `docs/server_measurements.md` or a named script in `experiments/`.

---

## Already asked

### 1. How is recall measured?

$\text{recall} = |A \cap V_S| \,/\, |A|$ — the fraction of gold answer nodes the
retriever returned, always reported next to the size of $V_S$.

- **The 0.80:** ten code-only diagnosis questions, each with $|A| = 1$, so
  per-question recall is 0 or 1. Eight returned their node, two did not. Before
  terminology resolution, none did.
- **Why not 1.00:** the synthetic catalogue covers 85% of codes by design.
- **If he asks why not precision or F1:** precision is the size axis of every
  plot, so the trade-off stays visible instead of collapsing into one number.

### 2. What connectivity constraint?

Every returned node must be reachable from every other using only returned edges
— a restriction on the feasible set, not a term in the objective.

- **Why it should matter:** it admits Steiner nodes (prize zero, but on a path
  between two valuable nodes), which no independently-scoring method can see.
- **What RQ2 measured:** nothing. Four strategies scored exactly 0.164 at eleven
  nodes; three agreed at 0.1461 (SD 0.0081) across ten patients. Prediction 3
  says why — 168 provenance hubs of average degree 69,525 make connectivity
  nearly free to satisfy.

---

## The next question he will ask

### 3. 0.164 is very low. Is the method failing?

No — it averages answer sets spanning three orders of magnitude:

| Question kind | Answer nodes | Recall |
|---|---|---|
| Named diagnosis (×2) | 1 | **1.000** |
| Code-only diagnosis (×2) | 1 | **0.000** |
| Drug (×5) | 10–60 | 0.017–0.093 |
| Analyte (×5) | 1,000–2,407 | 0.0004–0.0016 |

An analyte question wants every measurement of one substance: 2,407 answer nodes
against 11 retrieved, so the **ceiling is $11/2407 = 0.0046$**. Those questions
measure the budget-to-answer-size ratio, not retrieval quality.

**Concede before he does:** averaging $|A|=1$ with $|A|=2407$ is not a meaningful
number. What is meaningful is that *every strategy produced the identical 0.164*.
The analyte questions are really aggregations, which the metric section already
says recall cannot score, so they should be split out by answer-set size or
reclassified. Raise it as an open methodological point.

---

## Chapter 1

**4. Where do 21,537,305 and 15,810 come from?**
Read-only schema survey on CHIL, 19 August (`docs/recon.txt`); the per-patient
figure from `python -m ikgqa.data.sphn --patient 0`, 20 August. Every number in
the thesis has its command and date beside it.

**5. What is an "answer node" for a lab question versus a diagnosis?**
`BilledDiagnosis` for a diagnosis, `Drug` or its administration events for a
drug, every `LabObservation` of the analyte for a lab question. The loader
collapses SPHN's four-node laboratory construct into one node per result.

**6. Why exclude query latency?**
It measures an implementation and a deployment, not a selection strategy.
Wall-clock time is reported where it affects feasibility (candidate generation is
3–4× faster) but never in the comparison metric.

**7. Why does the 8,192-token context matter if you don't use that model?**
It is the context of the system we compare against, so it is the budget the
comparison actually runs under. One prepared patient serialises to roughly
$4.5\times10^5$ tokens — under one per cent fits. Caveat: the
ten-tokens-per-triple figure is an estimate; port 8001's `/tokenize` can make it
exact.

**8. Does graph preparation bias the comparison in your favour?**
Every strategy gets the identical prepared graph, each of the five steps is
switchable, and the unprepared graph is the baseline. The one care point: a
text-scoring baseline must be scored on embedding text, not display text — a test
pins that.

**9. How do you know the implementation reproduces the published one?**
A byte-identical copy of the published routine is kept as a test oracle; ours must
return the same nodes, edges and serialisation on curated and random graphs. 32
tests, and any divergence fails.

**10. Is it acceptable that RQ3 is conditional?**
It is scoped optional in the preparation document, and RQ1–RQ2 stand alone. Worth
volunteering: the negative RQ2 result makes RQ3 *more* interesting — if the
similarity signal decides everything, the prize function is where to intervene.

---

## Chapter 2

**11. Why is PCST the only family with both connectivity and size control?**
Top-$k$ has size control without connectivity; expansion and paths have
connectivity but hops are integers — BFS jumps from 0.05% to 97% of the graph
between one and two hops. PCST prices both in one objective. **But** Chapter 6
shows the dial does not work here: the full $c_e$ range moves the result only
8 → 11 nodes. True of the formulation, false of this instance, and the thesis
says so.

**12. Sentence encoders are multilingual now. Why the cross-walk?**
G-Retriever uses `all-roberta-large-v1`, which is English-only; 80% of one
patient's diagnoses carry no description at all, so there is nothing to embed in
any language; and a cross-walk is deterministic and auditable. Fair to grant that
a multilingual encoder is the right comparison — it would separate "wrong
language" from "absent label", and only the cross-walk addresses the second.

**13. Is comparing against SnapQuery fair?**
It compares a schema-reading query generator against a text-similarity retriever
on the shared metric — what fraction of the answer entities each puts in front of
the model. Two limits stated up front: SnapQuery has no size parameter, so one
point rather than a curve; and treating its result rows as a subgraph is our
interpretation, not its claim.

**14. Why not Think-on-Graph or LLM-guided traversal?**
Several LLM calls per question against one encoding pass, so not comparable on
cost, and it confounds retrieval with the driving model's quality. Listed as a
family, named as excluded.

---

## Chapter 3

**15. Why rank prizes rather than similarity values?**
Rank makes the prize scale independent of how tightly similarities cluster. The
cost is that magnitude is discarded — best and second-best always differ by one
unit — which matters here because many nodes are genuinely tied.

**16. Walk me through the virtual-node transformation.**
Solvers take node prizes and edge costs, not edge prizes. If $p(e) \le c_e$,
absorb it: cost becomes $c_e - p(e)$. If $p(e) > c_e$, that needs a negative
cost, so insert a virtual node with prize $p(e) - c_e$ and split the edge into
two of cost $c_e/2$. Both branches preserve the objective; the price is remapping
virtual nodes back to edges when decoding.

**17. Where does $\gamma = 0.01$ come from?**
The published implementation, not us. It caps per-edge cost just below the largest
edge prize. Chapter 3's point is that it reads as a safety rail but acts as a
coupling: cost can never exceed $\max_e p(e)$, so a tiny maximum collapses the
size dial whatever value you ask for.

**18. Talk me through the $10^{-6}$ in Prediction 1.**
Edge text is the relation type, so every use of a relation ties, and the tier
budget splits among them: $p(e) = b/n_\tau$. `hasCode` occurs 4,883,809 times, so
$p(e) \lesssim 10^{-6}k_e$; Eq. 3.8 drags $c_e$ below that, every edge becomes
almost free, and nothing reachable is worth excluding. Predicted whole graph,
measured 15,810 of 15,810.

**19. If PCST returns the whole graph, isn't your implementation broken?**
That was the first hypothesis and it was tested: the equivalence oracle shows
ours and the published code agree exactly, so both return everything on this
input. Chapter 3 derives it before Chapter 6 measures it.

**20. PCST is NP-hard. What does the approximation give you?**
Goemans–Williamson primal-dual in Hegde et al.'s near-linear formulation, via
`pcst_fast` — a constant-factor approximation, run unrooted with GW pruning.
Unrooted because rooting means naming a node the tree must contain, which assumes
where the answer is.

**21. Prediction 2 says two correct implementations disagree. Is that a bug?**
No — identical text gives exactly equal similarity, so rank is decided by the
sort's tie order. It is a property of the input: 4,832,744 lab tests resolve to
544 distinct descriptions; one patient's 15,810 nodes carry 614 texts, about 26
per text. Our runs pin the tie-break explicitly.

**22. Prediction 3 is about hubs, but you prune them.**
The prediction explains why the unqualified connectivity requirement is nearly
vacuous on the raw graph, which is what justifies pruning as method rather than
tidying. And pruning does not rescue connectivity: every strategy still scored
identically at matched size, for Prediction 2's separate reason.

---

## Volunteer these rather than be caught on them

1. **All experimental numbers come from the synthetic replica** — patient #0's
   structure exactly on nine of ten counts, invented content. Valid for
   mechanism, not for clinical retrieval quality.
2. **No real embeddings yet** — a deterministic bag-of-words encoder throughout,
   so absolute recall values are not clinical numbers.
3. **The SnapQuery comparison has not run.** Its graph path is broken on CHIL
   (deployed planner is Qwen3.8-27B-FP8, not the documented Qwen2.5); our client
   is written and tested against fixtures.
4. **The wrong-language result (1.00 → 0.00) is overstated by the fixture** — the
   synthetic German and English vocabularies are token-disjoint by construction;
   real clinical text shares cognates.
5. **The per-analyte distribution is a Zipf assumption**, not a measurement. Only
   the mean, 62 measurements per analyte, was measured.
