# Progress report — code and results

**Project:** Intent-Aware Subgraph Selection for Clinical Knowledge Graph Question Answering
**Author:** Serxhio Dosku, MSc, University of Basel
**Date:** 31 August 2026 (week 7 of 24)

---

## 1. Summary

The retrieval core is built, tested, and has been run against the live STCS
graph. **274 automated tests pass.** All experimental results so far come from a
synthetic graph that reproduces one real patient's measured structure exactly, so
they describe how the methods behave on this class of graph, not clinical
retrieval quality.

One item needs your attention: **SnapQuery's graph path is not working on the
CHIL server** (Section 5).

---

## 2. What has been built

| Component | Status |
|---|---|
| PCST subgraph selection (G-Retriever) | Verified byte-identical to the published implementation on shared inputs |
| Four baseline strategies | Top-*k* triples (KAPING), top-*k* nodes + neighbours, BFS expansion, shortest paths |
| Neo4j loader for the SPHN graph | Runs read-only against the live STCS graph |
| Answer-node recall metric | Aggregate questions recorded as undefined, not scored zero |
| Terminology cross-walk | Resolves ICD/LOINC/ATC codes to descriptions in five reported tiers |
| Two-stage candidate generation | Makes selection feasible at patient scale |
| Measured-shape replica | Reproduces patient #0 exactly, so work continues without server access |
| SnapQuery client | Complete and tested; cannot yet be run (Section 5) |
| Gold-set format and validation | Ready for questions to be entered |
| Cohort evaluation loop | Resumable, isolates per-patient failures |
| Figures | Four, generated from the result files |

All database access is read-only. No patient data leaves BioMedIT: only schema
names, property names and aggregate counts appear in the thesis.

---

## 3. Tests

`pytest` — **274 tests, all passing**, roughly 12 seconds.

| Area | Tests |
|---|---|
| Gold set and cohort loop | 45 |
| PCST, including equivalence to the published code | 32 |
| Retrievers and evaluation metric | 24 |
| Candidate generation | 19 |
| Terminology resolution | 19 |
| Graph type and encoders | 18 |
| SnapQuery client | 17 |
| Replica fidelity | 16 |
| Neo4j loader | 15 |
| Figures | 13 |
| Baselines | 10 |

The PCST tests include an equivalence check: a verbatim copy of the published
G-Retriever routine is kept as a test oracle, and our implementation must return
the same nodes, edges and serialisation on both curated and randomly generated
graphs. Any difference is a test failure.

---

## 4. Measurements and results

### The graph (read-only queries on CHIL, 19–20 August)

| Quantity | Value |
|---|---|
| Nodes / edges | 21,537,305 / 72,218,890 |
| Patients | 1,197 |
| One reified laboratory construct | 95.3% of all nodes |
| One patient after preparation | 15,810 nodes, 45,562 edges |
| Distinct descriptions for those nodes | 614, about 26 nodes per description |
| Diagnoses carrying no description | 178 of 222 (80%) |

### Experimental results (synthetic replica)

1. **With the published default parameters, PCST returned the whole graph** —
   15,810 of 15,810 nodes. The edge-prize budget is divided among all
   identically-scored edges, and with few heavily shared relation types this
   drives the per-edge cost to a negligible value.

2. **At matched subgraph size, connectivity contributed nothing.** At about
   eleven nodes, PCST, shortest paths, top-*k* nodes with neighbours and KAPING
   all scored exactly **0.164**. What is retrieved is decided by the similarity
   signal, not by the structural strategy.

3. **That holds across patients.** Over ten patients spanning 6,325 to 18,972
   nodes, three strategies agreed to four decimal places: mean **0.1461**,
   standard deviation **0.0081**.

4. **Terminology resolution is decisive.** Diagnoses that carry only a code went
   from **0.00 to 0.80** recall once codes were resolved to descriptions. The
   gain was identical for every retrieval strategy, which places it in graph
   preparation rather than retrieval.

5. **A catalogue in the wrong language is worse than none.** Applying an English
   catalogue to German questions took named diagnoses from **1.00 to 0.00**.
   This is why we need to know what language users actually write in.

6. **Candidate generation is free.** Seed expansion produced the same subgraph
   at the same recall, three to four times faster. But every generator discarded
   every code-only diagnosis, so terminology resolution must run first.

---

## 5. SnapQuery on CHIL — needs your attention

We measured the service on 30 August. It is reachable on port 8002, needs no
authentication from the server, and answers in 6 to 17 seconds per turn. The
request contract is `POST /chat/` with `query` and a client-issued `session_id`,
then `POST /chat/continue` with the session id alone.

**The graph path does not complete a query.** `POST /chat/` returns the planner
model's raw reasoning with an *unexecuted* tool call in the `answer` field, and
`POST /chat/continue` then returns HTTP 500 after about 30 seconds.

The likely cause is a model change. Port 8001 reports
**`Qwen3.8-27B-FP8`**, whereas the architecture document describes the planner as
Qwen2.5-14B-Instruct. Qwen3 emits `<think>` reasoning blocks, which we see in the
output, and its tool calls came back as `<function=...><parameter=...>` rather
than JSON inside `<tool_call>`. A parser written for Qwen2.5 would not read that
format, which would explain both the unexecuted tool call and the 500.

Session id `bd28adf9-125f-4d58-b84c-51c1c8ebe3a1` if the logs help.

Also worth noting: that model reports `max_model_len` of **8,192 tokens**, shared
between the live schema in its system prompt and its own reasoning.

Our client is written and tested against recorded fixtures, so when the service
works only one component changes. Until then the comparison cannot be run.

---

## 6. What we need from you

1. **Is the SnapQuery graph path expected to work on CHIL, and has the planner
   model changed?** This blocks the thesis's main comparison.
2. **Are user questions retained, in what form, and who can export them?** The
   evaluation depends on real questions.
3. **Who can validate roughly 50 gold questions?** The format and its checks are
   ready; this needs a clinician's or data manager's time.
4. **Which labels are in scope?** Our loader currently ignores `Sample` (6.0M
   nodes), `DrugPrescription` (35,784) and demographics. That rules out
   questions about specimens, prescribed-versus-administered, and mortality.
5. **Is `BilledDiagnosis` the right diagnosis label**, or should clinically
   recorded diagnoses be used instead?
6. **May aggregate counts be published?** The thesis reports node counts and
   label frequencies, with no patient-level information. We would like your
   confirmation on record.
