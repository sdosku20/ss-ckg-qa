# Server measurements

Everything measured on the CHIL server, kept here so it never has to be measured
twice and so the thesis has one source of truth for its numbers.

**Nothing on this page is patient data.** Every entry is a count, a schema name,
a property name, or an aggregate. Property *values*, identifiers, results and
dates never appear here and must never be added.

Each block records **what was run**, **when**, and **what came back**. If a
number is not on this page with a date next to it, it has not been measured, and
it must not appear in the thesis.

---

## 1. Environment

Measured 18–19 August 2026.

| Thing | Value |
|---|---|
| Host | `chil.scicoreplus.unibas.ch`, hostname `sphn-cohort-webserver-chil` |
| Login | BioMedIT SSO device flow, username `sdosku`. Not SSH keys |
| Neo4j | 5.20.0 Community, **read-only**, `bolt://localhost:7687`, HTTP `:7474`, db `neo4j` |
| CPU / RAM | 64 cores, 240 GB |
| GPU | 1× NVIDIA A100 |
| Python | 3.10 (server) vs 3.12 (laptop) — do not ship `.pyc` between them |
| git | 2.34.1 |
| Network | PyPI reachable. Arbitrary HTTP **not yet tested** — see §6 |

---

## 2. Whole graph

Measured 19 August 2026 via `~/recon.py`. Full output copied locally
20 August 2026 and committed as [`docs/recon.txt`](recon.txt) — read that for the
complete label, relation, property and degree tables. The derived facts below are
the ones that drive design decisions.

### 95.3% of the graph is one construct

| Label | Nodes |
|---|---|
| `Sample` | 6,025,659 |
| `LabTest` | 4,832,744 |
| `LabResult` | 4,832,744 |
| `LabTestEvent` | 4,832,744 |
| **subtotal** | **20,523,891 of 21,537,305 = 95.3%** |

SPHN represents one laboratory result as four event nodes plus shared
`Quantity`, `Unit`, `Code`, `ReferenceRange` and `BodySite` nodes. The loader
emits **one** `LabObservation` node for the whole construct, so graph preparation
is roughly a 4× node reduction on 95% of the graph before any pruning of
provenance. For patient #0 that is ≈49,400 raw event nodes reduced to 12,349.

### Three relations carry half the edges

| Relation | Uses | Target |
|---|---|---|
| `hasSourceSystem` | 11,680,076 | 168 `SourceSystem` nodes |
| `hasSubjectPseudoIdentifier` | 11,600,854 | 1,197 patient nodes |
| `hasAdministrativeCase` | 11,566,354 | 37,513 case nodes |
| `hasCode` | 4,883,809 | 10,014 `Code` nodes |

35M of 72M edges are those first three. Pruning provenance removes 11.7M edges on
its own.

### Degree, from the recon output

| Label | Nodes | Max degree | Avg degree |
|---|---|---|---|
| `SourceSystem` | 168 | **10,858,404** | 69,525 |
| `BodySite` | 6 | 2,831,985 | 575,814 |
| `TimePattern` | 2 | 557,488 | 313,012 |
| `Code` | 10,014 | 1,226,895 | 833 |
| `ReferenceRange` | 718 | 147,870 | 6,113 |
| `SubjectPseudoIdentifier` | 1,197 | 74,275 | 9,693 |

One `SourceSystem` node carries 10.86M edges — 15% of every edge in the graph on
a single node. Every label in this table is either pruned or folded by the
loader.

### What the loader does not represent — decisions to confirm

Recon revealed labels the loader ignores. Each is a deliberate scope choice, but
none has been agreed with the data owners:

| Label | Nodes | Consequence of omitting it |
|---|---|---|
| `Sample` | 6,025,659 | no question about specimens or collection time |
| `DrugPrescription` | 35,784 | cannot distinguish prescribed from administered |
| `Birth`, `BirthDate` | 1,214 / 1,174 | no age-based questions |
| `Death`, `DeathDate` | 241 / 229 | no outcome/mortality questions |
| `AdministrativeSex` | 1,214 | no sex-stratified questions |
| `Consent` | 1,214 | correctly out of scope |

Prescription-versus-administration and mortality are plausible clinical
questions, so this list has to be settled before the gold set is written, not
after.

| Quantity | Value |
|---|---|
| Nodes | 21,537,305 |
| Edges | 72,218,890 |
| Patients (`SubjectPseudoIdentifier`) | 1,197 |
| Nodes per patient, graph-wide average | ≈ 17,993 |

### Degenerate hubs

| Label | Count | Note |
|---|---|---|
| `SourceSystem` | 168 nodes | average degree **69,525** |
| `BodySite` (anatomical sites) | 6 nodes | absorb **3,454,881** edges between them |

These are why an unqualified connectivity requirement is nearly vacuous: almost
any two nodes are two or three hops apart through one of them. They are also
correct and required by the schema, which is why they are pruned in graph
preparation rather than reported as a data problem.

### Relation multiplicity

| Relation | Uses |
|---|---|
| `hasCode` | 4,883,809 |

This single number is the arithmetic behind finding F2: G-Retriever divides each
edge-prize tier among all edges tied in it, so a relation used 4.9 million times
receives a per-edge prize of order `k / 4.9e6 ≈ 1e-6`, and `adjust_edge_cost`
then lowers the per-edge cost to just under that.

---

## 3. Terminologies

Measured 19 August 2026 with restricted `MATCH` counts per label.

| Terminology | Distinct codes | Description coverage | Language |
|---|---|---|---|
| LOINC (laboratory) | 544 | 544 / 544 via `hasLongName` (100%) | English |
| ICD-10-GM (diagnoses) | 2,921 | 972 with a long name (33%) | German |
| ATC (drugs) | 737 | not yet measured | mixed |
| UCUM (units) | 70 | not yet measured | n/a |
| SNOMED CT | 69 | not yet measured | English |

Further measured facts:

- 4,832,744 laboratory test uses resolve to those **544** distinct descriptions.
  A similarity score over that text can take 544 distinct values across millions
  of nodes.
- ICD-10-GM codes are versioned by year, `10-GM-2012` through `10-GM-2024`,
  summing to exactly 2,921. The same code string can mean different things in
  different versions, so the version must travel with the code.
- All 46,143 diagnosis uses are ICD-10 shaped, so a public catalogue can fill
  the missing descriptions.
- Enrichment gap: **2,921 − 972 = 1,949** diagnosis codes with no description in
  the graph.
- `BilledDiagnosis` → `Code` property counts came back as
  `NULL | 2921 | 46143 | 0`, i.e. every use carries a code and none carries a
  free-text diagnosis name of its own.

---

## 4. One patient, end to end

Measured 20 August 2026:

```bash
# SERVER
source ~/.ikgqa.env
~/venvs/ikgqa/bin/python -m ikgqa.data.sphn --patient 0
```

| Quantity | Patient #0 |
|---|---|
| Nodes after graph preparation | 15,810 |
| Edges after graph preparation | 45,562 |
| `LabObservation` | 12,349 |
| `LabTestType` (distinct analytes) | 198 |
| `DrugAdministrationEvent` | 2,655 |
| `Drug` (distinct) | 279 |
| `BilledDiagnosis` | 222 |
| `AdministrativeCase` | 106 |
| `SubjectPseudoIdentifier` | 1 |
| Distinct `embed_text` values | **614** |
| Diagnoses with no description | **178 of 222 (80%)** |
| Lab tests with no description | 0 of 12,349 |
| Bag-of-words vocabulary over those texts | 497 dimensions |

Derived, and worth stating in the thesis:

- **25.7 nodes per distinct text.** Similarity can sort this graph into at most
  614 classes. Within a class the ordering is decided by iteration order.
- **62 measurements per analyte** on average, heavily skewed.
- **11.5× a whole WebQSP sample** (1,371 nodes average, per question) — for one
  patient out of 1,197.
- The graph is not a star: `AdministrativeCase`, `LabTestType` and `Drug` are
  shared, so there is genuine structure for a connectivity-aware method to use.

First retrieval on real clinical data, same session:

```bash
# SERVER
~/venvs/ikgqa/bin/python -m ikgqa.data.sphn --patient 0 --retrieve "creatinine kidney function"
```

Returned 3 nodes and 2 edges (2 `LabObservation`, 1 `LabTestType`) out of 15,810
nodes. The size dial demonstrably works at this setting. The encoder was the
bag-of-words placeholder, so this says nothing about retrieval *quality*.

---

## 5. What the replica reproduces

`ikgqa.data.replica` generates a synthetic patient whose structure matches the
table above. Achieved on 20 August 2026 with `python -m ikgqa.data.replica`:

| Quantity | Measured | Replica |
|---|---|---|
| nodes | 15,810 | 15,810 |
| edges | 45,562 | 45,562 |
| lab observations | 12,349 | 12,349 |
| analytes | 198 | 198 |
| diagnoses | 222 | 222 |
| unnamed diagnoses | 178 | 178 |
| drug administrations | 2,655 | 2,655 |
| drugs | 279 | 279 |
| cases | 106 | 106 |
| distinct embed texts | 614 | 680 (+66) |

Exact on every structural count. The text count is 11% high because the real data
shares slightly more text than the generator does; the gap is printed on every
run rather than hidden.

---

## 6. Not yet measured — do these next time there is server access

In priority order. Each is one command and answers something currently guessed.

Two of the original six are now done:

- ~~Copy `~/recon.txt` locally~~ — done 20 August 2026, see §2.
- ~~Can the server reach arbitrary HTTP?~~ — **yes**, `https://example.org`
  returned `200` on 20 August 2026. Terminology catalogues can therefore be
  downloaded on the server, and do not have to be `scp`-ed in.
- Raw-versus-prepared node counts are now **derivable** from the label table in
  §2 rather than needing a query, though a direct per-patient count would still
  be a cleaner number to quote.

Still outstanding:

1. **Distinct embed texts per label**, to explain the 614 and identify which
   label contributes most of the tie-breaking problem. Currently inferred.
2. **Patient size distribution.** Patient #0 may not be typical and the whole
   evaluation is per patient: minimum, median and maximum node counts across the
   1,197. This decides whether one graph per question is even feasible.
3. **ATC, UCUM and SNOMED description coverage**, to complete §3.
4. **What does one full SnapQuery exchange look like end to end?** The service is
   confirmed reachable (§7); what remains is a complete recorded exchange, from
   question through slot filling and confirmation to executed query and rows.
5. **Confirm the loader's label coverage** (the omissions table in §2) with the
   data owners.

---

## 7. SnapQuery, first contact — 30 August 2026

The port and endpoints had until now been copied from an architecture document
and never verified. They are correct.

```bash
# SERVER
python3 ~/snapquery_probe.py --discover
```

| Path | Status | Reading |
|---|---|---|
| `/` | 404 | no root route; the service is mounted at its endpoints only |
| `/docs` | 200 | Swagger UI, so this is FastAPI |
| `/openapi.json` | 200 | machine-readable contract available |
| `/health`, `/healthz` | 404 | no health endpoint under either common name |
| `/chat/` | 405 | route exists, GET not allowed — POST only |
| `/chat/continue` | 405 | same |
| `/api/chat/`, `/v1/chat/` | 404 | not versioned or prefixed |

Two endpoints are declared, matching the architecture document exactly:

```
POST /chat/           Start or continue chat with snapquer
POST /chat/continue   Continue snapquer's reasoning without new input
```

The distinction between them matters for the harness: `/chat/` carries new user
input, `/chat/continue` advances the agent's own reasoning **without** any. That
implies the LangGraph agent can yield control mid-run and be resumed, which is
what a confirmation gate looks like from outside.

Nothing here says whether authentication is required: a 405 is decided at
routing, before any auth dependency runs, and `/docs` being open says only that
the docs are open. The first POST settles it.

Also confirmed on the same session: the listening ports are 8001, 8002, 8080,
5432 (PostgreSQL, the tabular path), 6379 (Redis, plausibly the session store),
7474/7475/7476 and 7687/7688/7689 — three Neo4j instances on consecutive ports,
consistent with the three centers (Basel, Bern, Lausanne) described in §2.7.4 of
the thesis. Only the Basel graph is in scope here.

`/home/gretriver_master_thesis/README.md` covers Neo4j access only and says
nothing about SnapQuery.

### The request contract, measured

A first POST with a guessed field name returned `422`, and FastAPI's validation
error named the fields it wanted. `POST /chat/` requires exactly two:

| Field | Required | Note |
|---|---|---|
| `query` | yes | the user's question. **Not** `message` |
| `session_id` | yes | **client-issued** -- the caller invents it |

That `session_id` is required rather than server-issued is the most useful thing
learned so far. There is no handshake and no handle to scrape out of a response:
the client mints an identifier, the server keys its state off it, and an
evaluation loop can therefore start a clean session per question and be certain
no state leaks between them. Redis on 6379 is the plausible store.

`message` is not a field at all. It appeared in the `422` body only because
FastAPI echoes the whole rejected payload back under `input`.

```bash
# SERVER -- one turn, recorded and redacted
SESSION=$(python3 -c 'import uuid; print(uuid.uuid4())')
python3 ~/snapquery_probe.py --field query --session "$SESSION" --ask "..."
```

### The response contract, measured

First accepted POST, 30 August 2026. No token was sent and the reply was `200`,
so **the service requires no authentication** from the server itself.

| Field | Type on the first turn | Note |
|---|---|---|
| `answer` | string, 975-2986 chars | the model's text |
| `data` | `null` | result rows, when a query has run |
| `cypher` | `null` | the generated query, when one has been written |
| `status` | `"ok"` | |

Turn latency: 5.8 s, 8.3 s, 10.4 s, 16.7 s across four turns. A 50-question gold
set at three turns each is therefore around 20 minutes of wall clock, so the
evaluation loop is comfortably feasible.

### `/chat/` returns an unexecuted tool call; `/chat/continue` is not optional

The `answer` on the first turn contained the planner model's raw chain of thought
followed by a verbatim, unexecuted tool call in Qwen/Hermes format:

```
</think>
<tool_call>
<function=snapquery_schema_lookup>
<parameter=user_request>...</parameter>
</function>
</tool_call>
```

So `POST /chat/` runs the agent only until it wants a tool, then returns. The
client is expected to call `POST /chat/continue` -- "continue snapquer's
reasoning without new input" -- to execute it and advance. Sending further user
text to `/chat/` instead interrupts the tool call and produces more of the same
reasoning: measured, 2,986 characters of it.

This is the single most important thing about driving the service, and it is not
inferable from the endpoint list alone. It also means a harness turn is not one
HTTP call but a loop: `/chat/` once, then `/chat/continue` until `cypher` and
`data` come back non-null.

### The cohort is not in the schema

Unprompted, the planner observed: *"We have no explicit Transplant
node/relationship in schema!"* and proposed deriving the cohort from
`BilledDiagnosis` codes. This is correct against the label table in §2, and it
matters for the gold set: a question phrased around "kidney transplant patients"
is really a question about ICD-10 codes, which is the terminology-resolution
problem measured in §3 rather than a retrieval problem. Questions for the gold
set should either name the codes or accept that both systems are being scored on
their code handling.

### Why the exchange fails: the planner model was swapped

Port 8001 is a vLLM OpenAI-compatible server -- the planner. Measured
30 August 2026:

```bash
# SERVER
curl -s -m 10 http://localhost:8001/v1/models
```

| Property | Value |
|---|---|
| model | `/data/llm_models/Qwen3.8-27B-FP8` |
| served by | vLLM |
| `max_model_len` | **8,192** |

The architecture document describes the planner as Qwen2.5-14B-Instruct. It is
not: it is a Qwen3 reasoning model, which explains both symptoms exactly.

1. Qwen3 emits `<think>...</think>` reasoning blocks. That is the `</think>`
   appearing in `answer` with no opening tag.
2. Its tool calls came back as `<function=name><parameter=name>` -- the
   Qwen3-Coder XML convention -- rather than JSON inside `<tool_call>`, which is
   the Hermes-style format a Qwen2.5 deployment would emit and a parser written
   for it would expect.

So the backend cannot parse its own model's tool calls, nothing executes, and
`/chat/continue` raises rather than advancing. **This is a deployment regression
on this host, not a usage error**, and it is not fixable from the client side.
It needs the SnapQuery maintainers.

### The context budget is 8,192 tokens, and that is a thesis number

`max_model_len` is 8,192 for the model that writes every Cypher query the
comparator produces. The live database schema is embedded in its system prompt
and its own reasoning is generated inside the same window, so the budget
available to retrieved content is well under 8,192.

Against the measured patient in §4: 45,562 edges at roughly ten tokens per
serialised triple is on the order of 4.5x10^5 tokens for one patient. **Under 1%
of a single patient's prepared graph can be placed in front of this model at
once**, and that is the whole patient, not the cohort.

This converts the context-window argument in Chapter 1 from a general claim
about large language models into a measured property of the deployed system the
thesis compares against. The ten-tokens-per-triple figure is an estimate and
should be replaced with a tokenizer count -- port 8001 exposes `/tokenize`, so
it can be measured exactly on the real serialisation.

Still unknown: **whether the result rows carry node identity or only values**.
This cannot be answered until the deployment is repaired, and it is the one
remaining fact that changes how much work the comparison is.

---

## 8. Five questions on the live clinical graph — 8 September 2026

First retrieval comparison run against the real STCS graph rather than the
replica. Read-only, one patient.

```
# SERVER
source ~/.ikgqa.env
~/venvs/ikgqa/bin/python experiments/server_real_questions.py --derive 5 --out real_q1
```

Graph as loaded: 15,810 nodes, 45,562 edges, 614 distinct embed texts
(25.7 nodes per text), 178 of 222 diagnoses code-only. Identical to the
19-20 August figures, so the loader is stable.

Questions are **auto-derived ceilings**: the query is the target node's own
`embed_text`. No clinical validation, and a real question can only do worse.

### Result

| retriever | questions | mean recall | SD | mean nodes | mean s |
|---|---|---|---|---|---|
| PCST | 5 | **0.7667** | 0.3195 | **11.4** | 0.025 |
| shortest paths | 5 | **0.7667** | 0.3195 | **11.4** | 0.013 |
| BFS expansion | 5 | **0.7667** | 0.3195 | 23.8 | 0.015 |
| top-k nodes + neighbours | 5 | **0.7667** | 0.3195 | 23.8 | 0.016 |
| top-k triples (KAPING) | 5 | 0.2833 | 0.4118 | 9.4 | 0.020 |

### F10. The four-way tie replicates on real data

Four strategies returned *identical* mean recall and identical standard
deviation, as they did on the replica (0.164, and 0.1461 SD 0.0081 across ten
patients). The replica was not producing that agreement as an artefact of
synthetic content: it is a property of this graph's structure.

### F11. RETRACTED — was a measurement bug, not a finding

**This finding is withdrawn.** It read: KAPING scored 0.2833 against 0.7667, so
connectivity is decisive. The cause was a bug in `experiments/server_real_questions.py`:
`TopKTriples` was constructed without its `encoder` argument, and its own
docstring warns what that does —

> without it the ranking falls back to relation text alone, which is a
> materially weaker baseline than KAPING as published, and understating a
> baseline is as much a result error as overstating your own method.

SPHN relation names are camelCase (`hasDiagnosis`), which tokenise to one opaque
word, so every triple tied at cosine 0 and the baseline degraded to picking
whatever the edge order put first. On the replica, adding the encoder moves
KAPING from **0.2778 to 0.9583**, identical to the other four.

Fixed in `retrievers_for`, which now requires the encoder. **The ceiling and
gold runs in this section predate the fix and their KAPING column is void.**
Everything else in them stands, because the other four strategies were
unaffected and the code-only result was 0.0000 for all five.

The consequence is the opposite of what F11 claimed: the tie is **five-way**, not
four-way, which strengthens the negative RQ2 result rather than qualifying it.

### F12. PCST is 2.1x more compact at identical recall

PCST and shortest paths reach 0.7667 with 11.4 nodes; BFS and top-k plus
neighbours need 23.8 for the same 0.7667. On the replica the same comparison
gave 23x, so the magnitude is graph-dependent and only the direction is stable.
Recall-per-node is where PCST wins, which is what the size axis was built to
show.

### F13. The 0.4167 rows are arithmetic, not retrieval failure

Per question the pattern is bimodal. Answer sets of 1 and 4 nodes scored
**1.0000**. Both answer sets of 24 nodes scored **0.4167** for all four
connected strategies, and 0.4167 = **10/24** exactly. With `topk=10` at most ten
nodes can carry a prize, so ten of twenty-four is the ceiling. No retriever
could beat it at that budget.

Same phenomenon as the analyte questions capped at 11/2407, but clean enough
here to state as arithmetic rather than as a limitation.

### Correction to Table 2.1 (Chapter 2)

The table lists neighbourhood expansion as giving a connected result: **yes**.
Measured, `top-k nodes + neighbours` returned `connected = False` on the Drug
question. Expansion from a *single* seed is connected; expansion from *k* seeds
is a union of k balls and can be disconnected when they do not overlap. The
table needs either a qualification or a split row.

### What this run cannot see

The ceiling design is blind to the terminology problem. For a code-only
diagnosis the derived query *is* the code, which matches its own node exactly,
so it scores 1.0. A real user asks by disease name and matches nothing, which is
the 0.00 that terminology resolution lifts to 0.80. **The ceiling therefore
cannot measure the single largest effect in the thesis**, and only hand-written
questions can. That is the argument for the gold set, now with evidence.

Encoder was still bag-of-words, so absolute values are not clinical numbers;
the between-strategy comparison holds because all five share the same scores.

---

## 9. Five authored questions on the live graph — 8 September 2026

First measurement with **hand-written** questions on the real record, phrased as
a clinician would ask. Predictions were written down before the run.

```
# SERVER
~/venvs/ikgqa/bin/python experiments/server_real_questions.py \
  --draft-gold ~/q5.jsonl --pick Hypokali --pick Kachexie \
  --pick "bezeichneter Diabetes" --pick B96.0 --pick B96.5
~/venvs/ikgqa/bin/python experiments/server_real_questions.py --gold ~/q5.jsonl --out real_q2
```

Three questions target diagnoses that carry a German description; two target
diagnoses stored as **an ICD-10 code and nothing else**, and were asked using the
organism name a clinician would use.

### F14. Every prediction held, 5 of 5

| q | kind | answer nodes | predicted | measured |
|---|---|---|---|---|
| q1 | named | 2 | high | **1.0000** |
| q2 | named | 2 | high | **1.0000** |
| q3 | named | 8 | high | **1.0000** |
| q4 | code only | 3 | 0.00 | **0.0000** |
| q5 | code only | 16 | 0.00 | **0.0000** |

(KAPING column void, see F11. The four other strategies are as stated.)

### F15. Code-only diagnoses score 0.0000 for every strategy, on real data

Not one of the five retrievers returned a single answer node for q4 or q5. This
is the thesis's central claim measured on the real record rather than on the
replica: **when the graph stores a code and the question carries words, the
answer node is unreachable, and no selection strategy changes that.**

Scope: 178 of this patient's 222 diagnosis nodes are code-only. So roughly four
in five diagnoses are invisible to every method implemented here.

This is the strongest available argument for the terminology cross-walk, and it
is now an argument from measurement rather than from the fixture.

### F16. PCST is two orders of magnitude more compact at identical recall

On the three named questions, PCST and shortest paths returned **10 nodes** at
recall 1.0000. BFS expansion and top-k-plus-neighbours returned **2,000** — the
generator cap — for the same 1.0000. Averaged over the five questions,
**10.4 nodes against 1,208.4**, a factor of **116**.

Both expansion methods hit the cap on every named question, which is the "no
size control" row of Table 2.1 behaving exactly as predicted: hops are integers,
and one hop already reaches the whole region.

Note this is the same comparison that gave 2.1x on the ceiling questions and 23x
on the replica. The factor is highly question-dependent; only the direction is
stable. Report the direction, and give the range.

### F17. On q3 the returned subgraph was almost all answer

PCST returned 10 nodes for q3 and 8 of them were the 8 answer nodes, with one
case node and one patient node. Recall 1.0 at precision 0.8 in a 10-node budget.
Worth reporting because the thesis so far only shows recall against size, and
this is the one question where the subgraph is small enough for precision to be
meaningful by inspection.

### Still to do

Showing the *fix* on real data needs the real ICD-10-GM catalogue, which we do
not have. The cross-walk currently has only the synthetic one. Until Andi
supplies it we can measure the problem (F15) but not the repair.
