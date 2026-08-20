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
4. **Does SnapQuery respond on port 8002**, and what does one full exchange look
   like end to end? Nothing about the baseline harness can be finished until one
   real response has been observed.
5. **Confirm the loader's label coverage** (the omissions table in §2) with the
   data owners.
