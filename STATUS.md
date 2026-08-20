# Status and next steps

_Last updated: 19 August 2026_

Read this first when you come back to the project. It says where things stand,
the exact commands to get running again, and the facts that took effort to
discover so you never have to work them out twice.

---

## 1. Where things stand

Server access works. The clinical Neo4j database is reachable and authenticating.
The code side has a tested Python package implementing PCST retrieval, four
baselines and the thesis metric. What is still missing is the bridge between the
two: a loader that turns the real STCS graph into the package's data structure.
That is the next piece of work, and it needs the schema output from `recon.py`.

---

## 2. How to pick up again

### Working without server access

Everything below runs on the laptop with no database. `ikgqa.data.replica`
generates a synthetic patient matching patient #0's measured structure exactly,
so the whole pipeline can be developed and swept offline. Every server
measurement is written down in `docs/server_measurements.md`, including a list of
what still needs measuring next time there is access.

### On your laptop — the code

```powershell
cd "C:\Users\serxh\OneDrive\Documents\THESIS\Master Thesis Code"
PCST\.venv\Scripts\python.exe -m pytest
```

Expect **137 passed** in about 4 seconds. If that holds, nothing has rotted.

```powershell
PCST\.venv\Scripts\python.exe -m ikgqa.demo          # PCST, one stage at a time
PCST\.venv\Scripts\python.exe -m ikgqa.eval.toy_report   # all retrievers compared
PCST\.venv\Scripts\python.exe thesis\tools\checktex.py   # dangling refs, missing bib keys, page budget
```

### Sending the code to the server (no GitHub, one device login)

```powershell
cd "C:\Users\serxh\OneDrive\Documents\THESIS\Master Thesis Code"
tar --exclude='__pycache__' --exclude='*.egg-info' --exclude='*.pyc' --exclude='.venv' -czf ikgqa.tar.gz src tests experiments docs pyproject.toml README.md STATUS.md
scp ikgqa.tar.gz sdosku@chil.scicoreplus.unibas.ch:~/
```

Then on the server:

```bash
chmod -R u+rwX ~/thesis 2>/dev/null; rm -rf ~/thesis; mkdir -p ~/thesis
tar -xzf ~/ikgqa.tar.gz -C ~/thesis
chmod -R u+rwX ~/thesis                       # NOT optional -- see below
cd ~/thesis && ~/venvs/ikgqa/bin/pip install -e ".[dev]"
```

Two traps, both already paid for:

- **Windows `tar` records every directory as `dr-xr-xr-x`.** Windows has no POSIX
  permission bits, so bsdtar invents them and directories come out with no owner
  write bit. The first extraction works, then leaves the tree read-only, so the
  *next* extraction and `pip install -e .` both fail with `Permission denied`.
  The `chmod -R u+rwX` after extracting is the fix.
- **The `--exclude` flags are load-bearing.** Without them the archive carries
  `__pycache__` (compiled for Windows Python 3.12; the server runs 3.10) and a
  Windows `src/ikgqa.egg-info`, which makes the server's pip try to write into a
  read-only directory it did not create.

### On the CHIL server — the data

```bash
ssh sdosku@chil.scicoreplus.unibas.ch
```

It prints a `https://login.biomedit.ch/...device?user_code=XXXX-XXXX` URL.
Open it in a browser, log in, **then** press Enter in the terminal. Pressing
Enter early is what makes it loop with a new code.

```bash
source ~/.ikgqa.env                     # loads the Neo4j credentials
python3 ~/recon.py | less               # schema recon, read-only, no installs needed
```

Check the Bolt driver still connects:

```bash
~/venvs/ikgqa/bin/python -c "
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
d.verify_connectivity(); print('bolt OK'); d.close()"
```

---

## 3. Facts worth not re-deriving

| Thing | Value / behaviour |
|---|---|
| Server | `chil.scicoreplus.unibas.ch`, username **`sdosku`** |
| Hostname once inside | `sphn-cohort-webserver-chil` |
| Login | BioMedIT SSO device flow — **not** SSH keys. Your key is irrelevant here |
| Wrong usernames (already tried) | `dosku0000`, `932340783807@eduid.ch` — both rejected |
| Wrong server (already tried) | `stcs-universe.scicore.unibas.ch` is a Django/Keycloak **web portal**, no SSH |
| Neo4j via Bolt | `bolt://localhost:7687` |
| Neo4j via HTTP | `http://localhost:7474`, database name `neo4j` |
| Neo4j version | 5.20.0, Community edition, **read-only** — writes fail by design |
| `bolt://neo4j_usb:7687` | From the README. Only resolves **inside Docker**; from the host use `localhost` |
| `docker ps` | Permission denied. You are not in the docker group — and do not need to be |
| PyPI | Reachable (`HTTP/2 200`), so `pip install` works normally |
| Python venv on server | `~/venvs/ikgqa` (has the `neo4j` driver) |
| Credentials file | `~/.ikgqa.env`, mode `600`. **Never** in code, never committed |
| Andi's note | `/home/gretriver_master_thesis/README.md` — root-owned, read-only, holds the credentials |

### The credential trap that cost an hour

The password **contains literal double-quote characters** as its first and last
characters (15 chars total). In `~/.ikgqa.env` it must be wrapped in *single*
quotes so the shell does not eat them:

```
export NEO4J_PASSWORD='"xyz..."'
```

Two rules that follow:

- **Single quotes always** in that file.
- **`source ~/.ikgqa.env` after every edit**, or the old value stays in your shell.

Verify without revealing anything:

```bash
python3 -c "
import os; p = os.environ['NEO4J_PASSWORD']
print('length', len(p), '| first', repr(p[0]), '| last', repr(p[-1]))"
```

Length must read **15**. If it reads 13, the quotes were stripped.

Also: Neo4j locks you out with **HTTP 429** after a few failed logins. If you see
that, wait a minute — do not keep retrying. Test auth with a *single* request
before running anything that fires several.

---

## 4. What is built

Python package `ikgqa`, installed editable into `PCST/.venv`.

| Module | Contents |
|---|---|
| `ikgqa.graph` | `TextualGraph` — nodes, edges, embeddings, validated on construction |
| `ikgqa.encoders` | `BagOfWordsEncoder` (offline, deterministic), `SentenceTransformerEncoder` (lazy) |
| `ikgqa.pcst` | The PCST algorithm + a verbatim copy of upstream as a test oracle |
| `ikgqa.retrieval` | `PCST`, `TopKTriples`, `TopKNodesPlusNeighbors`, `BFSExpansion`, `ShortestPaths` |
| `ikgqa.eval` | `metrics.py` (answer-node recall), `sweep.py` (recall-vs-size curve), `toy_report.py` |
| `ikgqa.data` | Toy graphs, including a small clinical one |

Verified facts about it:

- `reference.py` is **byte-identical** to upstream G-Retriever `main` (diffed 18 Aug 2026).
- 122 tests pass, including PCST equivalence against that copy on curated *and* random graphs.
- `numpy<2` is mandatory: under NumPy 2 the `pcst_fast` wheel returns correctly shaped
  garbage without raising. `check_pcst_fast_sanity()` catches it at runtime.
- Aggregate questions ("how many patients…") are recorded as `NaN` with a reason,
  never scored as a miss. See the docstring in `eval/metrics.py` for why.

---

## 4a. Findings so far (Chapter 5/6 material)

Keep adding to this list. Each item is measured, not assumed.

**F1. `cost_e` saturates upward.** Above the top edge prize, raising it stops shrinking
the subgraph, because `adjust_edge_cost` caps it at `max_edge_prize * (1 - c/2)`. Sweep
`cost_e` *downward* from small values; use `topk`/`topk_e` to reach larger subgraphs.

**F2. G-Retriever's edge prizes collapse on this graph shape.** `compute_edge_prizes`
splits each tier's budget across all edges tied in that tier. With `hasCode` on
4,883,809 edges, its prize is `k / 4,883,809` ≈ 1e-6; `adjust_edge_cost` then lowers
`cost_e` to just under that, so **every edge becomes nearly free and size control
disappears**. Confirmed two ways: arithmetically, and empirically in
`experiments/synthetic_clinical.py`, where PCST with published defaults
(`topk_e=5, cost_e=0.5`) returned **the entire graph** — 6,055 of 6,055 nodes.
Fix: `topk_e=0` with an explicitly chosen `cost_e` returned 9 nodes on the same graph.

**F3. Node selection is decided by tie-breaking, not similarity.** 4,832,744 lab tests
share **544** distinct code descriptions, so thousands of nodes are exactly tied. In the
synthetic replica, 41 nodes tied for first place, and `tie_break="stable"` versus
`"auto"` (torch) returned **different node sets**. Two consequences: pin `tie_break`
explicitly for reproducibility, and accept that similarity can only identify a concept
*type* — identifying the *instance* must come from structure or time. That is the
argument for candidate generation, and it is the thesis's spine.

**F4. Hub pruning did not change subgraph size** in the synthetic test (`H3`, not
supported), but the provenance hub *was* included in the retrieved subgraph. So the
case for pruning is context pollution, not size. Retest on real data — the synthetic
had 2 hub nodes against the real graph's 168 `SourceSystem` nodes at average degree
69,525.

**F5. Text coverage is much better than first thought.** Lab codes carry
`hasLongName` on 100% of 4.8M uses (544 distinct). Diagnoses: 44% of 46,143 uses have a
long name; all 46,143 are ICD-10 shaped, so a public catalogue fills the rest.

**F6. The graph is bilingual — likely the biggest silent risk.** Lab codes are LOINC
(544 distinct, `hasType = "LOINC"`, English long names). Diagnosis codes are
**ICD-10-GM**, versioned by year (`10-GM-2012` … `10-GM-2024`, summing to exactly 2,921),
whose official labels are **German**. Also present: ATC 737, UCUM 70, SNOMED-CT 69.
`all-roberta-large-v1` — G-Retriever's encoder — is English-only, so it would
systematically under-rank diagnoses for a reason invisible in the results. Enrichment
gap: **2,921 − 972 = 1,949** diagnosis codes with no name.

**Resolution (decided 20 Aug 2026): cross-walk, not translation.** A code is
language-independent, so `N18.5` → canonical description is a *lookup*, deterministic and
auditable, where machine translation of clinical shorthand is neither. Four tiers, each
counted and reported in Chapter 6:

| Tier | Mechanism |
|---|---|
| 1 | code + version → official catalogue description (English) |
| 2 | plus the parent concept's description (`N18.5` → `N18` chronic kidney disease) |
| 3 | no catalogue entry: keep the graph's own label, marked as unresolved |
| 4 | nothing: bare code, counted as unreachable by similarity |

Tier 2 is not just a fallback — widening the matchable surface with hierarchy text *is* a
schema signal, which is RQ3. The whole diagnosis vocabulary is 2,921 codes, so this is one
reviewed CSV (`code,version,label,parent_label`) committed to the repo, not runtime MT.
Encoder choice (English-only vs multilingual) stays an ablation because the *question*
language is still unconfirmed — see §7.

**F8. Measured on real data (patient #0, 20 Aug 2026): the loader works and the
synthetic predictions hold.** One command:

```bash
source ~/.ikgqa.env && ~/venvs/ikgqa/bin/python -m ikgqa.data.sphn --patient 0
```

| Quantity | Patient #0 |
|---|---|
| Nodes / edges after preparation | 15,810 / 45,562 |
| Lab observations / distinct analytes | 12,349 / 198 |
| Drug administrations / distinct drugs | 2,655 / 279 |
| Diagnoses | 222 |
| Hospital cases | 106 |
| Distinct embed texts | **614** |
| Diagnoses with no description | **178 of 222 (80%)** |
| Lab tests with no description | 0 of 12,349 |

Three consequences, all Chapter 6 material:

- **Tie domination is real, not a synthetic artefact.** 15,810 nodes share 614
  distinct texts: 25.7 nodes per text. Similarity can sort the graph into at most
  614 classes, so within a class the ranking is decided by iteration order. F3 confirmed
  on real data.
- **One patient is already 11× a whole WebQSP sample** (1,371 nodes average). Candidate
  generation is not an optimisation, it is a precondition — and note this is *one* of
  1,197 patients.
- **The text gap is entirely on the diagnosis side.** Labs are fully described; 80% of
  this patient's diagnoses are unreachable by similarity under any wording. That makes
  the F6 cross-walk the highest-value single piece of work left.

**F9. Measured-shape replica + sweep, run offline (20 Aug 2026).**
`ikgqa.data.replica` generates a synthetic patient matching patient #0 **exactly** on
every structural count (15,810 nodes, 45,562 edges, all label counts). It does this by
generating rows in the Cypher's own shape and feeding them through the real
`build_patient_graph`, so the structure is identical by construction and the real
assembly code is exercised. Results in `experiments/results/`.

```powershell
PCST\.venv\Scripts\python.exe experiments\sweep_replica.py           # full size
PCST\.venv\Scripts\python.exe experiments\sweep_replica.py --small   # 10x smaller
```

Four results, all at full patient size:

- **F2 confirmed at scale.** PCST with published defaults (`topk=3, topk_e=5,
  cost_e=0.5`) returned **15,810 of 15,810 nodes — the entire graph.**
- **F1 sharpened: `cost_e` is barely a size dial here.** With `topk=10, topk_e=0`,
  sweeping `cost_e` from 3.0 down to 0.01 moved the subgraph from 8 nodes to 11 and
  then flatlined. The reachable range is 8–11 nodes; the real size dial in this regime
  is `topk`, not `cost_e`. Methodology must sweep both.
- **Neighbourhood expansion has no usable middle.** BFS at 1 hop gives 7.7 nodes
  (0.05% of the graph); at 2 hops, 15,334 (97%). An integer dial cannot land between
  them. This is the "size control lost on a dense graph" claim of §2.4, measured.
  Top-k-nodes-plus-neighbours does the same thing: k=30 → 61 nodes, k=100 → 5,237.
- **At equal size, connectivity contributed nothing.** At ~11 nodes, PCST, shortest
  paths, top-k nodes and KAPING all scored **0.1835** — identical. On this graph the
  similarity signal, not the structure, decides what is retrieved at small sizes. That
  is a direct partial answer to RQ2 and it needs the real encoder before it is final.

Also measured: KAPING returns a **disconnected** result for every k ≥ 3, the predicted
weakness of independent scoring.

Two of my own errors were caught by this run and are worth not repeating: the sweep
first constructed `TopKTriples(k=k)` without an encoder, which silently crippled the
baseline to relation-text-only ranking (understating a baseline is as much an error as
overstating your method); and the planted code-only diagnosis questions initially
quoted the ICD code, so they were answerable by string overlap and measured nothing.
Both are now pinned by tests in `tests/test_replica.py`.

**F10. Repeated-measurement questions may not be subgraph-retrievable at all.**
"All creatinine values" has 2,407 answer nodes for patient #0's replica, so recall at any
usable subgraph size is near zero for *every* method — not a retrieval failure but a sign
that such questions are aggregations, not retrievals. This is the same entity-vs-aggregate
issue that needs settling with Andi (§7), now with a number attached. Note also that
`AdministrativeCase` nodes carry no distinguishing text, so episode or time scoping cannot
come from similarity and must come from a structured pre-filter.

**F7. Upstream returns edge ids unsorted.** `decode_solution` concatenates solver edges
with edges recovered from virtual nodes and never re-sorts, so `selected_edges` comes
back in solver order (e.g. `[12, 6, 5, 2]`). Harmless for set-based metrics, but anything
comparing edge lists positionally will silently disagree. Normalised in
`ikgqa/retrieval/pcst.py` (not in `core.py`, which stays byte-faithful).

---

## 4b. Thesis writing

LaTeX lives in `thesis/`. Build with `thesis/Makefile`; check first with
`thesis/tools/checktex.py`, which reports dangling `\ref`s, missing bib keys and
each chapter against its page budget without needing a LaTeX install.

Target 30–50 pages. Budget encoded in `checktex.py`:

| Chapter | Pages | State |
|---|---|---|
| 1 Introduction | 5 | revised 19 Aug — measured problem statement, RQ1–RQ3 |
| 2 Background | 8 | revised 20 Aug — comparison table, SPHN + terminology sections |
| 3 G-Retriever and PCST | 6 | not started; owns all the algebra |
| 4 Methodology | 8 | not started |
| 5 Implementation | 6 | not started |
| 6 Results and Discussion | 8 | not started; F1–F7 go here |
| 7 Conclusion | 3 | not started |

The repetition Andi flagged had one cause: four ideas each had three or four homes.
Each now has exactly one, recorded in an editorial-note comment at the top of each
revised chapter. Rule to keep: **cross-reference, never restate.**

Still to write in the front matter: the Abstract still says the project is in its
second week and closely repeats Chapter 1.

---

## 5. Next actions, in order

Done: recon, loader, node-text format, first real-data run (F8).

1. **Get real questions.** This is now the only hard blocker: the metric, the SnapQuery
   comparison and the gold set all need them, and they all depend on other people.
   Chase Andi — see §6. Nothing downstream can be finished without this.
2. **Terminology cross-walk** (F6 resolution). 80% of patient #0's diagnoses have no
   text at all, so this is the single highest-value piece of code left. Public catalogue
   data, committed as `terminology/*.csv` (the `.gitignore` exception exists for it).
   First check what the server can reach: `pip` works, arbitrary HTTP may not — if not,
   fetch on the laptop and `scp` it over.
3. **Real embeddings.** `pip install sentence-transformers`, then encode the 614 distinct
   texts per patient on the A100. Note the encoding cost is per *distinct text*, not per
   node, so this is small — 614 encodes for 15,810 nodes.
4. **Candidate generation.** With one patient at 15,810 nodes, PCST needs a narrowing
   step before it, not after. Methodology decision, not a workaround.
5. **SnapQuery baseline harness.** `POST /chat/` then `/chat/continue` on port 8002, with
   slot-filling answered and the confirmation gate simulated. Needs no database access.
6. **Measure raw vs prepared** node counts for one patient, to justify graph preparation
   with a number in Chapter 6 instead of an argument.

---

## 6. Waiting on other people (chase these — longest lead time)

- **Who owns the SnapQuery question logs?** Retained at all? What format? Who exports?
  The whole evaluation depends on real user questions.
- **Who validates the ~50-question gold set?** Needs a clinician's or data manager's
  time. Andi's email: "treat it as a core artifact, not an afterthought."

---

## 7. Open design questions

- **Are the real questions entity-answerable or aggregates?** The architecture doc
  describes the graph agent as handling *cohort* questions with slot filling for time
  window and output shape, which sounds like counts. Answer-node recall works for
  "which drug was patient X given"; a count has no answer node anywhere in the graph.
  Agree with Andi how those are scored — or scope Phase 1 to entity questions.
- **Comparability of the size dials.** PCST sweeps `cost_e`; the baselines sweep `k`.
  These are not the same quantity. Plot both against a *measured* size (nodes, or prompt
  characters) rather than against their own parameter, and state the caveat in
  Methodology.
- **How SnapQuery's result becomes a subgraph.** It returns result rows, not a subgraph.
  The prep doc treats the entities its rows touch as its induced subgraph — workable,
  but it yields a single point rather than a curve, so it needs saying explicitly.

---

## 8. Rules that do not bend

- **No patient data leaves the server.** Schema, label names, counts, property *names*,
  aggregate statistics: fine to share. Property values, patient ids, lab results, dates:
  never. Governed clinical data stays on BioMedIT (thesis §4.4).
- **The database is read-only.** Writes fail by design. Nothing you run can damage it.
- **Credentials live only in `~/.ikgqa.env`.** Not in code, not in git, not in chat.
