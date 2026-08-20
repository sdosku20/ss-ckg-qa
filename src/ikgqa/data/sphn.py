"""
ikgqa.data.sphn
===============

Turn one patient's slice of the SPHN clinical graph into a TextualGraph.

Why this module exists at all
----------------------------

G-Retriever assumes a *textual graph*: every node carries a sentence to embed.
The STCS graph does not. Measured on 19 Aug 2026:

* 21,537,305 nodes, 72,218,890 edges, 1,197 patients (~18,000 nodes per patient)
* almost no node carries text. ``LabTest`` has only a ``uid``; ``Drug`` has no
  properties at all. Language lives on 10,014 ``Code`` nodes and a few hundred
  ``Substance`` / ``DrugArticle`` nodes.
* those ``Code`` nodes are shared at enormous multiplicity -- one has degree
  1,226,895 -- so keeping them as nodes puts every creatinine test in the
  hospital two hops from every other.
* administrative hubs dominate: 168 ``SourceSystem`` nodes average degree
  69,525; 6 ``BodySite`` nodes absorb 3,454,881 edges.

So a faithful dump of the graph is both unembeddable and structurally useless.
Three transformations fix that, and each is a *reportable methodological choice*,
not plumbing. They are named here so they can be described in the thesis and
switched off in an ablation.

1. **Vocabulary folding.** ``Code``, ``Unit``, ``Quantity``, ``BodySite`` and
   ``ReferenceRange`` stop being nodes and become text on the node that
   references them. This is what gives text-free nodes something to embed, and
   it removes the worst hubs. Done inside Cypher, so only flat rows cross the
   wire rather than 18,000 raw nodes.

2. **Provenance pruning.** ``SourceSystem``, ``DataProvider``, ``DataRelease``,
   ``TimePattern``, ``CareHandling``, ``Location`` and
   ``HealthcarePrimaryInformationSystem`` are dropped. They record where data
   came from, not what happened to a patient, and they collapse graph distance.

3. **Patient scoping.** The unit of retrieval is one patient (or a cohort),
   never the whole graph. This is what makes PCST tractable, and it is why
   sharing a ``LabTestType`` node is safe here: globally it is a hub with
   degree in the millions, but inside one patient it has degree ~7, where it
   usefully groups "all this patient's creatinine measurements".

What structure survives
-----------------------

Deliberately *not* a star graph. If every observation hung directly off the
patient, PCST would have nothing to connect and the connectivity constraint --
the entire object of study -- would be vacuous. Three things keep real
structure:

* ``AdministrativeCase`` ties the labs, diagnoses and drugs of one hospital
  stay together, which is what a clinical question usually means by "then".
* ``LabTestType`` groups repeated measurements of the same analyte.
* ``Drug`` is shared across its administration events.

Two text fields per node
------------------------

``embed_text`` is the semantic core -- code names, drug names, no values or
dates. ``display_text`` is what the LLM reads, values and dates included.

The reason is measured: 4,832,744 lab tests resolve to 544 distinct LOINC
descriptions, so ``embed_text`` deduplicates ~9,000-fold and the whole graph's
distinct text fits in a few thousand embeddings. Dates and values would make
every string unique, destroy that saving, and add nothing a cosine similarity
can use.

Safety
------

Every query here is read-only, and the database itself is read-only, so writes
fail by design. Patients are addressed **by index, not by identifier**
(``patient_index=0``), so pseudo-identifiers never need to be copied, pasted or
logged.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ikgqa.graph import TextualGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Labels dropped entirely (transformation 2). Provenance and bookkeeping.
PRUNED_LABELS = (
    "SourceSystem",
    "HealthcarePrimaryInformationSystem",
    "DataProvider",
    "DataRelease",
    "TimePattern",
    "CareHandling",
    "Location",
)

#: Labels folded into the text of their referencing node (transformation 1).
FOLDED_LABELS = ("Code", "Unit", "Quantity", "BodySite", "ReferenceRange")


@dataclasses.dataclass(frozen=True)
class Neo4jSettings:
    """Connection settings, read from the environment.

    Never hard-code credentials. On the CHIL server these come from
    ``~/.ikgqa.env`` (mode 600), which is sourced before running anything:

        source ~/.ikgqa.env
    """

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        missing = [k for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(
                f"missing environment variables {missing}. Run `source ~/.ikgqa.env` first.\n"
                "Note: bolt://neo4j_usb:7687 from the project README only resolves inside "
                "Docker; from the CHIL host use bolt://localhost:7687."
            )
        return cls(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        )


# ---------------------------------------------------------------------------
# Cypher. Read-only. Folding happens here so only flat rows cross the wire.
# ---------------------------------------------------------------------------

#: Address a patient by position, so pseudo-identifiers stay on the server.
#: ORDER BY makes the position stable across runs.
Q_PATIENT_BY_INDEX = """
MATCH (p:SubjectPseudoIdentifier)
RETURN p.hasIdentifier AS pid
ORDER BY p.hasIdentifier
SKIP $index LIMIT 1
"""

Q_PATIENT_COUNT = "MATCH (p:SubjectPseudoIdentifier) RETURN count(p) AS n"

#: Lab observations. Code/Quantity/Unit/ReferenceRange/BodySite are folded into
#: scalar columns rather than returned as nodes.
Q_LABS = """
MATCH (ev:LabTestEvent)-[:hasSubjectPseudoIdentifier]->(p:SubjectPseudoIdentifier)
WHERE p.hasIdentifier = $pid
MATCH (ev)-[:hasLabTest]->(t:LabTest)-[:hasCode]->(code:Code)
OPTIONAL MATCH (t)-[:hasResult]->(res:LabResult)
OPTIONAL MATCH (res)-[:hasQuantity]->(qty:Quantity)
OPTIONAL MATCH (qty)-[:hasUnit]->(unit:Unit)
OPTIONAL MATCH (res)-[:hasNumericalReference]->(ref:ReferenceRange)
OPTIONAL MATCH (ev)-[:hasAdministrativeCase]->(case:AdministrativeCase)
OPTIONAL MATCH (ev)-[:hasSample]->(smp:Sample)
OPTIONAL MATCH (smp)-[:hasMaterialTypeCode]->(mat:Code)
RETURN ev.uid              AS event_uid,
       toString(ev.hasDateTime) AS observed_at,
       code.hasValue       AS loinc_code,
       code.hasType        AS code_system,
       coalesce(code.hasLongName, code.hasShortName, code.hasName) AS test_name,
       qty.hasValue        AS numeric_value,
       unit.hasValue       AS unit,
       res.hasStringValue  AS text_value,
       ref.hasRawRange     AS reference_range,
       case.hasIdentifier  AS case_id,
       coalesce(mat.hasLongName, mat.hasName, mat.hasValue) AS material
ORDER BY observed_at, event_uid
LIMIT $limit
"""

Q_DIAGNOSES = """
MATCH (dx:BilledDiagnosis)-[:hasSubjectPseudoIdentifier]->(p:SubjectPseudoIdentifier)
WHERE p.hasIdentifier = $pid
MATCH (dx)-[:hasCode]->(code:Code)
OPTIONAL MATCH (dx)-[:hasAdministrativeCase]->(case:AdministrativeCase)
RETURN toString(dx.hasRecordDateTime) AS recorded_at,
       code.hasValue AS icd_code,
       code.hasType  AS code_system,
       coalesce(code.hasLongName, code.hasShortName, code.hasName) AS diagnosis_name,
       case.hasIdentifier AS case_id
ORDER BY recorded_at, icd_code
LIMIT $limit
"""

Q_DRUGS = """
MATCH (ad:DrugAdministrationEvent)-[:hasSubjectPseudoIdentifier]->(p:SubjectPseudoIdentifier)
WHERE p.hasIdentifier = $pid
MATCH (ad)-[:hasDrug]->(d:Drug)
OPTIONAL MATCH (d)-[:hasActiveIngredient]->(sub:Substance)
OPTIONAL MATCH (d)-[:hasArticle]->(art:DrugArticle)
OPTIONAL MATCH (d)-[:hasQuantity]->(dose:Quantity)
OPTIONAL MATCH (dose)-[:hasUnit]->(dose_unit:Unit)
OPTIONAL MATCH (ad)-[:hasAdministrativeCase]->(case:AdministrativeCase)
RETURN toString(ad.hasStartDateTime) AS started_at,
       toString(ad.hasEndDateTime)   AS ended_at,
       elementId(d)          AS drug_key,
       sub.hasGenericName    AS substance,
       art.hasName           AS article,
       dose.hasValue         AS dose_value,
       dose_unit.hasValue    AS dose_unit,
       case.hasIdentifier    AS case_id
ORDER BY started_at, drug_key
LIMIT $limit
"""

Q_CASES = """
MATCH (case:AdministrativeCase)-[:hasSubjectPseudoIdentifier]->(p:SubjectPseudoIdentifier)
WHERE p.hasIdentifier = $pid
OPTIONAL MATCH (case)-[:hasAdmission]->(adm:Admission)
OPTIONAL MATCH (case)-[:hasDischarge]->(dis:Discharge)
RETURN case.hasIdentifier AS case_id,
       toString(adm.hasDateTime) AS admitted_at,
       toString(dis.hasDateTime) AS discharged_at
ORDER BY admitted_at, case_id
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------


def _clean(value: Any) -> str:
    """Render a Neo4j scalar as trimmed text, with nulls becoming empty."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _join(parts: Iterable[str], sep: str = "; ") -> str:
    return sep.join(p for p in (p.strip() for p in parts) if p)


def lab_test_type_text(row: Dict[str, Any]) -> Tuple[str, str]:
    """Text for a lab *analyte* (shared across this patient's measurements).

    Returns (embed_text, display_text).

    Falls back to the bare code when no name exists. That fallback is a signal
    worth counting, not hiding: a node whose only text is "17861-6" cannot be
    retrieved by meaning, and ``GraphStats.unnamed_*`` reports how often it
    happens.
    """
    name = _clean(row.get("test_name"))
    code = _clean(row.get("loinc_code"))
    system = _clean(row.get("code_system")) or "code"
    label = name or code or "unknown lab test"
    embed = f"lab test: {label}"
    display = _join([f"Lab test: {label}", f"{system} {code}" if code else ""])
    return embed, display


def lab_observation_text(row: Dict[str, Any]) -> Tuple[str, str]:
    """Text for one measurement: value and time live only in display_text."""
    name = _clean(row.get("test_name")) or _clean(row.get("loinc_code")) or "lab test"
    value = _clean(row.get("numeric_value")) or _clean(row.get("text_value"))
    unit = _clean(row.get("unit"))
    embed = f"lab result: {name}"
    display = _join(
        [
            f"Lab result: {name}",
            f"value: {value} {unit}".strip() if value else "",
            f"reference: {_clean(row.get('reference_range'))}"
            if _clean(row.get("reference_range"))
            else "",
            f"material: {_clean(row.get('material'))}" if _clean(row.get("material")) else "",
            f"observed: {_clean(row.get('observed_at'))[:19]}"
            if _clean(row.get("observed_at"))
            else "",
        ]
    )
    return embed, display


def diagnosis_text(row: Dict[str, Any]) -> Tuple[str, str]:
    """Text for a billed diagnosis.

    ICD-10-GM labels are German while LOINC labels are English, so a graph built
    from both is bilingual. An English-only encoder will systematically
    under-rank one of them; see ``ikgqa.encoders`` for the multilingual option.
    """
    name = _clean(row.get("diagnosis_name"))
    code = _clean(row.get("icd_code"))
    system = _clean(row.get("code_system")) or "ICD-10"
    label = name or code or "unknown diagnosis"
    embed = f"diagnosis: {label}"
    display = _join(
        [
            f"Diagnosis: {label}",
            f"{system} {code}" if code else "",
            f"recorded: {_clean(row.get('recorded_at'))[:19]}"
            if _clean(row.get("recorded_at"))
            else "",
        ]
    )
    return embed, display


def drug_text(row: Dict[str, Any]) -> Tuple[str, str]:
    """Text for a drug (shared across its administrations)."""
    substance = _clean(row.get("substance"))
    article = _clean(row.get("article"))
    label = substance or article or "unknown drug"
    embed = f"drug: {label}"
    display = _join([f"Drug: {label}", f"product: {article}" if article and substance else ""])
    return embed, display


def drug_administration_text(row: Dict[str, Any]) -> Tuple[str, str]:
    label = _clean(row.get("substance")) or _clean(row.get("article")) or "drug"
    dose = _clean(row.get("dose_value"))
    unit = _clean(row.get("dose_unit"))
    embed = f"drug administration: {label}"
    display = _join(
        [
            f"Drug administration: {label}",
            f"dose: {dose} {unit}".strip() if dose else "",
            f"started: {_clean(row.get('started_at'))[:19]}"
            if _clean(row.get("started_at"))
            else "",
        ]
    )
    return embed, display


def case_text(row: Dict[str, Any]) -> Tuple[str, str]:
    admitted = _clean(row.get("admitted_at"))[:10]
    discharged = _clean(row.get("discharged_at"))[:10]
    embed = "hospital case admission encounter"
    display = _join(
        [
            "Hospital case",
            f"admitted: {admitted}" if admitted else "",
            f"discharged: {discharged}" if discharged else "",
        ]
    )
    return embed, display


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GraphStats:
    """What the loader did, so a run can be reported rather than guessed at.

    ``unnamed_labs`` and ``unnamed_diagnoses`` matter for the thesis: they count
    nodes whose only text is a bare code, i.e. nodes that similarity search
    cannot reach. That number is the size of the terminology-enrichment gap.
    """

    patient_index: int
    labs: int = 0
    diagnoses: int = 0
    drug_administrations: int = 0
    cases: int = 0
    lab_types: int = 0
    drugs: int = 0
    unnamed_labs: int = 0
    unnamed_diagnoses: int = 0
    distinct_embed_texts: int = 0
    truncated: Tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"patient #{self.patient_index}",
            f"  lab observations      {self.labs:>7}  (across {self.lab_types} analytes)",
            f"  diagnoses             {self.diagnoses:>7}",
            f"  drug administrations  {self.drug_administrations:>7}  (across {self.drugs} drugs)",
            f"  hospital cases        {self.cases:>7}",
            f"  distinct embed texts  {self.distinct_embed_texts:>7}",
        ]
        if self.unnamed_labs or self.unnamed_diagnoses:
            lines.append(
                f"  UNNAMED (code only)   labs={self.unnamed_labs} "
                f"diagnoses={self.unnamed_diagnoses}  <- unreachable by similarity"
            )
        if self.truncated:
            lines.append(f"  TRUNCATED at limit: {', '.join(self.truncated)}")
        return "\n".join(lines)


class _Builder:
    """Accumulates nodes and edges, de-duplicating shared entities by key."""

    def __init__(self) -> None:
        self.embed: List[str] = []
        self.display: List[str] = []
        self.labels: List[str] = []
        self.by_key: Dict[Tuple[str, Any], int] = {}
        self.edges: List[Tuple[int, str, int]] = []

    def node(self, key: Tuple[str, Any], label: str, embed: str, display: str) -> int:
        """Add a node, or return the existing id if this key was seen before."""
        if key in self.by_key:
            return self.by_key[key]
        idx = len(self.embed)
        self.embed.append(embed)
        self.display.append(display)
        self.labels.append(label)
        self.by_key[key] = idx
        return idx

    def edge(self, src: int, relation: str, dst: int) -> None:
        self.edges.append((src, relation, dst))


def default_encoder(embed_texts: Iterable[str], relations: Iterable[str]) -> Any:
    """The dev-mode encoder, defined in exactly one place.

    A question has to be encoded in the same space as the graph, and with a
    bag-of-words encoder "the same space" means "the same vocabulary in the same
    order". Callers that build a graph with ``encoder=None`` reproduce its space
    by calling this with the graph's own texts, rather than guessing the corpus.

    Fitting on composed text and not on raw database rows is deliberate: rows
    put every distinct timestamp and measured value into the vocabulary, which
    on patient #0 gave a 15,010-dimensional space for 614 distinct texts and let
    dates dominate a score meant to be about words.
    """
    from ikgqa.encoders import BagOfWordsEncoder

    return BagOfWordsEncoder(sorted(set(embed_texts)) + sorted(set(relations)))


def build_patient_graph(
    rows: Dict[str, Sequence[Dict[str, Any]]],
    encoder: Any = None,
    patient_index: int = 0,
    name: Optional[str] = None,
) -> Tuple[TextualGraph, pd.DataFrame, GraphStats]:
    """Assemble a TextualGraph from fetched rows.

    Separated from the database on purpose: this function is pure, so it is unit
    tested against fixture rows without a Neo4j connection. ``fetch_patient_rows``
    does the I/O.

    Args:
        rows: mapping with keys "labs", "diagnoses", "drugs", "cases", each a
            sequence of dicts as returned by the module's Cypher.
        encoder: anything with ``encode(list[str]) -> [n, d]``. None fits a
            BagOfWordsEncoder on the composed text, which needs no downloads.
        patient_index: recorded in the stats and the graph name.
        name: overrides the graph name.

    Returns:
        (graph, node_table, stats) where ``node_table`` carries display_text and
        the SPHN label per node -- the graph itself embeds ``embed_text`` only.
    """
    b = _Builder()
    stats = GraphStats(patient_index=patient_index)

    patient = b.node(("patient", patient_index), "SubjectPseudoIdentifier",
                     "patient subject", f"Patient #{patient_index}")

    # -- hospital cases: the structure that ties a stay together --------------
    for row in rows.get("cases", []):
        embed, display = case_text(row)
        case = b.node(("case", row.get("case_id")), "AdministrativeCase", embed, display)
        b.edge(patient, "hasAdministrativeCase", case)
        stats.cases += 1

    def case_node(case_id: Any) -> Optional[int]:
        """A case referenced by an observation may not have been listed."""
        if case_id is None:
            return None
        key = ("case", case_id)
        if key not in b.by_key:
            idx = b.node(key, "AdministrativeCase", "hospital case admission encounter",
                         "Hospital case")
            b.edge(patient, "hasAdministrativeCase", idx)
            stats.cases += 1
            return idx
        return b.by_key[key]

    # -- lab observations, grouped under their analyte ------------------------
    for i, row in enumerate(rows.get("labs", [])):
        type_key = ("lab_type", row.get("loinc_code") or row.get("test_name"))
        t_embed, t_display = lab_test_type_text(row)
        analyte = b.node(type_key, "LabTestType", t_embed, t_display)

        o_embed, o_display = lab_observation_text(row)
        obs = b.node(("lab_obs", row.get("event_uid") or i), "LabObservation", o_embed, o_display)

        b.edge(obs, "hasLabTest", analyte)
        b.edge(patient, "hasLabObservation", obs)
        case = case_node(row.get("case_id"))
        if case is not None:
            b.edge(obs, "hasAdministrativeCase", case)

        stats.labs += 1
        if not _clean(row.get("test_name")):
            stats.unnamed_labs += 1

    # -- diagnoses -----------------------------------------------------------
    for i, row in enumerate(rows.get("diagnoses", [])):
        embed, display = diagnosis_text(row)
        dx = b.node(("dx", (row.get("icd_code"), row.get("recorded_at"), i)),
                    "BilledDiagnosis", embed, display)
        b.edge(patient, "hasDiagnosis", dx)
        case = case_node(row.get("case_id"))
        if case is not None:
            b.edge(dx, "hasAdministrativeCase", case)
        stats.diagnoses += 1
        if not _clean(row.get("diagnosis_name")):
            stats.unnamed_diagnoses += 1

    # -- drug administrations, grouped under their drug -----------------------
    for i, row in enumerate(rows.get("drugs", [])):
        d_embed, d_display = drug_text(row)
        drug = b.node(("drug", row.get("drug_key")), "Drug", d_embed, d_display)

        a_embed, a_display = drug_administration_text(row)
        admin = b.node(("drug_admin", (row.get("drug_key"), row.get("started_at"), i)),
                       "DrugAdministrationEvent", a_embed, a_display)

        b.edge(admin, "hasDrug", drug)
        b.edge(patient, "hasDrugAdministration", admin)
        case = case_node(row.get("case_id"))
        if case is not None:
            b.edge(admin, "hasAdministrativeCase", case)
        stats.drug_administrations += 1

    stats.lab_types = sum(1 for k in b.by_key if k[0] == "lab_type")
    stats.drugs = sum(1 for k in b.by_key if k[0] == "drug")
    stats.distinct_embed_texts = len(set(b.embed))

    # -- embed distinct text once, then map back (the 9,000-fold saving) -----
    unique = sorted(set(b.embed))
    relations = sorted({r for _, r, _ in b.edges})

    if encoder is None:
        encoder = default_encoder(unique, relations)

    matrix = np.asarray(encoder.encode(unique), dtype=np.float32)
    lookup = {text: i for i, text in enumerate(unique)}
    node_emb = matrix[[lookup[t] for t in b.embed]]

    rel_emb_unique = np.asarray(encoder.encode(relations), dtype=np.float32) if relations else None
    rel_lookup = {r: i for i, r in enumerate(relations)}
    edge_emb = (
        rel_emb_unique[[rel_lookup[r] for _, r, _ in b.edges]]
        if relations
        else np.zeros((0, matrix.shape[1]), dtype=np.float32)
    )

    nodes_df = pd.DataFrame({"node_id": np.arange(len(b.embed)), "node_attr": b.display})
    edges_df = pd.DataFrame(
        {
            "src": [s for s, _, _ in b.edges],
            "edge_attr": [r for _, r, _ in b.edges],
            "dst": [d for _, _, d in b.edges],
        }
    )
    graph = TextualGraph(
        nodes=nodes_df,
        edges=edges_df,
        node_emb=node_emb,
        edge_emb=edge_emb,
        name=name or f"stcs-patient-{patient_index}",
    )

    node_table = pd.DataFrame(
        {
            "node_id": np.arange(len(b.embed)),
            "sphn_label": b.labels,
            "embed_text": b.embed,
            "display_text": b.display,
        }
    )
    return graph, node_table, stats


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


def fetch_patient_rows(
    patient_index: int = 0,
    settings: Optional[Neo4jSettings] = None,
    limit: int = 20000,
    driver: Any = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], GraphStats]:
    """Run the read-only queries for one patient.

    Args:
        patient_index: position in the identifier-ordered patient list. Using an
            index rather than an identifier keeps pseudo-identifiers on the
            server.
        settings: defaults to ``Neo4jSettings.from_env()``.
        limit: per-query row cap. Reaching it is recorded in the stats -- a
            silently truncated graph would look like a small patient.
        driver: an existing neo4j driver, mainly for tests.

    Returns:
        (rows, stats) with the same keys ``build_patient_graph`` expects.
    """
    settings = settings or Neo4jSettings.from_env()
    close_after = driver is None
    if driver is None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "the neo4j driver is not installed.\n"
                "  ~/venvs/ikgqa/bin/pip install neo4j\n"
                "For development without a database, build graphs from fixture rows with "
                "build_patient_graph() instead."
            ) from e
        driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))

    stats = GraphStats(patient_index=patient_index)
    try:
        with driver.session(database=settings.database) as session:
            found = session.run(Q_PATIENT_BY_INDEX, index=int(patient_index)).data()
            if not found:
                total = session.run(Q_PATIENT_COUNT).single()
                raise IndexError(
                    f"no patient at index {patient_index} "
                    f"(the graph has {total['n'] if total else 'unknown'})"
                )
            pid = found[0]["pid"]

            rows: Dict[str, List[Dict[str, Any]]] = {}
            truncated: List[str] = []
            for key, query in (
                ("cases", Q_CASES),
                ("labs", Q_LABS),
                ("diagnoses", Q_DIAGNOSES),
                ("drugs", Q_DRUGS),
            ):
                got = session.run(query, pid=pid, limit=int(limit)).data()
                rows[key] = got
                if len(got) >= limit:
                    truncated.append(key)
            stats.truncated = tuple(truncated)
    finally:
        if close_after:
            driver.close()

    return rows, stats


def load_patient_graph(
    patient_index: int = 0,
    encoder: Any = None,
    settings: Optional[Neo4jSettings] = None,
    limit: int = 20000,
    driver: Any = None,
) -> Tuple[TextualGraph, pd.DataFrame, GraphStats]:
    """Fetch and assemble one patient's graph. Convenience wrapper.

    On the CHIL server, prefer the module entry point over a multi-line
    ``python -c``, which the shell mangles on paste::

        source ~/.ikgqa.env
        ~/venvs/ikgqa/bin/python -m ikgqa.data.sphn --patient 0

    With ``encoder=None`` a BagOfWordsEncoder is fitted on the graph's own
    composed text, which needs no model download and is enough to inspect
    structure. Real numbers need ``SentenceTransformerEncoder``.
    """
    rows, fetch_stats = fetch_patient_rows(
        patient_index=patient_index, settings=settings, limit=limit, driver=driver
    )
    graph, table, stats = build_patient_graph(rows, encoder, patient_index=patient_index)
    stats.truncated = fetch_stats.truncated
    return graph, table, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Inspect one patient's graph from the command line.

    Prints only aggregates and SPHN label names -- never a property value -- so
    the output is safe to paste into a report or an email.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ikgqa.data.sphn",
        description="Load one patient's graph from the clinical Neo4j and report its shape.",
    )
    parser.add_argument("--patient", type=int, default=0, help="patient index (not an identifier)")
    parser.add_argument("--limit", type=int, default=20000, help="max rows per query")
    parser.add_argument(
        "--retrieve",
        metavar="QUESTION",
        help="also run PCST for this question and report the subgraph size",
    )
    args = parser.parse_args(argv)

    graph, table, stats = load_patient_graph(patient_index=args.patient, limit=args.limit)
    print(stats.summary())
    print()
    print(graph)
    print(table["sphn_label"].value_counts().to_string())

    # The ratio that decides whether similarity can identify an instance at all.
    n_nodes = len(table)
    n_texts = stats.distinct_embed_texts
    print(
        f"\ntext multiplicity: {n_nodes} nodes share {n_texts} distinct embed texts "
        f"({n_nodes / max(n_texts, 1):.1f} nodes per text)"
    )

    if args.retrieve:
        from ikgqa.retrieval import PCST, assert_valid

        # The question must live in the same vector space as the graph, which
        # default_encoder guarantees by construction. The assert keeps a future
        # change from producing plausible but meaningless rankings in silence.
        encoder = default_encoder(table["embed_text"], graph.edges["edge_attr"])
        q_emb = encoder.encode_one(args.retrieve)
        assert q_emb.shape[0] == graph.node_emb.shape[1], (
            f"question dim {q_emb.shape[0]} != graph dim {graph.node_emb.shape[1]}; "
            "the default encoder in build_patient_graph no longer matches this one"
        )

        # topk_e=0 deliberately: with few, heavily shared relation types the
        # edge-prize mechanism collapses the effective edge cost towards zero.
        selection = PCST(topk=3, topk_e=0, cost_e=0.5).retrieve(graph, q_emb)
        assert_valid(selection, graph)
        print(
            f"\nPCST for {args.retrieve!r}: {selection.num_nodes} nodes, "
            f"{selection.num_edges} edges"
        )
        print(table.loc[selection.node_ids, "sphn_label"].value_counts().to_string())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
