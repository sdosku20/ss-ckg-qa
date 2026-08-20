"""
ikgqa.data.replica
==================

A synthetic patient graph that reproduces the *measured* shape of the real STCS
graph, so that every piece of the pipeline can be developed and evaluated
without server access.

Why this exists
---------------
The clinical graph lives on BioMedIT and cannot leave it. Access to it is also
intermittent. Everything downstream of the loader -- the recall-vs-size sweep,
the terminology cross-walk, the ablations -- needs a graph to run against, and a
toy graph of twenty nodes does not exhibit the properties that make the real one
hard.

How fidelity is achieved
------------------------
Two decisions do the work:

1. The generator produces *database rows in the exact shape the Cypher returns*,
   and then hands them to ``ikgqa.data.sphn.build_patient_graph``. The structure
   is therefore identical to the real thing by construction rather than by
   imitation, and the replica exercises the real assembly code.

2. The counts come from one measured patient (``PATIENT_0``), not from
   guesses. Where a measured value is a target the generator cannot hit exactly,
   ``describe_fidelity`` reports measured against achieved rather than hiding the
   gap.

What is synthetic
-----------------
Every string. Analyte names are composed from a LOINC-style vocabulary,
diagnosis labels are generic German-morphology placeholders, and drug names are
invented. No value in this module came from a patient record. The German labels
are not translations of the codes they sit next to: they exist so that an
English-only encoder is measurably disadvantaged on the diagnosis portion of the
graph, which is the property the terminology work has to fix.

What NOT to do with it
----------------------
Do not report recall numbers from this graph as results about the STCS cohort.
It is faithful in *shape*, not in content: its similarity structure comes from a
synthetic vocabulary. It is valid for testing mechanism (does the size dial
work, does tie-breaking decide the outcome, does the cross-walk change coverage)
and invalid for claims about clinical retrieval quality.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ikgqa.data.sphn import build_patient_graph
from ikgqa.eval.metrics import KIND_ENTITY, Question
from ikgqa.graph import TextualGraph

# ---------------------------------------------------------------------------
# The measured profile
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PatientProfile:
    """Counts measured on one real patient. See docs/server_measurements.md.

    Every field is an aggregate. None of them identifies anybody, which is why
    they may live in the repository at all.
    """

    label: str
    lab_observations: int
    analytes: int
    diagnoses: int
    unnamed_diagnoses: int
    drug_administrations: int
    drugs: int
    cases: int
    distinct_embed_texts: int
    nodes: int
    edges: int

    @property
    def measurements_per_analyte(self) -> float:
        return self.lab_observations / max(self.analytes, 1)

    @property
    def nodes_per_distinct_text(self) -> float:
        return self.nodes / max(self.distinct_embed_texts, 1)


# Measured 20 August 2026 on the CHIL server with
#   python -m ikgqa.data.sphn --patient 0
# Recorded here so offline work has a target and the thesis has one source of
# truth for these numbers.
PATIENT_0 = PatientProfile(
    label="stcs-patient-0",
    lab_observations=12_349,
    analytes=198,
    diagnoses=222,
    unnamed_diagnoses=178,
    drug_administrations=2_655,
    drugs=279,
    cases=106,
    distinct_embed_texts=614,
    nodes=15_810,
    edges=45_562,
)

# A tenth of the above, for fast tests and for sweeps where a full patient is
# slower than it needs to be. Ratios are preserved; absolute size is not.
PATIENT_0_SMALL = PatientProfile(
    label="stcs-patient-0-small",
    lab_observations=1_235,
    analytes=60,
    diagnoses=40,
    unnamed_diagnoses=32,
    drug_administrations=265,
    drugs=45,
    cases=12,
    distinct_embed_texts=0,  # not a target at this size
    nodes=0,
    edges=0,
)


# ---------------------------------------------------------------------------
# Synthetic vocabularies
# ---------------------------------------------------------------------------

# LOINC descriptions have the shape "<analyte> [<property>] in <specimen>".
# Reproducing that shape matters: it is why real analyte texts are long, share
# most of their tokens, and are therefore hard for a bag-of-words encoder to
# separate.
_ANALYTES = (
    "Creatinine", "Urea", "Sodium", "Potassium", "Chloride", "Calcium",
    "Phosphate", "Magnesium", "Glucose", "Albumin", "Bilirubin.total",
    "Bilirubin.direct", "Alanine aminotransferase", "Aspartate aminotransferase",
    "Alkaline phosphatase", "Gamma glutamyl transferase", "Lactate dehydrogenase",
    "C reactive protein", "Hemoglobin", "Hematocrit", "Erythrocytes",
    "Leukocytes", "Platelets", "Neutrophils", "Lymphocytes", "Monocytes",
    "Eosinophils", "Reticulocytes", "Prothrombin time", "Fibrinogen",
    "D dimer", "Troponin T", "Creatine kinase", "Cholesterol", "Triglyceride",
    "Thyrotropin", "Cortisol", "Tacrolimus", "Ciclosporin", "Mycophenolate",
    "Cytomegalovirus DNA", "Epstein Barr virus DNA", "BK virus DNA",
    "Protein", "Osmolality",
)
_PROPERTIES = (
    "Mass/volume", "Moles/volume", "Enzymatic activity/volume",
    "Count/volume", "Ratio",
)
_SPECIMENS = ("Serum or Plasma", "Blood", "Urine", "Plasma")

_UNITS = ("mg/dL", "mmol/L", "U/L", "g/L", "10*9/L", "ug/L", "%")

_SUBSTANCES = (
    "tacrolimus", "ciclosporin", "mycophenolate mofetil", "prednisolone",
    "basiliximab", "valganciclovir", "cotrimoxazole", "amoxicillin",
    "piperacillin", "meropenem", "vancomycin", "fluconazole", "acyclovir",
    "pantoprazole", "furosemide", "torasemide", "amlodipine", "metoprolol",
    "lisinopril", "candesartan", "atorvastatin", "insulin glargine",
    "metformin", "levothyroxine", "calcium carbonate", "cholecalciferol",
    "magnesium oxide", "sodium bicarbonate", "paracetamol", "metamizole",
    "morphine", "ondansetron", "heparin", "enoxaparin", "acetylsalicylic acid",
    "erythropoietin", "filgrastim", "iron sucrose", "epoetin", "rituximab",
)

# German-morphology placeholders. NOT translations of any code: they exist only
# so that a German-language portion of the graph exists to be measured against.
_DE_STEMS = (
    "Chronische Erkrankung", "Akute Entzuendung", "Sekundaere Stoerung",
    "Nicht naeher bezeichnete Komplikation", "Transplantatabstossung",
    "Nierenfunktionsstoerung", "Bluthochdruck", "Zuckerkrankheit",
    "Blutarmut", "Infektion des Harntrakts", "Lungenentzuendung",
    "Herzinsuffizienz",
)
_DE_QUALIFIERS = (
    "Stadium 1", "Stadium 2", "Stadium 3", "Stadium 4", "Stadium 5",
    "ohne Angabe", "mit Komplikationen", "nicht naeher bezeichnet",
)

_ICD_LETTERS = "ABCDEFGIJKLMNQRSTZ"


def _analyte_names(n: int) -> List[str]:
    """n distinct LOINC-shaped descriptions, deterministically."""
    out: List[str] = []
    for prop in _PROPERTIES:
        for spec in _SPECIMENS:
            for analyte in _ANALYTES:
                out.append(f"{analyte} [{prop}] in {spec}")
                if len(out) == n:
                    return out
    raise ValueError(f"vocabulary exhausted at {len(out)} names, need {n}")


def _drug_names(n: int) -> List[Tuple[str, str]]:
    """n distinct (substance, article) pairs."""
    out: List[Tuple[str, str]] = []
    for strength in (1, 2, 5, 10, 20, 40, 100, 250, 500):
        for sub in _SUBSTANCES:
            out.append((sub, f"{sub.split()[0].capitalize()} {strength}mg"))
            if len(out) == n:
                return out
    raise ValueError(f"vocabulary exhausted at {len(out)} drugs, need {n}")


def _icd_codes(n: int, rng: random.Random) -> List[str]:
    """n distinct ICD-10-GM-shaped codes."""
    seen: List[str] = []
    used = set()
    while len(seen) < n:
        code = f"{rng.choice(_ICD_LETTERS)}{rng.randrange(0, 100):02d}"
        if rng.random() < 0.6:
            code += f".{rng.randrange(0, 10)}"
        if code not in used:
            used.add(code)
            seen.append(code)
    return seen


def _zipf_allocation(total: int, buckets: int) -> List[int]:
    """Split ``total`` over ``buckets`` with a 1/rank weighting.

    Real laboratory activity is heavily skewed: a handful of analytes are
    measured daily and the tail is measured once. A uniform split would make
    tie-breaking look less decisive than it is, so the skew is reproduced.
    """
    weights = [1.0 / (i + 1) for i in range(buckets)]
    scale = total / sum(weights)
    counts = [max(1, int(w * scale)) for w in weights]
    # Repair rounding drift on the largest bucket, which can absorb it.
    drift = total - sum(counts)
    counts[0] = max(1, counts[0] + drift)
    return counts


def _stamp(day: int, hour: int = 7) -> str:
    """A deterministic ISO timestamp. No wall clock, so runs are reproducible."""
    year = 2019 + day // 365
    remainder = day % 365
    month = remainder // 30 + 1
    dom = remainder % 30 + 1
    return f"{year:04d}-{month:02d}-{dom:02d}T{hour:02d}:00:00"


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------


def generate_rows(
    profile: PatientProfile = PATIENT_0, seed: int = 0
) -> Dict[str, List[Dict[str, Any]]]:
    """Rows in exactly the shape ``ikgqa.data.sphn``'s Cypher returns.

    Returning rows rather than a graph is what keeps the replica honest: the
    same assembly code runs on real and synthetic input, so a bug in assembly
    cannot hide here and appear only on the server.
    """
    rng = random.Random(seed)

    cases = [
        {
            "case_id": f"case-{i}",
            "admitted_at": _stamp(i * 14, 8),
            "discharged_at": _stamp(i * 14 + 3, 11),
        }
        for i in range(profile.cases)
    ]
    case_ids = [c["case_id"] for c in cases] or [None]

    # -- laboratory: few analytes, many measurements, skewed ------------------
    names = _analyte_names(profile.analytes)
    per_analyte = _zipf_allocation(profile.lab_observations, profile.analytes)
    labs: List[Dict[str, Any]] = []
    for a_idx, (name, count) in enumerate(zip(names, per_analyte)):
        code = f"{1000 + a_idx * 7}-{a_idx % 10}"
        unit = _UNITS[a_idx % len(_UNITS)]
        for j in range(count):
            if len(labs) >= profile.lab_observations:
                break
            labs.append(
                {
                    "event_uid": f"ev-{len(labs)}",
                    "observed_at": _stamp(len(labs) % 1800, 6 + j % 12),
                    "loinc_code": code,
                    "code_system": "LOINC",
                    "test_name": name,
                    "numeric_value": round(rng.uniform(0.1, 400.0), 2),
                    "unit": unit,
                    "text_value": None,
                    "reference_range": "0.6-1.2" if a_idx % 3 == 0 else None,
                    "case_id": case_ids[len(labs) % len(case_ids)],
                    "material": _SPECIMENS[a_idx % len(_SPECIMENS)],
                }
            )

    # -- diagnoses: most carrying no description at all -----------------------
    codes = _icd_codes(profile.diagnoses, rng)
    named_from = profile.unnamed_diagnoses  # the first N are the unnamed ones
    diagnoses: List[Dict[str, Any]] = []
    for i, code in enumerate(codes):
        stem = _DE_STEMS[i % len(_DE_STEMS)]
        qual = _DE_QUALIFIERS[i % len(_DE_QUALIFIERS)]
        true_name = f"{stem}, {qual}"
        has_name = i >= named_from
        diagnoses.append(
            {
                "recorded_at": _stamp(i * 5, 0),
                "icd_code": code,
                "code_system": f"10-GM-{2012 + i % 13}",
                "diagnosis_name": true_name if has_name else None,
                "case_id": case_ids[i % len(case_ids)],
                # Every code has a description in the world; only some have one
                # *in the graph*. Keeping the withheld description here is what
                # makes the terminology gap measurable: a question can be
                # phrased in the words a clinician would use, and a code-only
                # node stays unreachable until a cross-walk supplies them.
                # Ignored by the text composers, which read diagnosis_name only.
                "_true_name": true_name,
            }
        )

    # -- drug administrations: repeated doses of few drugs --------------------
    drugs_vocab = _drug_names(profile.drugs)
    per_drug = _zipf_allocation(profile.drug_administrations, profile.drugs)
    drugs: List[Dict[str, Any]] = []
    for d_idx, ((sub, article), count) in enumerate(zip(drugs_vocab, per_drug)):
        for j in range(count):
            if len(drugs) >= profile.drug_administrations:
                break
            drugs.append(
                {
                    "started_at": _stamp(len(drugs) % 1800, 8),
                    "ended_at": _stamp(len(drugs) % 1800 + 7, 8),
                    "drug_key": f"drug-{d_idx}",
                    "substance": sub,
                    "article": article,
                    "dose_value": float((d_idx % 8 + 1) * 5),
                    "dose_unit": "mg",
                    "case_id": case_ids[(d_idx + j) % len(case_ids)],
                }
            )

    return {"cases": cases, "labs": labs, "diagnoses": diagnoses, "drugs": drugs}


def build_replica(
    profile: PatientProfile = PATIENT_0,
    encoder: Any = None,
    seed: int = 0,
) -> Tuple[TextualGraph, Any, Any, Dict[str, List[Dict[str, Any]]]]:
    """Rows -> the real assembly code -> (graph, node table, stats, rows).

    The rows come back too because ``planted_questions`` needs the descriptions
    that were deliberately withheld from the graph.
    """
    rows = generate_rows(profile, seed=seed)
    graph, table, stats = build_patient_graph(
        rows, encoder, patient_index=0, name=f"{profile.label}-replica"
    )
    return graph, table, stats, rows


def describe_fidelity(profile: PatientProfile, stats: Any, graph: TextualGraph) -> str:
    """Measured against achieved, side by side.

    Printed by every experiment that uses the replica. A replica whose gap is
    not visible is a replica that will eventually be mistaken for the real
    thing.
    """
    rows = [
        ("nodes", profile.nodes, graph.num_nodes),
        ("edges", profile.edges, graph.num_edges),
        ("lab observations", profile.lab_observations, stats.labs),
        ("analytes", profile.analytes, stats.lab_types),
        ("diagnoses", profile.diagnoses, stats.diagnoses),
        ("unnamed diagnoses", profile.unnamed_diagnoses, stats.unnamed_diagnoses),
        ("drug administrations", profile.drug_administrations, stats.drug_administrations),
        ("drugs", profile.drugs, stats.drugs),
        ("cases", profile.cases, stats.cases),
        ("distinct embed texts", profile.distinct_embed_texts, stats.distinct_embed_texts),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = [f"fidelity of {profile.label}-replica (measured -> achieved)"]
    for name, measured, achieved in rows:
        if not measured:
            lines.append(f"  {name:<{width}}  {'(not a target)':>14}  {achieved:>8}")
            continue
        delta = achieved - measured
        flag = "" if delta == 0 else f"  ({delta:+d})"
        lines.append(f"  {name:<{width}}  {measured:>14}  {achieved:>8}{flag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planted questions
# ---------------------------------------------------------------------------


def planted_questions(
    graph: TextualGraph,
    table: Any,
    rows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    n_per_kind: int = 5,
) -> List[Question]:
    """Questions whose answer nodes are known because we planted them.

    Three kinds, chosen to probe three different failure modes:

    * analyte questions -- the answer is every observation of one analyte, so
      recall exposes whether tie-breaking picks arbitrary members of a large
      identically-worded group.
    * drug questions -- the answer is every administration of one drug, testing
      the same thing through a different label.
    * diagnosis questions -- half target *named* diagnoses and half target
      code-only ones, which is the pair that makes the terminology gap
      measurable: the second half is unreachable by similarity until codes are
      resolved.

    These are synthetic and carry ``validated=False``. They test the harness and
    the mechanism, never clinical quality.
    """
    questions: List[Question] = []
    is_label = lambda lab: table.index[table["sphn_label"] == lab]  # noqa: E731

    obs = table.loc[is_label("LabObservation")]
    types = table.loc[is_label("LabTestType")]
    # Pick the analytes with the most observations: the hard cases, not the easy
    # singletons, because a singleton says nothing about tie-breaking.
    by_text = obs.groupby("embed_text").size().sort_values(ascending=False)
    for i, text in enumerate(by_text.index[:n_per_kind]):
        answer = sorted(obs.index[obs["embed_text"] == text].tolist())
        type_hits = types.index[
            types["embed_text"].apply(lambda t, s=text: _shares_analyte(t, s))
        ].tolist()
        questions.append(
            Question(
                text=_question_from_text(text),
                answer_nodes=tuple(answer + sorted(type_hits)),
                answer_edges=(),
                qid=f"analyte-{i}",
                kind=KIND_ENTITY,
                validated=False,
            )
        )

    admins = table.loc[is_label("DrugAdministrationEvent")]
    by_drug = admins.groupby("embed_text").size().sort_values(ascending=False)
    for i, text in enumerate(by_drug.index[:n_per_kind]):
        answer = sorted(admins.index[admins["embed_text"] == text].tolist())
        questions.append(
            Question(
                text=_question_from_text(text),
                answer_nodes=tuple(answer),
                answer_edges=(),
                qid=f"drug-{i}",
                kind=KIND_ENTITY,
                validated=False,
            )
        )

    # Diagnoses. Both halves are asked in *clinical words*, never by code: a
    # question that quotes the code is trivially answerable by string overlap
    # and measures nothing. The named half is therefore reachable and the
    # code-only half is not, and the distance between the two is exactly what
    # the terminology cross-walk has to close.
    if rows is not None:
        dx = table.loc[is_label("BilledDiagnosis")]
        by_code = {}
        for idx in dx.index:
            display = str(dx.at[idx, "display_text"])
            by_code.setdefault(_code_key_in(display), idx)

        dx_rows = rows.get("diagnoses", [])
        named_rows = [r for r in dx_rows if r.get("diagnosis_name")]
        codeonly_rows = [r for r in dx_rows if not r.get("diagnosis_name")]
        half = max(1, n_per_kind // 2)
        for qid_prefix, subset in (("dx-named", named_rows), ("dx-codeonly", codeonly_rows)):
            for i, row in enumerate(subset[:half]):
                key = f"{row['code_system']} {row['icd_code']}"
                idx = by_code.get(key)
                if idx is None:
                    continue
                questions.append(
                    Question(
                        # The withheld description, for both halves, so the two
                        # differ only in what the graph knows -- not in how the
                        # question is phrased.
                        text=str(row["_true_name"]).lower(),
                        answer_nodes=(int(idx),),
                        answer_edges=(),
                        qid=f"{qid_prefix}-{i}",
                        kind=KIND_ENTITY,
                        validated=False,
                    )
                )
    return questions


def _code_key_in(display_text: str) -> str:
    """Pull the "10-GM-YYYY CODE" pair out of a composed diagnosis text.

    Matching on the versioned pair rather than the bare code avoids the
    substring trap, where looking for B28 would also match B28.4.
    """
    for part in display_text.split(";"):
        token = part.strip()
        if token.startswith("10-GM-"):
            return token
    return display_text


def _shares_analyte(type_text: str, obs_text: str) -> bool:
    """True when a test-type text and an observation text name the same analyte."""
    a = set(type_text.lower().split())
    b = set(obs_text.lower().split())
    overlap = a & b
    return len(overlap) >= 3


def _question_from_text(embed_text: str) -> str:
    """Turn a node's text into something question-shaped.

    Deliberately lossy: a real question is not the node's text, and a retriever
    that only works when the question repeats the text verbatim is not being
    tested at all. Dropping the bracketed LOINC property and the code keeps the
    words a person would actually type.
    """
    cleaned = embed_text.split("[")[0].strip()
    cleaned = cleaned.replace("Lab test:", "").replace("Diagnosis:", "")
    cleaned = cleaned.replace("Drug:", "").strip(" ,:")
    return cleaned.lower()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the replica's shape and its fidelity against the measured profile."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ikgqa.data.replica")
    parser.add_argument("--small", action="store_true", help="use the 10x smaller profile")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    profile = PATIENT_0_SMALL if args.small else PATIENT_0
    graph, table, stats, rows = build_replica(profile, seed=args.seed)
    print(describe_fidelity(profile, stats, graph))
    print()
    print(graph)
    print(table["sphn_label"].value_counts().to_string())
    questions = planted_questions(graph, table, rows)
    print(f"\n{len(questions)} planted questions, e.g.")
    for q in questions[:3]:
        print(f"  {q.qid:16s} {q.text[:48]!r} -> {len(q.answer_nodes)} answer nodes")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
