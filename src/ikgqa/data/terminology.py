"""
ikgqa.data.terminology
======================

Resolving terminology codes to descriptions, so that nodes carrying only a code
can be found by asking a question in words.

The problem, measured
---------------------
For patient #0, 178 of 222 diagnoses (80%) carry no description in the graph at
all -- only an ICD-10-GM code. Those nodes are unreachable by similarity under
any wording of any question. Graph-wide, 1,949 of 2,921 distinct diagnosis codes
have no description. Meanwhile every laboratory code has an English LOINC
description, so the graph's text is both incomplete *and* bilingual.

Why this is a cross-walk and not a translation
----------------------------------------------
A terminology code is language-independent by construction: ``N18.5`` denotes the
same concept whichever language its label is written in. So the operation needed
is a lookup against a published catalogue, not machine translation. That matters
for three reasons a thesis has to care about:

* it is deterministic, so a rerun a year later gives the same numbers;
* it is auditable, because every resolution names the catalogue entry it came
  from and a clinician can check a list of 2,921 rows;
* it fills codes that have *no* label at all, which translation cannot do -- you
  cannot translate a string that does not exist.

Nothing here modifies the database. The graph is read-only, and the resolved
description is written into a derived text layer on the way into the retrieval
pipeline, leaving the source record untouched.

The tiers
---------
Resolution is tried in order, and *which tier answered* is recorded for every
node so coverage can be reported rather than assumed:

``exact``
    The code and its coding-system version are both in the catalogue.
``version``
    The code is in the catalogue under a *different* version of the same system.
    ICD-10-GM is reissued annually and a code's meaning is stable far more often
    than it changes, so this is a good trade -- but it is a distinct tier
    precisely because it is an assumption, and the count of how often it fires
    belongs in the results.
``parent``
    No entry for the code, but there is one for its parent concept: ``N18.5``
    falls back to ``N18``. The label is broader than the node deserves, which is
    why it is separated out. Note this tier is not only a fallback: adding the
    parent's words to a leaf that *does* resolve widens the surface a question
    can match, which is the schema-signal idea of RQ3.
``local``
    Nothing in the catalogue; keep whatever label the graph itself carried.
``none``
    No catalogue entry and no local label. The node keeps its bare code and is
    counted as unreachable by similarity.

Bootstrapping without any external download
-------------------------------------------
``Terminology.from_graph_labels`` builds a catalogue out of the labels the graph
*already* carries. Because ICD-10-GM is versioned by year and the graph contains
thirteen versions, a code described in ``10-GM-2022`` can fill the same code
appearing bare under ``10-GM-2018`` through the ``version`` tier. That needs no
network access, no licence, and no trust in an outside file -- and it establishes
a floor that an external catalogue then has to beat.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Tier names. Strings rather than an enum because they are written to CSV
# results and read back by the thesis's plotting code.
TIER_EXACT = "exact"
TIER_VERSION = "version"
TIER_PARENT = "parent"
TIER_LOCAL = "local"
TIER_NONE = "none"

TIERS: Tuple[str, ...] = (TIER_EXACT, TIER_VERSION, TIER_PARENT, TIER_LOCAL, TIER_NONE)

CSV_FIELDS = ("system", "code", "label", "parent_code", "parent_label", "language")


@dataclasses.dataclass(frozen=True)
class Concept:
    """One catalogue entry.

    Args:
        system: coding system *with* version where the system is versioned,
            e.g. "10-GM-2022". The version is part of the identity: the same
            code string can denote different concepts in different years.
        code: the code as it appears in the graph, e.g. "N18.5".
        label: the canonical description.
        parent_code: the code one level up the hierarchy, if known.
        parent_label: its description, used by the parent tier and by RQ3's
            hierarchy expansion.
        language: ISO 639-1 code for ``label``. Recorded, not enforced: an
            experiment comparing an English-only against a multilingual encoder
            needs to know which language each description is in.
    """

    system: str
    code: str
    label: str
    parent_code: str = ""
    parent_label: str = ""
    language: str = ""

    @property
    def family(self) -> str:
        """The system without its version, e.g. "10-GM-2022" -> "10-GM".

        Used by the version tier. LOINC and ATC are unversioned in this graph
        and are their own family.
        """
        return system_family(self.system)


@dataclasses.dataclass(frozen=True)
class Resolution:
    """The outcome of one lookup, including which tier answered it."""

    tier: str
    label: str
    parent_label: str = ""
    language: str = ""
    source_system: str = ""

    @property
    def resolved(self) -> bool:
        """True when a *catalogue* answered. ``local`` is not a resolution."""
        return self.tier in (TIER_EXACT, TIER_VERSION, TIER_PARENT)

    @property
    def reachable(self) -> bool:
        """True when the node ends up with words on it, from wherever."""
        return self.tier != TIER_NONE


def system_family(system: str) -> str:
    """Strip a trailing year from a coding-system name.

    "10-GM-2022" -> "10-GM"; "LOINC" -> "LOINC". Deliberately narrow: only a
    four-digit trailing component is treated as a version, so a system whose
    name merely ends in digits is left alone.
    """
    text = (system or "").strip()
    head, _, tail = text.rpartition("-")
    if head and len(tail) == 4 and tail.isdigit():
        return head
    return text


def parent_code_of(code: str) -> str:
    """The obvious parent of a hierarchical code, or "" if there is none.

    ICD-10 is hierarchical by string prefix: N18.5 sits under N18, which sits
    under the N18-N19 block. Only the first step is taken here, because that is
    the one that is unambiguous from the code alone. A catalogue that ships real
    parent codes should be preferred over this, which is why
    ``Terminology.resolve`` uses a catalogue's ``parent_code`` when present and
    only falls back to this.
    """
    text = (code or "").strip()
    if "." in text:
        head = text.split(".", 1)[0]
        return head if head else ""
    return ""


class Terminology:
    """A code catalogue with tiered lookup.

    Indexes are built once at construction. Lookup is a dict hit, so resolving
    every node of a 15,810-node graph costs microseconds -- the expensive part of
    this pipeline is embedding, and this step *reduces* that cost by making more
    texts identical.
    """

    def __init__(self, concepts: Iterable[Concept] = ()):
        self._by_exact: Dict[Tuple[str, str], Concept] = {}
        self._by_family: Dict[Tuple[str, str], Concept] = {}
        for concept in concepts:
            if not concept.code or not concept.label:
                continue
            self._by_exact.setdefault((concept.system, concept.code), concept)
            # First writer wins, so a deterministic input gives a deterministic
            # index even where versions disagree. Feed concepts newest-first if
            # a preference is wanted.
            self._by_family.setdefault((concept.family, concept.code), concept)

    def __len__(self) -> int:
        return len(self._by_exact)

    def __repr__(self) -> str:
        families = sorted({family for family, _ in self._by_family})
        return f"Terminology({len(self)} concepts; families={families})"

    # -- construction --------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Any) -> "Terminology":
        """Load a catalogue from CSV with the columns in ``CSV_FIELDS``.

        Unknown columns are ignored and missing optional columns default to
        empty, so a catalogue exported from somewhere else usually loads as-is.
        """
        path = pathlib.Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"system", "code", "label"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing required column(s) {sorted(missing)}; "
                    f"expected at least system,code,label"
                )
            concepts = [
                Concept(
                    system=(row.get("system") or "").strip(),
                    code=(row.get("code") or "").strip(),
                    label=(row.get("label") or "").strip(),
                    parent_code=(row.get("parent_code") or "").strip(),
                    parent_label=(row.get("parent_label") or "").strip(),
                    language=(row.get("language") or "").strip(),
                )
                for row in reader
            ]
        return cls(concepts)

    @classmethod
    def from_graph_labels(
        cls, rows: Sequence[Dict[str, Any]], code_key: str, system_key: str, name_key: str
    ) -> "Terminology":
        """Harvest a catalogue from the labels the graph already carries.

        This is the no-download bootstrap described in the module docstring. It
        buys coverage in exactly one way -- a code labelled under one version of
        a system fills the same code appearing bare under another -- and it is
        worth doing first because it needs nothing external and establishes the
        floor an external catalogue has to beat.
        """
        seen: Dict[Tuple[str, str], Concept] = {}
        for row in rows:
            code = str(row.get(code_key) or "").strip()
            label = str(row.get(name_key) or "").strip()
            system = str(row.get(system_key) or "").strip()
            if not code or not label or label.lower() in {"none", "null", "nan"}:
                continue
            seen.setdefault((system, code), Concept(system, code, label))
        return cls(seen.values())

    def to_csv(self, path: Any) -> int:
        """Write the catalogue out, sorted, so diffs are reviewable."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        concepts = sorted(self._by_exact.values(), key=lambda c: (c.system, c.code))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for concept in concepts:
                writer.writerow(dataclasses.asdict(concept))
        return len(concepts)

    # -- lookup --------------------------------------------------------------

    def resolve(self, code: str, system: str, local_label: str = "") -> Resolution:
        """Look up one code, trying each tier in order.

        Args:
            code: the code as the graph carries it.
            system: the coding system and version as the graph carries it.
            local_label: the description the graph already had, if any. Used by
                the ``local`` tier and never overwritten by a lower tier.

        Returns:
            A Resolution naming the tier that answered, so coverage can be
            counted per tier instead of reported as one number.
        """
        code = (code or "").strip()
        system = (system or "").strip()
        local_label = (local_label or "").strip()

        hit = self._by_exact.get((system, code))
        if hit is not None:
            return Resolution(
                TIER_EXACT, hit.label, self._parent_label(hit), hit.language, hit.system
            )

        hit = self._by_family.get((system_family(system), code))
        if hit is not None:
            return Resolution(
                TIER_VERSION, hit.label, self._parent_label(hit), hit.language, hit.system
            )

        parent = parent_code_of(code)
        if parent:
            for key in ((system, parent), (system_family(system), parent)):
                hit = self._by_exact.get(key) or self._by_family.get(key)
                if hit is not None:
                    return Resolution(
                        TIER_PARENT, hit.label, hit.parent_label, hit.language, hit.system
                    )

        if local_label:
            return Resolution(TIER_LOCAL, local_label, "", "", system)
        return Resolution(TIER_NONE, "", "", "", system)

    def _parent_label(self, concept: Concept) -> str:
        """A concept's parent description, from the catalogue or by prefix."""
        if concept.parent_label:
            return concept.parent_label
        parent = concept.parent_code or parent_code_of(concept.code)
        if not parent:
            return ""
        hit = self._by_exact.get((concept.system, parent)) or self._by_family.get(
            (concept.family, parent)
        )
        return hit.label if hit is not None else ""


# ---------------------------------------------------------------------------
# Applying a catalogue to fetched rows
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EnrichmentReport:
    """Per-tier counts for one enrichment pass. This is a results table."""

    what: str
    tiers: Dict[str, int] = dataclasses.field(
        default_factory=lambda: {tier: 0 for tier in TIERS}
    )
    hierarchy_expanded: int = 0

    @property
    def total(self) -> int:
        return sum(self.tiers.values())

    @property
    def reachable(self) -> int:
        """Rows that ended up with words on them."""
        return self.total - self.tiers[TIER_NONE]

    @property
    def resolved_by_catalogue(self) -> int:
        return sum(self.tiers[t] for t in (TIER_EXACT, TIER_VERSION, TIER_PARENT))

    def summary(self) -> str:
        if not self.total:
            return f"{self.what}: nothing to enrich"
        lines = [f"{self.what}: {self.total} rows"]
        for tier in TIERS:
            count = self.tiers[tier]
            if count:
                lines.append(f"  {tier:<8} {count:>7}  ({100 * count / self.total:5.1f}%)")
        lines.append(
            f"  reachable by similarity: {self.reachable}/{self.total} "
            f"({100 * self.reachable / self.total:.1f}%)"
        )
        if self.hierarchy_expanded:
            lines.append(f"  hierarchy text added to {self.hierarchy_expanded} rows")
        return "\n".join(lines)


def enrich_diagnosis_rows(
    rows: Sequence[Dict[str, Any]],
    terminology: Terminology,
    expand_hierarchy: bool = False,
) -> Tuple[List[Dict[str, Any]], EnrichmentReport]:
    """Add resolved descriptions to diagnosis rows, non-destructively.

    Returns new row dicts; the input is not modified, and neither is the
    database. Two keys are written, which is what keeps the two audiences apart:

    ``embed_name``
        what the encoder should see -- the canonical description, optionally
        widened with the parent concept's words.
    ``diagnosis_name``
        left exactly as the graph had it, so ``display_text`` still shows a
        clinician the label their system recorded (or the bare code if there
        was none).

    Args:
        expand_hierarchy: append the parent concept's description to
            ``embed_name``. This is the RQ3 schema signal in its simplest form:
            it widens what a question can match without inventing content.

    Also written: ``resolution_tier``, so a result table can group by it.
    """
    report = EnrichmentReport(what="diagnoses")
    out: List[Dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        code = str(row.get("icd_code") or "").strip()
        system = str(row.get("code_system") or "").strip()
        local = str(row.get("diagnosis_name") or "").strip()
        if local.lower() in {"none", "null", "nan"}:
            local = ""

        resolution = terminology.resolve(code, system, local_label=local)
        report.tiers[resolution.tier] += 1

        embed_name = resolution.label
        if expand_hierarchy and resolution.parent_label and embed_name:
            embed_name = f"{embed_name}; {resolution.parent_label}"
            report.hierarchy_expanded += 1

        new["embed_name"] = embed_name
        new["resolution_tier"] = resolution.tier
        out.append(new)
    return out, report


def enrich_rows(
    rows: Dict[str, Sequence[Dict[str, Any]]],
    terminology: Terminology,
    expand_hierarchy: bool = False,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, EnrichmentReport]]:
    """Enrich every row group a catalogue can help with.

    Only diagnoses are enriched today, because laboratory codes already carry a
    description on 100% of their 4.8 million uses -- there is nothing to fill.
    The signature is plural so that ATC drug codes, whose coverage has not been
    measured yet, can be added without changing any caller.
    """
    enriched = {key: list(group) for key, group in rows.items()}
    reports: Dict[str, EnrichmentReport] = {}
    diagnoses, report = enrich_diagnosis_rows(
        rows.get("diagnoses", []), terminology, expand_hierarchy=expand_hierarchy
    )
    enriched["diagnoses"] = diagnoses
    reports["diagnoses"] = report
    return enriched, reports


def bootstrap_from_rows(
    rows: Dict[str, Sequence[Dict[str, Any]]]
) -> Terminology:
    """Build the no-download catalogue from a fetch's own diagnosis labels."""
    return Terminology.from_graph_labels(
        rows.get("diagnoses", []),
        code_key="icd_code",
        system_key="code_system",
        name_key="diagnosis_name",
    )
