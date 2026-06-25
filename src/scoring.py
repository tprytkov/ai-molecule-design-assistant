"""Calculate novelty-aware compound prioritization scores offline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DESCRIPTOR_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "qed",
    "lipinski_violations",
    "lipinski_pass",
)
SIMILARITY_COLUMNS = (
    "molecule_id",
    "best_reference_name",
    "tanimoto_similarity",
    "similarity_category",
)
NLP_COLUMNS = (
    "molecule_id",
    "max_relevance_score",
)
PUBLIC_LOOKUP_COLUMNS = (
    "molecule_id",
    "source_database",
    "match_type",
    "public_id",
    "similarity",
    "lookup_status",
)
PATENT_COLUMNS = (
    "molecule_id",
    "search_status",
)
SURECHEMBL_COLUMNS = (
    "molecule_id",
    "tanimoto_similarity",
    "lookup_status",
)
IDENTITY_COLUMNS = (
    "molecule_id",
    "identity_status",
    "lookup_status",
)
CONTEXT_COLUMNS = (
    "molecule_id",
    "context_status",
)
CHEMBERTA_COLUMNS = (
    "molecule_id",
    "embedding_available",
)
EVIDENCE_STATUS_COLUMNS = (
    "chemical_identity_status",
    "chemical_identity_lookup_status",
    "public_lookup_status",
    "pubchem_status",
    "chembl_status",
    "nlp_status",
    "surechembl_query_status",
    "chemberta_status",
    "context_status",
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "qed",
    "lipinski_violations",
    "lipinski_pass",
    "best_reference_name",
    "tanimoto_similarity",
    "similarity_category",
    "validity_score",
    "property_score",
    "qed_score",
    "lipinski_score",
    "differentiation_score",
    "prioritization_score",
    "prioritization_category",
    "scoring_notes",
    *EVIDENCE_STATUS_COLUMNS,
)
NLP_OUTPUT_COLUMNS = OUTPUT_COLUMNS + (
    "nlp_evidence_score",
    "nlp_relevance_category",
    "nlp_evidence_count",
    "prioritization_score_with_nlp",
    "prioritization_category_with_nlp",
)
PUBLIC_OUTPUT_COLUMNS = NLP_OUTPUT_COLUMNS + (
    "known_public_match",
    "known_public_match_source",
    "known_public_match_id",
    "novelty_flag",
    "ip_potential_category",
    "ip_potential_notes",
)
PATENT_OUTPUT_COLUMNS = PUBLIC_OUTPUT_COLUMNS + (
    "patent_evidence_count",
    "patent_max_similarity",
    "patent_relevance_score",
    "patent_signal_category",
    "patent_signal_notes",
)
SURECHEMBL_OUTPUT_COLUMNS = PUBLIC_OUTPUT_COLUMNS + (
    "surechembl_evidence_count",
    "surechembl_max_similarity",
    "surechembl_signal_category",
    "surechembl_signal_notes",
)

SCORE_WEIGHTS = {
    "validity": 0.20,
    "property": 0.25,
    "qed": 0.20,
    "lipinski": 0.15,
    "differentiation": 0.20,
}


@dataclass(frozen=True)
class PrioritizationResult:
    """Scoring result for one generated molecule."""

    molecule_id: str
    canonical_smiles: str
    valid_smiles: bool
    molecular_weight: str
    logp: str
    tpsa: str
    hbd: str
    hba: str
    rotatable_bonds: str
    qed: str
    lipinski_violations: str
    lipinski_pass: str
    best_reference_name: str
    tanimoto_similarity: str
    similarity_category: str
    validity_score: str
    property_score: str
    qed_score: str
    lipinski_score: str
    differentiation_score: str
    prioritization_score: str
    prioritization_category: str
    scoring_notes: str
    chemical_identity_status: str = "not_run"
    chemical_identity_lookup_status: str = "not_run"
    public_lookup_status: str = "not_available"
    pubchem_status: str = "not_available"
    chembl_status: str = "not_available"
    nlp_status: str = "not_run"
    surechembl_query_status: str = "not_available"
    chemberta_status: str = "not_run"
    context_status: str = "not_run"
    nlp_evidence_score: str = "0.000"
    nlp_relevance_category: str = "not_relevant"
    nlp_evidence_count: str = "0"
    prioritization_score_with_nlp: str = ""
    prioritization_category_with_nlp: str = ""
    known_public_match: str = "False"
    known_public_match_source: str = ""
    known_public_match_id: str = ""
    novelty_flag: str = "not_available"
    ip_potential_category: str = "not_available"
    ip_potential_notes: str = ""
    patent_evidence_count: str = "0"
    patent_max_similarity: str = ""
    patent_relevance_score: str = "0.000"
    patent_signal_category: str = "not_available"
    patent_signal_notes: str = ""
    surechembl_evidence_count: str = "0"
    surechembl_max_similarity: str = ""
    surechembl_signal_category: str = "not_available"
    surechembl_signal_notes: str = ""


@dataclass(frozen=True)
class NlpEvidenceSummary:
    """Aggregated NLP evidence for one molecule."""

    score: float
    category: str
    count: int
    status: str = "available"


@dataclass(frozen=True)
class PublicMatchSummary:
    """Aggregated exact-match and similarity evidence from public lookups."""

    known_public_match: bool
    exact_source: str = ""
    exact_id: str = ""
    max_similarity: float | None = None
    public_lookup_status: str = "not_available"
    pubchem_status: str = "not_available"
    chembl_status: str = "not_available"


@dataclass(frozen=True)
class PatentEvidenceSummary:
    """Aggregated patent text evidence for one molecule."""

    evidence_count: int
    relevance_score: float
    category: str
    notes: str


@dataclass(frozen=True)
class SurechemblEvidenceSummary:
    """Aggregated SureChEMBL structure evidence for one molecule."""

    evidence_count: int
    max_similarity: float | None
    category: str
    notes: str
    query_status: str = "not_available"


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: object) -> float | None:
    """Parse a finite float from CSV data, returning None when unavailable."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def calculate_property_score(
    valid_smiles: bool,
    molecular_weight: float | None,
    logp: float | None,
    tpsa: float | None,
    hbd: float | None,
    hba: float | None,
    rotatable_bonds: float | None,
) -> float:
    """Score specified drug-like property thresholds from zero to one."""
    values = (
        molecular_weight,
        logp,
        tpsa,
        hbd,
        hba,
        rotatable_bonds,
    )
    if not valid_smiles or any(value is None for value in values):
        return 0.0

    violations = sum(
        (
            molecular_weight > 500,
            logp > 5,
            tpsa > 140,
            hbd > 5,
            hba > 10,
            rotatable_bonds > 10,
        )
    )
    return max(0.0, 1.0 - violations / 6.0)


def calculate_qed_score(valid_smiles: bool, qed: float | None) -> float:
    """Use QED directly, constrained to the zero-to-one score range."""
    if not valid_smiles or qed is None:
        return 0.0
    return min(1.0, max(0.0, qed))


def calculate_lipinski_score(
    valid_smiles: bool,
    lipinski_pass: bool,
    lipinski_violations: float | None,
) -> float:
    """Score Lipinski results using the specified violation categories."""
    if not valid_smiles:
        return 0.0
    if lipinski_pass:
        return 1.0
    if lipinski_violations is None:
        return 0.0
    if lipinski_violations == 1:
        return 0.5
    if lipinski_violations == 2:
        return 0.25
    return 0.0


def calculate_differentiation_score(
    valid_smiles: bool, tanimoto_similarity: float | None
) -> float:
    """Reward chemical differentiation from the local reference set."""
    if not valid_smiles or tanimoto_similarity is None:
        return 0.0
    if tanimoto_similarity >= 0.85:
        return 0.25
    if tanimoto_similarity >= 0.70:
        return 0.50
    if tanimoto_similarity >= 0.50:
        return 0.75
    return 1.0


def calculate_prioritization_score(
    validity_score: float,
    property_score: float,
    qed_score: float,
    lipinski_score: float,
    differentiation_score: float,
) -> float:
    """Calculate the weighted novelty-aware prioritization score."""
    score = (
        validity_score * SCORE_WEIGHTS["validity"]
        + property_score * SCORE_WEIGHTS["property"]
        + qed_score * SCORE_WEIGHTS["qed"]
        + lipinski_score * SCORE_WEIGHTS["lipinski"]
        + differentiation_score * SCORE_WEIGHTS["differentiation"]
    )
    return min(1.0, max(0.0, score))


def categorize_prioritization(score: float, valid_smiles: bool = True) -> str:
    """Assign the requested prioritization category."""
    if not valid_smiles:
        return "deprioritized"
    if score >= 0.80:
        return "high_priority"
    if score >= 0.60:
        return "medium_priority"
    if score >= 0.40:
        return "low_priority"
    return "deprioritized"


def categorize_nlp_relevance(score: float) -> str:
    """Assign the NLP relevance category from an aggregated score."""
    if score >= 0.75:
        return "high_nlp_relevance"
    if score >= 0.55:
        return "medium_nlp_relevance"
    if score >= 0.35:
        return "low_nlp_relevance"
    return "not_relevant"


def aggregate_nlp_evidence(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, NlpEvidenceSummary]:
    """Aggregate NLP rows by molecule using maximum score and row count."""
    scores_by_molecule: dict[str, list[float]] = {}
    counts_by_molecule: dict[str, int] = {}
    statuses_by_molecule: dict[str, list[str]] = {}

    for row in rows:
        molecule_id = row.get("molecule_id", "").strip()
        counts_by_molecule[molecule_id] = counts_by_molecule.get(
            molecule_id, 0
        ) + 1
        statuses_by_molecule.setdefault(molecule_id, []).append(
            row.get("nlp_status", "").strip()
        )
        score = parse_float(
            row.get("similarity_score", "")
            or row.get("max_relevance_score", "")
        )
        if score is not None:
            scores_by_molecule.setdefault(molecule_id, []).append(score)

    summaries: dict[str, NlpEvidenceSummary] = {}
    for molecule_id, count in counts_by_molecule.items():
        statuses = {
            status for status in statuses_by_molecule.get(molecule_id, []) if status
        }
        if statuses == {"model_unavailable"}:
            summaries[molecule_id] = NlpEvidenceSummary(
                score=0.0,
                category="not_run",
                count=count,
                status="model_unavailable",
            )
            continue
        available_scores = scores_by_molecule.get(molecule_id, [])
        raw_score = max(available_scores) if available_scores else 0.0
        score = min(1.0, max(0.0, raw_score))
        summaries[molecule_id] = NlpEvidenceSummary(
            score=score,
            category=categorize_nlp_relevance(score),
            count=count,
            status="available",
        )
    return summaries


def summarize_statuses(statuses: Iterable[str]) -> str:
    """Summarize evidence row statuses into one conservative stage status."""
    cleaned = {status.strip() for status in statuses if status and status.strip()}
    if not cleaned:
        return "not_available"
    for status in (
        "lookup_error",
        "match_found",
        "no_match",
        "not_queried",
        "not_run",
        "offline",
        "invalid_molecule",
    ):
        if status in cleaned:
            return status
    return sorted(cleaned)[0]


def aggregate_public_matches(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, PublicMatchSummary]:
    """Aggregate successful public exact matches and similarity results."""
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("molecule_id", "").strip(), []).append(row)

    summaries: dict[str, PublicMatchSummary] = {}
    for molecule_id, molecule_rows in grouped.items():
        pubchem_status = summarize_statuses(
            row.get("lookup_status", "")
            for row in molecule_rows
            if row.get("source_database", "").strip().lower() == "pubchem"
        )
        chembl_status = summarize_statuses(
            row.get("lookup_status", "")
            for row in molecule_rows
            if row.get("source_database", "").strip().lower() == "chembl"
        )
        public_lookup_status = summarize_statuses(
            row.get("lookup_status", "") for row in molecule_rows
        )
        successful = [
            row
            for row in molecule_rows
            if row.get("lookup_status", "").strip() == "match_found"
        ]
        exact_rows = [
            row
            for row in successful
            if row.get("match_type", "").strip() == "exact_inchikey"
        ]
        similarity_scores = [
            score
            for row in successful
            if row.get("match_type", "").strip() == "similarity"
            for score in [parse_float(row.get("similarity", ""))]
            if score is not None
        ]
        exact = exact_rows[0] if exact_rows else {}
        summaries[molecule_id] = PublicMatchSummary(
            known_public_match=bool(exact_rows),
            exact_source=exact.get("source_database", "").strip(),
            exact_id=exact.get("public_id", "").strip(),
            max_similarity=(
                max(similarity_scores) if similarity_scores else None
            ),
            public_lookup_status=public_lookup_status,
            pubchem_status=pubchem_status,
            chembl_status=chembl_status,
        )
    return summaries


def classify_ip_potential(
    valid_smiles: bool,
    public_summary: PublicMatchSummary | None,
) -> tuple[str, str, str]:
    """Classify novelty and IP-potential signals from public evidence."""
    if not valid_smiles:
        return (
            "not_available",
            "not_available",
            "IP-potential research signal is unavailable for an invalid molecule.",
        )
    if public_summary is None:
        return (
            "not_available",
            "not_available",
            "Public lookup evidence was not provided; novelty signal is unavailable.",
        )
    if public_summary.known_public_match:
        return (
            "known_public_compound",
            "low_ip_potential_signal",
            "An exact known/public match reduces the novelty and IP-potential "
            "research signal while leaving general property scores unchanged.",
        )
    similarity = public_summary.max_similarity
    if similarity is None:
        return (
            "not_available",
            "not_available",
            "No exact public match found and no successful public similarity evidence was available.",
        )
    if similarity >= 0.85:
        return (
            "very_close_public_analog",
            "reduced_ip_potential_signal",
            "A very close public analog reduces the chemical differentiation "
            "and IP-potential research signal.",
        )
    if similarity >= 0.70:
        return (
            "related_public_chemotype",
            "moderate_ip_potential_signal",
            "A related public chemotype supports a moderate IP-potential "
            "research signal that requires further evidence review.",
        )
    return (
        "chemically_differentiated_candidate",
        "higher_ip_potential_signal",
        "Available public similarity evidence indicates greater chemical "
        "differentiation, supporting a higher IP-potential research signal.",
    )


def aggregate_patent_evidence(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, PatentEvidenceSummary]:
    """Aggregate patent-search rows into conservative text-evidence signals."""
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("molecule_id", "").strip(), []).append(row)

    summaries: dict[str, PatentEvidenceSummary] = {}
    for molecule_id, molecule_rows in grouped.items():
        statuses = {
            row.get("search_status", "").strip() for row in molecule_rows
        }
        evidence_count = sum(
            row.get("search_status", "").strip() == "match_found"
            for row in molecule_rows
        )
        if evidence_count:
            category = "patent_text_evidence_signal"
            score = 1.0
            if "offline" in statuses:
                notes = (
                    "Public-safe demonstration patent-style evidence was "
                    "generated in offline mode. This is a text-evidence "
                    "research signal and does not establish structural claim scope."
                )
            else:
                notes = (
                    "Public patent text matches were found. This is a text-evidence "
                    "research signal and does not establish structural claim scope."
                )
        elif "offline" in statuses:
            category = "patent_search_not_available"
            score = 0.0
            notes = "Patent search ran in offline mode; no patent API request was made."
        elif "lookup_error" in statuses:
            category = "patent_lookup_error"
            score = 0.0
            notes = "Patent search encountered a lookup error."
        elif "invalid_molecule" in statuses:
            category = "not_available"
            score = 0.0
            notes = "Patent evidence is unavailable for an invalid molecule."
        else:
            category = "no_patent_text_signal"
            score = 0.0
            notes = "No public patent title or abstract text matches were available."
        summaries[molecule_id] = PatentEvidenceSummary(
            evidence_count=evidence_count,
            relevance_score=score,
            category=category,
            notes=notes,
        )
    return summaries


def categorize_surechembl_signal(max_similarity: float | None) -> tuple[str, str]:
    """Classify SureChEMBL patent-associated structure similarity evidence."""
    if max_similarity is None:
        return (
            "not_available",
            "No SureChEMBL structure evidence was available.",
        )
    if max_similarity >= 0.85:
        return (
            "very_close_patent_compound_signal",
            "A very close patent-associated chemical structure signal was found.",
        )
    if max_similarity >= 0.70:
        return (
            "related_patent_chemotype_signal",
            "A related patent-associated chemical structure signal was found.",
        )
    if max_similarity >= 0.50:
        return (
            "moderate_patent_chemistry_signal",
            "A moderate patent-associated chemical structure signal was found.",
        )
    return (
        "low_patent_chemistry_similarity_signal",
        "Only low structural similarity to SureChEMBL patent-associated structures was found.",
    )


def aggregate_surechembl_evidence(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, SurechemblEvidenceSummary]:
    """Aggregate local SureChEMBL structure evidence rows."""
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("molecule_id", "").strip(), []).append(row)

    summaries: dict[str, SurechemblEvidenceSummary] = {}
    for molecule_id, molecule_rows in grouped.items():
        scores = [
            score
            for row in molecule_rows
            if row.get("lookup_status", "").strip() == "match_found"
            for score in [parse_float(row.get("tanimoto_similarity", ""))]
            if score is not None
        ]
        max_similarity = max(scores) if scores else None
        category, notes = categorize_surechembl_signal(max_similarity)
        query_status = summarize_statuses(
            row.get("lookup_status", "") for row in molecule_rows
        )
        summaries[molecule_id] = SurechemblEvidenceSummary(
            evidence_count=len(scores),
            max_similarity=max_similarity,
            category=category,
            notes=(
                notes
                + " This is a SureChEMBL structure evidence signal, not a legal conclusion."
            ),
            query_status=query_status,
        )
    return summaries


def score_molecule(
    descriptor_row: Mapping[str, str],
    similarity_row: Mapping[str, str] | None,
    nlp_summary: NlpEvidenceSummary | None = None,
    public_summary: PublicMatchSummary | None = None,
    patent_summary: PatentEvidenceSummary | None = None,
    surechembl_summary: SurechemblEvidenceSummary | None = None,
    identity_row: Mapping[str, str] | None = None,
    context_row: Mapping[str, str] | None = None,
    chemberta_row: Mapping[str, str] | None = None,
    nlp_was_run: bool = False,
) -> PrioritizationResult:
    """Calculate component and final scores for one descriptor row."""
    valid_smiles = parse_boolean(descriptor_row.get("valid_smiles", ""))
    molecular_weight = parse_float(descriptor_row.get("molecular_weight", ""))
    logp = parse_float(descriptor_row.get("logp", ""))
    tpsa = parse_float(descriptor_row.get("tpsa", ""))
    hbd = parse_float(descriptor_row.get("hbd", ""))
    hba = parse_float(descriptor_row.get("hba", ""))
    rotatable_bonds = parse_float(
        descriptor_row.get("rotatable_bonds", "")
    )
    qed = parse_float(descriptor_row.get("qed", ""))
    lipinski_violations = parse_float(
        descriptor_row.get("lipinski_violations", "")
    )
    lipinski_pass = parse_boolean(
        descriptor_row.get("lipinski_pass", "")
    )

    similarity_row = similarity_row or {}
    tanimoto_similarity = parse_float(
        similarity_row.get("tanimoto_similarity", "")
    )

    validity_score = 1.0 if valid_smiles else 0.0
    property_score = calculate_property_score(
        valid_smiles=valid_smiles,
        molecular_weight=molecular_weight,
        logp=logp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotatable_bonds,
    )
    qed_score = calculate_qed_score(valid_smiles, qed)
    lipinski_score = calculate_lipinski_score(
        valid_smiles=valid_smiles,
        lipinski_pass=lipinski_pass,
        lipinski_violations=lipinski_violations,
    )
    differentiation_score = calculate_differentiation_score(
        valid_smiles, tanimoto_similarity
    )
    prioritization_score = calculate_prioritization_score(
        validity_score=validity_score,
        property_score=property_score,
        qed_score=qed_score,
        lipinski_score=lipinski_score,
        differentiation_score=differentiation_score,
    )
    category = categorize_prioritization(
        prioritization_score, valid_smiles=valid_smiles
    )
    nlp_status = nlp_summary.status if nlp_summary else ""
    nlp_score_available = (
        nlp_summary is not None and nlp_status != "model_unavailable"
    )
    nlp_score = nlp_summary.score if nlp_summary else None
    score_with_nlp = (
        0.8 * prioritization_score + 0.2 * nlp_score
        if nlp_score_available and nlp_score is not None
        else prioritization_score
    )
    category_with_nlp = categorize_prioritization(
        score_with_nlp, valid_smiles=valid_smiles
    )
    novelty_flag, ip_category, ip_notes = classify_ip_potential(
        valid_smiles, public_summary
    )

    notes: list[str] = []
    if not valid_smiles:
        upstream_error = descriptor_row.get("descriptor_error", "").strip()
        notes.append(upstream_error or "Invalid molecule; scores set to zero.")
    else:
        required_properties = (
            molecular_weight,
            logp,
            tpsa,
            hbd,
            hba,
            rotatable_bonds,
        )
        if any(value is None for value in required_properties):
            notes.append("Incomplete property data; property score set to zero.")
        if qed is None:
            notes.append("QED unavailable; QED score set to zero.")
        if tanimoto_similarity is None:
            notes.append(
                "Similarity unavailable; differentiation score set to zero."
            )
    notes.append(
        "Research prioritization signal only; not a legal determination."
    )

    return PrioritizationResult(
        molecule_id=descriptor_row.get("molecule_id", "").strip(),
        canonical_smiles=descriptor_row.get("canonical_smiles", ""),
        valid_smiles=valid_smiles,
        molecular_weight=descriptor_row.get("molecular_weight", ""),
        logp=descriptor_row.get("logp", ""),
        tpsa=descriptor_row.get("tpsa", ""),
        hbd=descriptor_row.get("hbd", ""),
        hba=descriptor_row.get("hba", ""),
        rotatable_bonds=descriptor_row.get("rotatable_bonds", ""),
        qed=descriptor_row.get("qed", ""),
        lipinski_violations=descriptor_row.get(
            "lipinski_violations", ""
        ),
        lipinski_pass=descriptor_row.get("lipinski_pass", ""),
        best_reference_name=similarity_row.get("best_reference_name", ""),
        tanimoto_similarity=similarity_row.get(
            "tanimoto_similarity", ""
        ),
        similarity_category=similarity_row.get(
            "similarity_category", "not_available"
        )
        or "not_available",
        validity_score=f"{validity_score:.3f}",
        property_score=f"{property_score:.3f}",
        qed_score=f"{qed_score:.3f}",
        lipinski_score=f"{lipinski_score:.3f}",
        differentiation_score=f"{differentiation_score:.3f}",
        prioritization_score=f"{prioritization_score:.3f}",
        prioritization_category=category,
        scoring_notes=" ".join(notes),
        chemical_identity_status=(
            identity_row.get("identity_status", "").strip()
            if identity_row
            else "not_run"
        )
        or "not_run",
        chemical_identity_lookup_status=(
            identity_row.get("lookup_status", "").strip()
            if identity_row
            else "not_run"
        )
        or "not_run",
        public_lookup_status=(
            public_summary.public_lookup_status if public_summary else "not_available"
        ),
        pubchem_status=(
            public_summary.pubchem_status if public_summary else "not_available"
        ),
        chembl_status=(
            public_summary.chembl_status if public_summary else "not_available"
        ),
        nlp_status=(
            nlp_summary.status
            if nlp_summary
            else ("no_match" if nlp_was_run else "not_run")
        ),
        surechembl_query_status=(
            surechembl_summary.query_status if surechembl_summary else "not_available"
        ),
        chemberta_status=(
            "available"
            if chemberta_row
            and parse_boolean(chemberta_row.get("embedding_available", ""))
            else ("not_available" if chemberta_row else "not_run")
        ),
        context_status=(
            context_row.get("context_status", "").strip()
            if context_row
            else "not_run"
        )
        or "not_run",
        nlp_evidence_score=f"{nlp_score:.3f}" if nlp_score is not None else "",
        nlp_relevance_category=(
            nlp_summary.category
            if nlp_summary
            else ("no_match" if nlp_was_run else "not_run")
        ),
        nlp_evidence_count=str(nlp_summary.count if nlp_summary else 0),
        prioritization_score_with_nlp=f"{score_with_nlp:.3f}",
        prioritization_category_with_nlp=category_with_nlp,
        known_public_match=str(
            bool(public_summary and public_summary.known_public_match)
        ),
        known_public_match_source=(
            public_summary.exact_source if public_summary else ""
        ),
        known_public_match_id=(
            public_summary.exact_id if public_summary else ""
        ),
        novelty_flag=novelty_flag,
        ip_potential_category=ip_category,
        ip_potential_notes=ip_notes,
        patent_evidence_count=str(
            patent_summary.evidence_count if patent_summary else 0
        ),
        patent_max_similarity=(
            f"{patent_summary.relevance_score:.3f}" if patent_summary else ""
        ),
        patent_relevance_score=(
            f"{patent_summary.relevance_score:.3f}"
            if patent_summary
            else "0.000"
        ),
        patent_signal_category=(
            patent_summary.category if patent_summary else "not_available"
        ),
        patent_signal_notes=(
            patent_summary.notes
            if patent_summary
            else "Patent evidence was not provided to scoring."
        ),
        surechembl_evidence_count=str(
            surechembl_summary.evidence_count if surechembl_summary else 0
        ),
        surechembl_max_similarity=(
            f"{surechembl_summary.max_similarity:.3f}"
            if surechembl_summary and surechembl_summary.max_similarity is not None
            else ""
        ),
        surechembl_signal_category=(
            surechembl_summary.category if surechembl_summary else "not_available"
        ),
        surechembl_signal_notes=(
            surechembl_summary.notes
            if surechembl_summary
            else "SureChEMBL structure evidence was not provided to scoring."
        ),
    )


def read_csv_with_columns(
    path: Path, required_columns: Sequence[str], label: str
) -> list[dict[str, str]]:
    """Read a CSV and validate its required columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(required_columns) - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{label} CSV is missing required columns: {missing}")
        return [dict(row) for row in reader]


def index_by_molecule_id(
    rows: Iterable[Mapping[str, str]], label: str
) -> dict[str, Mapping[str, str]]:
    """Index rows by molecule_id and reject ambiguous duplicate IDs."""
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        molecule_id = row.get("molecule_id", "").strip()
        if molecule_id in indexed:
            raise ValueError(
                f"{label} CSV contains duplicate molecule_id: {molecule_id}"
            )
        indexed[molecule_id] = row
    return indexed


def score_rows(
    descriptor_rows: Iterable[Mapping[str, str]],
    similarity_rows: Iterable[Mapping[str, str]],
    nlp_summaries: Mapping[str, NlpEvidenceSummary] | None = None,
    public_summaries: Mapping[str, PublicMatchSummary] | None = None,
    patent_summaries: Mapping[str, PatentEvidenceSummary] | None = None,
    surechembl_summaries: Mapping[str, SurechemblEvidenceSummary] | None = None,
    identity_rows: Mapping[str, Mapping[str, str]] | None = None,
    context_rows: Mapping[str, Mapping[str, str]] | None = None,
    chemberta_rows: Mapping[str, Mapping[str, str]] | None = None,
    nlp_was_run: bool = False,
) -> list[PrioritizationResult]:
    """Merge by molecule_id and score every descriptor row."""
    similarity_index = index_by_molecule_id(similarity_rows, "Similarity")
    return [
        score_molecule(
            descriptor_row=row,
            similarity_row=similarity_index.get(
                row.get("molecule_id", "").strip()
            ),
            nlp_summary=(nlp_summaries or {}).get(
                row.get("molecule_id", "").strip()
            ),
            public_summary=(public_summaries or {}).get(
                row.get("molecule_id", "").strip()
            ),
            patent_summary=(patent_summaries or {}).get(
                row.get("molecule_id", "").strip()
            ),
            surechembl_summary=(surechembl_summaries or {}).get(
                row.get("molecule_id", "").strip()
            ),
            identity_row=(identity_rows or {}).get(
                row.get("molecule_id", "").strip()
            ),
            context_row=(context_rows or {}).get(
                row.get("molecule_id", "").strip()
            ),
            chemberta_row=(chemberta_rows or {}).get(
                row.get("molecule_id", "").strip()
            ),
            nlp_was_run=nlp_was_run,
        )
        for row in descriptor_rows
    ]


def write_output_csv(
    output_path: Path,
    records: Iterable[PrioritizationResult],
    include_nlp: bool = False,
    include_public_lookup: bool = False,
    include_patent: bool = False,
    include_surechembl: bool = False,
) -> None:
    """Write prioritization results to a CSV file."""
    if include_surechembl:
        fieldnames = SURECHEMBL_OUTPUT_COLUMNS
    elif include_patent:
        fieldnames = PATENT_OUTPUT_COLUMNS
    elif include_public_lookup:
        fieldnames = PUBLIC_OUTPUT_COLUMNS
    elif include_nlp:
        fieldnames = NLP_OUTPUT_COLUMNS
    else:
        fieldnames = OUTPUT_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def scoring_csv(
    descriptor_path: Path,
    similarity_path: Path,
    output_path: Path,
    nlp_path: Path | None = None,
    public_lookup_path: Path | None = None,
    patent_path: Path | None = None,
    surechembl_path: Path | None = None,
    identity_path: Path | None = None,
    context_path: Path | None = None,
    chemberta_path: Path | None = None,
    nlp_was_run: bool | None = None,
) -> int:
    """Score local descriptor and similarity outputs."""
    descriptor_rows = read_csv_with_columns(
        descriptor_path, DESCRIPTOR_COLUMNS, "Descriptor"
    )
    similarity_rows = read_csv_with_columns(
        similarity_path, SIMILARITY_COLUMNS, "Similarity"
    )
    nlp_summaries = None
    if nlp_path is not None:
        nlp_rows = read_csv_with_columns(nlp_path, NLP_COLUMNS, "NLP")
        nlp_summaries = aggregate_nlp_evidence(nlp_rows)
        if nlp_was_run is None:
            nlp_was_run = bool(nlp_rows)
    public_summaries = None
    if public_lookup_path is not None:
        public_rows = read_csv_with_columns(
            public_lookup_path, PUBLIC_LOOKUP_COLUMNS, "Public lookup"
        )
        public_summaries = aggregate_public_matches(public_rows)
    patent_summaries = None
    if patent_path is not None:
        patent_rows = read_csv_with_columns(
            patent_path, PATENT_COLUMNS, "Patent"
        )
        patent_summaries = aggregate_patent_evidence(patent_rows)
    surechembl_summaries = None
    if surechembl_path is not None:
        surechembl_rows = read_csv_with_columns(
            surechembl_path, SURECHEMBL_COLUMNS, "SureChEMBL structure evidence"
        )
        surechembl_summaries = aggregate_surechembl_evidence(surechembl_rows)
    identity_rows = None
    if identity_path is not None:
        identity_rows = index_by_molecule_id(
            read_csv_with_columns(identity_path, IDENTITY_COLUMNS, "Chemical identity"),
            "Chemical identity",
        )
    context_rows = None
    if context_path is not None:
        context_rows = index_by_molecule_id(
            read_csv_with_columns(context_path, CONTEXT_COLUMNS, "Compound context"),
            "Compound context",
        )
    chemberta_rows = None
    if chemberta_path is not None:
        chemberta_rows = index_by_molecule_id(
            read_csv_with_columns(
                chemberta_path, CHEMBERTA_COLUMNS, "ChemBERTa embeddings"
            ),
            "ChemBERTa embeddings",
        )
    results = score_rows(
        descriptor_rows,
        similarity_rows,
        nlp_summaries,
        public_summaries,
        patent_summaries,
        surechembl_summaries,
        identity_rows,
        context_rows,
        chemberta_rows,
        bool(nlp_was_run),
    )
    write_output_csv(
        output_path,
        results,
        include_nlp=nlp_path is not None,
        include_public_lookup=public_lookup_path is not None,
        include_patent=patent_path is not None,
        include_surechembl=surechembl_path is not None,
    )
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate novelty-aware compound prioritization scores."
    )
    parser.add_argument(
        "--descriptors",
        required=True,
        type=Path,
        help="CSV containing molecular descriptors.",
    )
    parser.add_argument(
        "--similarity",
        required=True,
        type=Path,
        help="CSV containing best-reference similarity results.",
    )
    parser.add_argument(
        "--nlp",
        type=Path,
        help="Optional CSV containing text NLP evidence scores.",
    )
    parser.add_argument(
        "--public-lookup",
        type=Path,
        help="Optional CSV containing existing public lookup evidence.",
    )
    parser.add_argument(
        "--patent",
        type=Path,
        help="Optional CSV containing existing patent text evidence.",
    )
    parser.add_argument(
        "--surechembl",
        type=Path,
        help="Optional CSV containing local SureChEMBL structure evidence.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV for compound prioritization scores.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compound prioritization command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        row_count = scoring_csv(
            args.descriptors,
            args.similarity,
            args.output,
            nlp_path=args.nlp,
            public_lookup_path=args.public_lookup,
            patent_path=args.patent,
            surechembl_path=args.surechembl,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {row_count} prioritization records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
