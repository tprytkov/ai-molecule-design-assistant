"""SureChEMBL structure evidence using local files or explicit online mode."""

from __future__ import annotations

import argparse
import csv
import json
import socket
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rdkit import DataStructs

from src.similarity import create_morgan_fingerprint, parse_boolean


DESCRIPTOR_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")
SURECHEMBL_COLUMNS = (
    "surechembl_id",
    "patent_id",
    "patent_title",
    "patent_date",
    "compound_name",
    "smiles",
    "source_section",
    "evidence_note",
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "surechembl_id",
    "compound_name",
    "patent_id",
    "patent_number",
    "patent_title",
    "patent_date",
    "patent_section",
    "patent_metadata_status",
    "patent_metadata_source",
    "patent_compound_smiles",
    "tanimoto_similarity",
    "similarity_category",
    "source_section",
    "evidence_note",
    "lookup_status",
    "error_message",
)
SURECHEMBL_API_BASE_URL = "https://www.surechembl.org/api"
SURECHEMBL_STRUCTURE_SEARCH_URL = f"{SURECHEMBL_API_BASE_URL}/search/structure"
SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL = (
    f"{SURECHEMBL_API_BASE_URL}/search/documents_for_structures"
)
SURECHEMBL_SEARCH_TYPE = "SIMILARITY"
DEFAULT_TIMEOUT = 20.0
ONLINE_DISCLOSURE_WARNING = (
    "Online SureChEMBL mode sends query structures to an external public "
    "database. Use only for public/demo molecules unless you intentionally "
    "accept this disclosure risk."
)


class SurechemblClient(Protocol):
    """Minimal JSON client interface for online lookup and tests."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        timeout: float,
        debug: bool = False,
    ) -> Mapping[str, object]:
        """POST JSON and return a decoded JSON object."""

    def get_json(
        self, url: str, timeout: float, debug: bool = False
    ) -> Mapping[str, object]:
        """Return a decoded JSON object."""


class UrllibSurechemblClient:
    """Small standard-library JSON client with a descriptive user agent."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        timeout: float,
        debug: bool = False,
    ) -> Mapping[str, object]:
        body = json.dumps(payload).encode("utf-8")
        if debug:
            print(f"SureChEMBL request URL: {url}")
            print(f"SureChEMBL request JSON payload: {json.dumps(payload)}")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "molecule-intelligence-demo/1.0",
            },
        )
        return self._request_json(request, timeout, debug=debug)

    def get_json(
        self, url: str, timeout: float, debug: bool = False
    ) -> Mapping[str, object]:
        if debug:
            print(f"SureChEMBL request URL: {url}")
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "molecule-intelligence-demo/1.0",
            },
        )
        return self._request_json(request, timeout, debug=debug)

    def _request_json(
        self, request: Request, timeout: float, *, debug: bool
    ) -> Mapping[str, object]:
        """Execute a JSON request and normalize API errors."""
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                if debug:
                    print(f"SureChEMBL response status code: {response.status}")
                    print(f"SureChEMBL response body text: {body}")
                payload = json.loads(body) if body else {}
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if debug:
                print(f"SureChEMBL response status code: {exc.code}")
                print(f"SureChEMBL response body text: {body}")
            message = f"HTTP {exc.code} returned by SureChEMBL."
            if body:
                message = f"{message} Response body: {body}"
            raise RuntimeError(message) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise RuntimeError(f"SureChEMBL request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("SureChEMBL returned an invalid JSON response.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("SureChEMBL returned an unexpected response.")
        return payload


@dataclass(frozen=True)
class PatentCompound:
    """Valid patent-associated compound from the local demo dataset."""

    surechembl_id: str
    patent_id: str
    patent_title: str
    patent_date: str
    compound_name: str
    smiles: str
    source_section: str
    evidence_note: str
    fingerprint: object


@dataclass(frozen=True)
class SurechemblHit:
    """One SureChEMBL structure evidence output row."""

    molecule_id: str
    canonical_smiles: str
    valid_smiles: bool
    surechembl_id: str = ""
    compound_name: str = ""
    patent_id: str = ""
    patent_number: str = ""
    patent_title: str = ""
    patent_date: str = ""
    patent_section: str = ""
    patent_metadata_status: str = ""
    patent_metadata_source: str = ""
    patent_compound_smiles: str = ""
    tanimoto_similarity: str = ""
    similarity_category: str = ""
    source_section: str = ""
    evidence_note: str = ""
    lookup_status: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class PatentDocumentMetadata:
    """Best-available patent/document metadata for a SureChEMBL structure."""

    patent_id: str = ""
    patent_number: str = ""
    patent_title: str = ""
    patent_date: str = ""
    patent_section: str = ""
    metadata_source: str = ""
    evidence_note: str = ""


def categorize_similarity(score: float) -> str:
    """Categorize patent-associated compound similarity."""
    if score >= 0.85:
        return "very_close_patent_analog"
    if score >= 0.70:
        return "related_patent_chemotype"
    if score >= 0.50:
        return "moderate_patent_similarity"
    return "structurally_distinct_from_patent_compound"


def parse_similarity(value: object) -> float | None:
    """Normalize a similarity score to the zero-to-one range."""
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if score > 1:
        score /= 100.0
    if score < 0 or score > 1:
        return None
    return score


def placeholder_hit(
    molecule_id: str,
    canonical_smiles: str,
    valid_smiles: bool,
    *,
    lookup_status: str,
    evidence_note: str = "",
    error_message: str = "",
) -> SurechemblHit:
    """Create an invalid, skipped, no-match, or error output row."""
    return SurechemblHit(
        molecule_id=molecule_id,
        canonical_smiles=canonical_smiles,
        valid_smiles=valid_smiles,
        evidence_note=evidence_note,
        lookup_status=lookup_status,
        error_message=error_message,
    )


def read_csv_with_columns(
    path: Path, required_columns: Sequence[str], label: str
) -> list[dict[str, str]]:
    """Read a CSV and validate required columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing = set(required_columns) - fieldnames
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{label} CSV is missing required columns: {names}")
        return [dict(row) for row in reader]


def prepare_patent_compounds(
    rows: Iterable[Mapping[str, str]],
) -> list[PatentCompound]:
    """Create fingerprints for valid local patent-associated compounds."""
    compounds: list[PatentCompound] = []
    for row in rows:
        smiles = row.get("smiles", "").strip()
        try:
            fingerprint = create_morgan_fingerprint(smiles)
        except (ValueError, RuntimeError):
            continue
        compounds.append(
            PatentCompound(
                surechembl_id=row.get("surechembl_id", "").strip(),
                patent_id=row.get("patent_id", "").strip(),
                patent_title=row.get("patent_title", "").strip(),
                patent_date=row.get("patent_date", "").strip(),
                compound_name=row.get("compound_name", "").strip(),
                smiles=smiles,
                source_section=row.get("source_section", "").strip(),
                evidence_note=row.get("evidence_note", "").strip(),
                fingerprint=fingerprint,
            )
        )
    return compounds


def rank_hits(
    molecule_id: str,
    canonical_smiles: str,
    compounds: Sequence[PatentCompound],
    top_k: int,
) -> list[SurechemblHit]:
    """Rank local patent-associated compounds by descending similarity."""
    if top_k < 1:
        raise ValueError("top-k must be at least 1.")
    fingerprint = create_morgan_fingerprint(canonical_smiles)
    scored = [
        (
            DataStructs.TanimotoSimilarity(fingerprint, compound.fingerprint),
            index,
            compound,
        )
        for index, compound in enumerate(compounds)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        SurechemblHit(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=True,
            surechembl_id=compound.surechembl_id,
            compound_name=compound.compound_name,
            patent_id=compound.patent_id,
            patent_number=compound.patent_id,
            patent_title=compound.patent_title,
            patent_date=compound.patent_date,
            patent_section=compound.source_section,
            patent_metadata_status=(
                "found" if compound.patent_id else "structure_match_only"
            ),
            patent_metadata_source=(
                "local_demo_surechembl_compounds"
                if compound.patent_id
                else "not_available"
            ),
            patent_compound_smiles=compound.smiles,
            tanimoto_similarity=f"{score:.3f}",
            similarity_category=categorize_similarity(score),
            source_section=compound.source_section,
            evidence_note=compound.evidence_note,
            lookup_status="match_found",
        )
        for score, _, compound in scored[:top_k]
    ]


def lookup_rows(
    descriptor_rows: Iterable[Mapping[str, str]],
    compounds: Sequence[PatentCompound],
    top_k: int,
) -> list[SurechemblHit]:
    """Lookup every generated molecule against local patent compounds."""
    results: list[SurechemblHit] = []
    for row in descriptor_rows:
        molecule_id = row.get("molecule_id", "").strip()
        canonical_smiles = row.get("canonical_smiles", "").strip()
        if not parse_boolean(row.get("valid_smiles", "")):
            results.append(
                SurechemblHit(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    valid_smiles=False,
                    lookup_status="invalid_molecule",
                    error_message=(
                        row.get("descriptor_error", "").strip()
                        or "Generated molecule is invalid."
                    ),
                )
            )
            continue
        if not canonical_smiles:
            results.append(
                SurechemblHit(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    valid_smiles=False,
                    lookup_status="invalid_molecule",
                    error_message="Canonical SMILES is missing.",
                )
            )
            continue
        results.extend(rank_hits(molecule_id, canonical_smiles, compounds, top_k))
    return results


def build_structure_search_payload(
    canonical_smiles: str, top_k: int
) -> dict[str, object]:
    """Build the documented SureChEMBL StructureSearchRequest body."""
    return {
        "StructureSearchRequest": {
            "struct": canonical_smiles,
            "structSearchType": SURECHEMBL_SEARCH_TYPE,
            "maxResults": top_k,
            "saveSearch": "false",
        }
    }


def build_search_results_url(search_hash: str, top_k: int) -> str:
    """Build the documented SureChEMBL search-results URL."""
    return (
        f"{SURECHEMBL_API_BASE_URL}/search/{search_hash}/results"
        f"?page=0&max_results={top_k}"
    )


def build_documents_for_structures_url(
    chemical_ids: Sequence[str], items_per_page: int
) -> str:
    """Build the documented structure-to-document metadata URL."""
    query = urlencode(
        {
            "chemicalIds": list(chemical_ids),
            "page": 1,
            "itemsPerPage": items_per_page,
        },
        doseq=True,
    )
    return f"{SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL}?{query}"


def first_text(record: Mapping[str, object], keys: Sequence[str]) -> str:
    """Return the first non-empty string-like value for a set of keys."""
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_chemical_ids(value: object) -> tuple[str, ...]:
    """Normalize common chemical ID fields to strings."""
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value is not None and str(value).strip():
        return (str(value).strip(),)
    return ()


def extract_record_chemical_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    """Extract SureChEMBL chemical IDs from document-metadata records."""
    ids: list[str] = []
    for key in (
        "chemical_id",
        "chemicalId",
        "chemicalIds",
        "chemical_ids",
        "structure_id",
        "structureId",
        "surechembl_id",
        "id",
    ):
        ids.extend(normalize_chemical_ids(record.get(key)))
    structures = record.get("structures")
    if isinstance(structures, list):
        for item in structures:
            if isinstance(item, dict):
                ids.extend(extract_record_chemical_ids(item))
            else:
                ids.extend(normalize_chemical_ids(item))
    return tuple(dict.fromkeys(ids))


def extract_patent_id(record: Mapping[str, object]) -> str:
    """Extract a patent identifier from common flat or nested API fields."""
    direct = first_text(
        record,
        (
            "patent_id",
            "patent_number",
            "document_id",
            "publication_number",
        ),
    )
    if direct:
        return direct
    patents = record.get("patents")
    if isinstance(patents, list) and patents:
        first = patents[0]
        if isinstance(first, dict):
            return first_text(
                first,
                (
                    "patent_id",
                    "patent_number",
                    "document_id",
                    "publication_number",
                ),
            )
        return str(first).strip()
    return ""


def extract_patent_number(record: Mapping[str, object]) -> str:
    """Extract a publication/patent number from common API fields."""
    direct = first_text(
        record,
        (
            "patent_number",
            "publication_number",
            "document_number",
            "patent_id",
            "document_id",
        ),
    )
    if direct:
        return direct
    patents = record.get("patents")
    if isinstance(patents, list) and patents:
        first = patents[0]
        if isinstance(first, dict):
            return first_text(
                first,
                (
                    "patent_number",
                    "publication_number",
                    "document_number",
                    "patent_id",
                    "document_id",
                ),
            )
        return str(first).strip()
    return ""


def extract_document_records(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Extract document records from common SureChEMBL response shapes."""
    data = extract_response_data(payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and data is not payload:
        nested = extract_document_records(data)
        if nested:
            return nested
    for key in (
        "documents",
        "docs",
        "results",
        "content",
        "items",
        "patents",
        "patent_documents",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = extract_document_records(value)
            if nested:
                return nested
    return []


def parse_document_metadata(
    payload: Mapping[str, object],
) -> dict[str, PatentDocumentMetadata]:
    """Map SureChEMBL chemical IDs to best-available document metadata."""
    metadata: dict[str, PatentDocumentMetadata] = {}
    for record in extract_document_records(payload):
        chemical_ids = extract_record_chemical_ids(record)
        if not chemical_ids:
            continue
        patent_id = extract_patent_id(record)
        patent_number = extract_patent_number(record)
        patent_title = first_text(
            record,
            (
                "patent_title",
                "title",
                "document_title",
                "doc_title",
                "invention_title",
            ),
        )
        patent_date = first_text(
            record,
            (
                "patent_date",
                "publication_date",
                "document_date",
                "date",
                "published",
            ),
        )
        source_section = first_text(
            record,
            ("source_section", "section", "document_section", "field"),
        )
        note = (
            "SureChEMBL documents_for_structures metadata linked this "
            "structure-level SureChEMBL hit to mapped patent document metadata."
        )
        parsed = PatentDocumentMetadata(
            patent_id=patent_id or patent_number,
            patent_number=patent_number or patent_id,
            patent_title=patent_title,
            patent_date=patent_date,
            patent_section=source_section,
            metadata_source="SureChEMBL documents_for_structures",
            evidence_note=note,
        )
        for chemical_id in chemical_ids:
            metadata.setdefault(chemical_id, parsed)
    return metadata


def extract_response_data(payload: Mapping[str, object]) -> object:
    """Return SureChEMBL response data when wrapped in the documented envelope."""
    return payload.get("data", payload)


def extract_search_hash(payload: Mapping[str, object]) -> str:
    """Extract a search hash from common SureChEMBL response fields."""
    candidates: list[object] = [
        payload.get("hash"),
        payload.get("search_hash"),
        payload.get("searchHash"),
    ]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("hash"),
                data.get("search_hash"),
                data.get("searchHash"),
            ]
        )
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ""


def extract_result_records(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Extract candidate hit records from common SureChEMBL-like payload shapes."""
    data = extract_response_data(payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and data is not payload:
        nested = extract_result_records(data)
        if nested:
            return nested
    for key in (
        "structures",
        "results",
        "compounds",
        "hits",
        "content",
        "molecules",
        "items",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = extract_result_records(value)
            if nested:
                return nested
    return []


def parse_online_hits(
    payload: Mapping[str, object],
    molecule_id: str,
    canonical_smiles: str,
    top_k: int,
) -> list[SurechemblHit]:
    """Parse online SureChEMBL-like JSON into output rows."""
    parsed: list[tuple[float, int, Mapping[str, object]]] = []
    for index, record in enumerate(extract_result_records(payload)):
        score = parse_similarity(
            record.get("tanimoto_similarity")
            or record.get("similarity")
            or record.get("score")
        )
        if score is None:
            continue
        parsed.append((score, index, record))

    parsed.sort(key=lambda item: (-item[0], item[1]))
    results: list[SurechemblHit] = []
    for score, _, record in parsed[:top_k]:
        patent_id = extract_patent_id(record)
        patent_number = extract_patent_number(record)
        source_section = first_text(
            record,
            ("source_section", "section", "source"),
        ) or "SureChEMBL API"
        results.append(
            SurechemblHit(
                molecule_id=molecule_id,
                canonical_smiles=canonical_smiles,
                valid_smiles=True,
                surechembl_id=first_text(
                    record,
                    ("surechembl_id", "schembl_id", "compound_id", "id"),
                ),
                compound_name=first_text(
                    record,
                    ("compound_name", "name", "preferred_name"),
                ),
                patent_id=patent_id,
                patent_number=patent_number or patent_id,
                patent_title=first_text(record, ("patent_title", "title")),
                patent_date=first_text(
                    record,
                    ("patent_date", "publication_date", "date"),
                ),
                patent_section=source_section,
                patent_metadata_status="",
                patent_metadata_source="",
                patent_compound_smiles=first_text(
                    record,
                    ("smiles", "canonical_smiles", "compound_smiles"),
                ),
                tanimoto_similarity=f"{score:.3f}",
                similarity_category=categorize_similarity(score),
                source_section=source_section,
                evidence_note=(
                    "Online SureChEMBL public structure evidence search sent the "
                    "query structure to the SureChEMBL external public API. "
                    "This row is a structure-level public SureChEMBL hit."
                ),
                lookup_status="match_found",
            )
        )
    return results


def enrich_hits_with_document_metadata(
    hits: Sequence[SurechemblHit],
    metadata_by_chemical_id: Mapping[str, PatentDocumentMetadata],
) -> list[SurechemblHit]:
    """Attach best-available patent/document metadata to structure hits."""
    enriched: list[SurechemblHit] = []
    for hit in hits:
        metadata = metadata_by_chemical_id.get(hit.surechembl_id)
        if metadata is None:
            enriched.append(
                replace(
                    hit,
                    patent_id="not_available",
                    patent_number="not_available",
                    patent_title="not_available",
                    patent_date="not_available",
                    patent_section=hit.patent_section or hit.source_section,
                    patent_metadata_status="structure_match_only",
                    patent_metadata_source="not_available",
                    evidence_note=(
                        hit.evidence_note
                        + " Structure-level SureChEMBL hits were found, but "
                        "patent document metadata was not returned for these hits."
                    ),
                )
            )
            continue
        enriched.append(
            replace(
                hit,
                patent_id=metadata.patent_id or hit.patent_id or "not_available",
                patent_number=(
                    metadata.patent_number
                    or hit.patent_number
                    or metadata.patent_id
                    or "not_available"
                ),
                patent_title=(
                    metadata.patent_title or hit.patent_title or "not_available"
                ),
                patent_date=(
                    metadata.patent_date or hit.patent_date or "not_available"
                ),
                patent_section=metadata.patent_section or hit.patent_section,
                patent_metadata_status="found",
                patent_metadata_source=(
                    metadata.metadata_source
                    or "SureChEMBL documents_for_structures"
                ),
                source_section=metadata.patent_section or hit.source_section,
                evidence_note=metadata.evidence_note or hit.evidence_note,
            )
        )
    return enriched


def fetch_document_metadata_for_hits(
    hits: Sequence[SurechemblHit],
    client: SurechemblClient,
    timeout: float,
    debug_api: bool,
) -> dict[str, PatentDocumentMetadata]:
    """Fetch patent/document metadata for returned SureChEMBL structure IDs."""
    chemical_ids = [
        hit.surechembl_id
        for hit in hits
        if hit.lookup_status == "match_found" and hit.surechembl_id
    ]
    if not chemical_ids:
        return {}
    url = build_documents_for_structures_url(
        tuple(dict.fromkeys(chemical_ids)),
        items_per_page=max(len(chemical_ids), 1),
    )
    try:
        payload = client.post_json(url, {}, timeout, debug=debug_api)
    except RuntimeError as exc:
        if debug_api:
            print(f"SureChEMBL document metadata lookup failed: {exc}")
        return {}
    return parse_document_metadata(payload)


def lookup_online_rows(
    descriptor_rows: Iterable[Mapping[str, str]],
    *,
    top_k: int,
    max_molecules: int | None,
    client: SurechemblClient | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    debug_api: bool = False,
) -> list[SurechemblHit]:
    """Query SureChEMBL online mode for valid generated molecules."""
    if top_k < 1:
        raise ValueError("top-k must be at least 1.")
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("max-molecules must be at least 1.")

    active_client = client or UrllibSurechemblClient()
    results: list[SurechemblHit] = []
    queried = 0
    for row in descriptor_rows:
        molecule_id = row.get("molecule_id", "").strip()
        canonical_smiles = row.get("canonical_smiles", "").strip()
        if not parse_boolean(row.get("valid_smiles", "")):
            results.append(
                placeholder_hit(
                    molecule_id,
                    canonical_smiles,
                    False,
                    lookup_status="invalid_molecule",
                    evidence_note="SureChEMBL lookup was skipped for an invalid molecule.",
                    error_message=(
                        row.get("descriptor_error", "").strip()
                        or "Generated molecule is invalid."
                    ),
                )
            )
            continue
        if not canonical_smiles:
            results.append(
                placeholder_hit(
                    molecule_id,
                    canonical_smiles,
                    False,
                    lookup_status="invalid_molecule",
                    evidence_note="SureChEMBL lookup was skipped for a missing SMILES.",
                    error_message="Canonical SMILES is missing.",
                )
            )
            continue
        if max_molecules is not None and queried >= max_molecules:
            results.append(
                placeholder_hit(
                    molecule_id,
                    canonical_smiles,
                    True,
                    lookup_status="not_queried",
                    evidence_note=(
                        "Online SureChEMBL lookup skipped because max-molecules "
                        "limit was reached."
                    ),
                )
            )
            continue

        queried += 1
        try:
            payload = active_client.post_json(
                SURECHEMBL_STRUCTURE_SEARCH_URL,
                build_structure_search_payload(canonical_smiles, top_k),
                timeout,
                debug=debug_api,
            )
            search_hash = extract_search_hash(payload)
            if search_hash and not extract_result_records(payload):
                payload = active_client.get_json(
                    build_search_results_url(search_hash, top_k),
                    timeout,
                    debug=debug_api,
                )
            hits = parse_online_hits(payload, molecule_id, canonical_smiles, top_k)
            if hits:
                metadata = fetch_document_metadata_for_hits(
                    hits,
                    active_client,
                    timeout,
                    debug_api,
                )
                hits = enrich_hits_with_document_metadata(hits, metadata)
        except RuntimeError as exc:
            results.append(
                placeholder_hit(
                    molecule_id,
                    canonical_smiles,
                    True,
                    lookup_status="lookup_error",
                    evidence_note="Online SureChEMBL lookup could not be completed.",
                    error_message=str(exc),
                )
            )
            continue
        if hits:
            results.extend(hits)
        else:
            results.append(
                placeholder_hit(
                    molecule_id,
                    canonical_smiles,
                    True,
                    lookup_status="no_match",
                    evidence_note="No SureChEMBL API hit was parsed for this molecule.",
                )
            )
    return results


def write_output_csv(
    output_path: Path, records: Iterable[SurechemblHit]
) -> None:
    """Write SureChEMBL structure evidence rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def surechembl_lookup_csv(
    descriptor_path: Path,
    surechembl_path: Path | None,
    output_path: Path,
    top_k: int,
    *,
    online_surechembl: bool = False,
    max_molecules: int | None = None,
    client: SurechemblClient | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    debug_api: bool = False,
) -> int:
    """Run local or explicit online SureChEMBL structure lookup and return row count."""
    descriptors = read_csv_with_columns(
        descriptor_path, DESCRIPTOR_COLUMNS, "Descriptor"
    )
    if online_surechembl:
        print(ONLINE_DISCLOSURE_WARNING)
        results = lookup_online_rows(
            descriptors,
            top_k=top_k,
            max_molecules=max_molecules,
            client=client,
            timeout=timeout,
            debug_api=debug_api,
        )
    else:
        if surechembl_path is None:
            raise ValueError("Local mode requires --surechembl demo compound CSV.")
        compounds = prepare_patent_compounds(
            read_csv_with_columns(
                surechembl_path, SURECHEMBL_COLUMNS, "SureChEMBL structure evidence"
            )
        )
        if not compounds:
            raise ValueError("No valid SureChEMBL structure evidence compounds are available.")
        results = lookup_rows(descriptors, compounds, top_k)
    write_output_csv(output_path, results)
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run local SureChEMBL structure evidence lookup."
    )
    parser.add_argument("--descriptors", required=True, type=Path)
    parser.add_argument("--surechembl", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--online-surechembl",
        action="store_true",
        help="Query SureChEMBL online. Default uses the local demo CSV.",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        help="Limit valid generated molecules sent to online SureChEMBL.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--debug-api",
        action="store_true",
        help="Print SureChEMBL request and response details for online debugging.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SureChEMBL structure evidence lookup CLI."""
    args = build_parser().parse_args(argv)
    try:
        count = surechembl_lookup_csv(
            args.descriptors,
            args.surechembl,
            args.output,
            args.top_k,
            online_surechembl=args.online_surechembl,
            max_molecules=args.max_molecules,
            timeout=args.timeout,
            debug_api=args.debug_api,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {count} SureChEMBL structure evidence records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
