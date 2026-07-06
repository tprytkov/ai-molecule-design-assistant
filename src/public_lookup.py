"""Look up generated molecules in public compound databases."""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


OUTPUT_COLUMNS = (
    "molecule_id",
    "smiles",
    "standardized_smiles",
    "canonical_smiles",
    "inchi_key",
    "valid_smiles",
    "source",
    "source_database",
    "source_available",
    "public_match_found",
    "match_type",
    "public_id",
    "public_name",
    "public_smiles",
    "similarity",
    "public_url",
    "evidence_note",
    "lookup_status",
    "error_type",
    "error_message",
    "fallback_used",
)
STANDARDIZED_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "inchi_key",
    "valid_smiles",
)
DESCRIPTOR_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
CHEMBL_BASE_URL = "https://www.ebi.ac.uk"
DEFAULT_TIMEOUT = 15.0
CHEMBL_SIMILARITY_THRESHOLD = 70
DEGRADED_LOOKUP_NOTE = (
    "Public lookup source unavailable or interrupted; downstream analysis "
    "continues with degraded public-evidence status."
)
MAX_ERROR_MESSAGE_LENGTH = 240


class JsonClient(Protocol):
    """Minimal JSON HTTP client interface for API lookup and tests."""

    def get_json(self, url: str, timeout: float) -> Mapping[str, object]:
        """Return a decoded JSON object."""


class PublicLookupHttpError(RuntimeError):
    """HTTP error from a public compound database request."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_error_message(exc: BaseException | str, *, max_length: int = MAX_ERROR_MESSAGE_LENGTH) -> str:
    """Return a short public-safe error message."""
    message = " ".join(str(exc).replace("\r", " ").replace("\n", " ").split())
    return message[:max_length]


def error_type(exc: BaseException | str) -> str:
    """Return a stable error type label for external lookup failures."""
    if isinstance(exc, PublicLookupHttpError):
        if exc.status_code == 429:
            return "http_429_rate_limit"
        if exc.status_code in {500, 502, 503, 504}:
            return f"http_{exc.status_code}_server_error"
        return f"http_{exc.status_code}"
    if isinstance(exc, http.client.IncompleteRead):
        return "incomplete_read"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode_error"
    if isinstance(exc, UnicodeDecodeError):
        return "unicode_decode_error"
    if isinstance(exc, URLError):
        return "url_error"
    return type(exc).__name__ if isinstance(exc, BaseException) else "lookup_error"


class UrllibJsonClient:
    """Small standard-library JSON client with a descriptive user agent."""

    def get_json(self, url: str, timeout: float) -> Mapping[str, object]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "molecule-intelligence-demo/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise PublicLookupHttpError(
                exc.code,
                f"HTTP {exc.code} returned by public database."
            ) from exc
        except (http.client.IncompleteRead, URLError, ConnectionError, ConnectionResetError, socket.timeout, TimeoutError) as exc:
            raise RuntimeError(f"Public database request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Public database returned an invalid JSON response."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Public database returned an unexpected response.")
        return payload


@dataclass(frozen=True)
class LookupResult:
    """One public-database evidence row."""

    molecule_id: str
    smiles: str = ""
    standardized_smiles: str = ""
    canonical_smiles: str = ""
    inchi_key: str = ""
    valid_smiles: bool = False
    source: str = ""
    source_database: str = ""
    source_available: str = ""
    public_match_found: str = ""
    match_type: str = ""
    public_id: str = ""
    public_name: str = ""
    public_smiles: str = ""
    similarity: str = ""
    public_url: str = ""
    evidence_note: str = ""
    lookup_status: str = ""
    error_type: str = ""
    error_message: str = ""
    fallback_used: str = ""


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_similarity(value: object) -> float | None:
    """Normalize a ChEMBL similarity value to the zero-to-one range."""
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if score > 1:
        score /= 100.0
    if score < 0 or score > 1:
        return None
    return score


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


def index_descriptors(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, Mapping[str, str]]:
    """Index descriptor rows by molecule ID."""
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        molecule_id = row.get("molecule_id", "").strip()
        if molecule_id in indexed:
            raise ValueError(
                f"Descriptor CSV contains duplicate molecule_id: {molecule_id}"
            )
        indexed[molecule_id] = row
    return indexed


def parse_pubchem_exact_match(
    payload: Mapping[str, object],
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
) -> LookupResult | None:
    """Parse the first PubChem exact InChIKey property result."""
    property_table = payload.get("PropertyTable")
    if not isinstance(property_table, dict):
        return None
    properties = property_table.get("Properties")
    if not isinstance(properties, list) or not properties:
        return None
    record = properties[0]
    if not isinstance(record, dict) or not record.get("CID"):
        return None

    cid = str(record["CID"])
    public_smiles = str(
        record.get("ConnectivitySMILES")
        or record.get("CanonicalSMILES")
        or record.get("SMILES")
        or ""
    )
    return LookupResult(
        molecule_id=molecule_id,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        valid_smiles=True,
        source_database="PubChem",
        match_type="exact_inchikey",
        public_id=f"CID:{cid}",
        public_name=str(record.get("Title") or ""),
        public_smiles=public_smiles,
        similarity="1.000",
        public_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        evidence_note="Exact public PubChem record returned for the InChIKey.",
        lookup_status="match_found",
    )


def parse_chembl_similarity_match(
    payload: Mapping[str, object],
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
) -> LookupResult | None:
    """Parse the closest ChEMBL similarity result."""
    molecules = payload.get("molecules")
    if not isinstance(molecules, list):
        return None

    parsed: list[tuple[float, dict[str, object]]] = []
    for molecule in molecules:
        if not isinstance(molecule, dict):
            continue
        score = parse_similarity(molecule.get("similarity"))
        if score is not None:
            parsed.append((score, molecule))
    if not parsed:
        return None

    similarity, record = max(parsed, key=lambda item: item[0])
    chembl_id = str(record.get("molecule_chembl_id") or "")
    structures = record.get("molecule_structures")
    public_smiles = ""
    if isinstance(structures, dict):
        public_smiles = str(structures.get("canonical_smiles") or "")

    return LookupResult(
        molecule_id=molecule_id,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        valid_smiles=True,
        source_database="ChEMBL",
        match_type="similarity",
        public_id=chembl_id,
        public_name=str(record.get("pref_name") or ""),
        public_smiles=public_smiles,
        similarity=f"{similarity:.3f}",
        public_url=(
            f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}"
            if chembl_id
            else ""
        ),
        evidence_note=(
            "Closest ChEMBL compound returned by the public similarity service."
        ),
        lookup_status="match_found",
    )


def placeholder_result(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    valid_smiles: bool,
    *,
    source_database: str,
    match_type: str,
    lookup_status: str,
    evidence_note: str,
    error_message: str = "",
) -> LookupResult:
    """Create an invalid, offline, no-match, or error result."""
    public_match_found = "true" if lookup_status == "match_found" else "false"
    source_available = "false" if lookup_status in {"lookup_error", "unavailable_network_error"} else "true"
    return LookupResult(
        molecule_id=molecule_id,
        smiles=canonical_smiles,
        standardized_smiles=canonical_smiles,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        valid_smiles=valid_smiles,
        source=source_database,
        source_database=source_database,
        source_available=source_available,
        public_match_found=public_match_found,
        match_type=match_type,
        evidence_note=evidence_note,
        lookup_status=lookup_status,
        error_type="lookup_error" if lookup_status in {"lookup_error", "unavailable_network_error"} else "",
        error_message=error_message,
        fallback_used="true" if lookup_status in {"lookup_error", "unavailable_network_error"} else "false",
    )


def source_failure_result(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    *,
    source_database: str,
    exc: BaseException | str,
    valid_smiles: bool = True,
) -> LookupResult:
    """Create a degraded public lookup row for one failed source request."""
    return LookupResult(
        molecule_id=molecule_id,
        smiles=canonical_smiles,
        standardized_smiles=canonical_smiles,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        valid_smiles=valid_smiles,
        source=source_database,
        source_database=source_database,
        source_available="false",
        public_match_found="false",
        match_type="lookup_error",
        lookup_status="lookup_error",
        evidence_note=DEGRADED_LOOKUP_NOTE,
        error_type=error_type(exc),
        error_message=sanitize_error_message(exc),
        fallback_used="true",
    )


def lookup_pubchem(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    client: JsonClient,
    timeout: float,
) -> LookupResult:
    """Look up an exact PubChem record by InChIKey."""
    if not inchi_key:
        return placeholder_result(
            molecule_id,
            canonical_smiles,
            inchi_key,
            True,
            source_database="PubChem",
            match_type="no_match",
            lookup_status="no_match",
            evidence_note="No InChIKey was available for exact PubChem lookup.",
        )

    encoded_key = quote(inchi_key, safe="")
    properties = quote(
        "Title,ConnectivitySMILES,CanonicalSMILES,InChIKey", safe=","
    )
    url = (
        f"{PUBCHEM_BASE_URL}/rest/pug/compound/inchikey/{encoded_key}"
        f"/property/{properties}/JSON"
    )
    try:
        payload = client.get_json(url, timeout)
        result = parse_pubchem_exact_match(
            payload, molecule_id, canonical_smiles, inchi_key
        )
    except PublicLookupHttpError as exc:
        if exc.status_code == 404:
            return placeholder_result(
                molecule_id,
                canonical_smiles,
                inchi_key,
                True,
                source_database="PubChem",
                match_type="no_exact_pubchem_match",
                lookup_status="no_match",
                evidence_note="No exact PubChem record found for this molecule.",
            )
        return source_failure_result(
            molecule_id,
            canonical_smiles,
            inchi_key,
            source_database="PubChem",
            exc=exc,
        )
    except Exception as exc:
        return source_failure_result(
            molecule_id,
            canonical_smiles,
            inchi_key,
            source_database="PubChem",
            exc=exc,
        )
    return result or placeholder_result(
        molecule_id,
        canonical_smiles,
        inchi_key,
        True,
        source_database="PubChem",
        match_type="no_exact_pubchem_match",
        lookup_status="no_match",
        evidence_note="No exact PubChem record found for this molecule.",
    )


def lookup_chembl(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    client: JsonClient,
    timeout: float,
) -> LookupResult:
    """Look up the closest ChEMBL compound by canonical SMILES."""
    encoded_smiles = quote(canonical_smiles, safe="")
    url = (
        f"{CHEMBL_BASE_URL}/chembl/api/data/similarity/{encoded_smiles}/"
        f"{CHEMBL_SIMILARITY_THRESHOLD}.json?limit=20"
    )
    try:
        payload = client.get_json(url, timeout)
        result = parse_chembl_similarity_match(
            payload, molecule_id, canonical_smiles, inchi_key
        )
    except Exception as exc:
        return source_failure_result(
            molecule_id,
            canonical_smiles,
            inchi_key,
            source_database="ChEMBL",
            exc=exc,
        )
    return result or placeholder_result(
        molecule_id,
        canonical_smiles,
        inchi_key,
        True,
        source_database="ChEMBL",
        match_type="no_match",
        lookup_status="no_match",
        evidence_note=(
            "No ChEMBL similarity match met the configured service threshold."
        ),
    )


def lookup_rows(
    standardized_rows: Iterable[Mapping[str, str]],
    descriptor_rows: Iterable[Mapping[str, str]],
    *,
    offline: bool,
    max_molecules: int | None,
    client: JsonClient | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[LookupResult]:
    """Merge inputs and produce public lookup evidence rows."""
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("max-molecules must be at least 1.")
    descriptor_index = index_descriptors(descriptor_rows)
    active_client = client or UrllibJsonClient()
    results: list[LookupResult] = []

    selected_rows = list(standardized_rows)
    queried = 0

    for row in selected_rows:
        molecule_id = row.get("molecule_id", "").strip()
        descriptor = descriptor_index.get(molecule_id, {})
        canonical_smiles = (
            row.get("canonical_smiles", "").strip()
            or descriptor.get("canonical_smiles", "").strip()
        )
        inchi_key = row.get("inchi_key", "").strip()
        valid_smiles = parse_boolean(row.get("valid_smiles", ""))

        if not valid_smiles or not canonical_smiles:
            error = (
                row.get("error_message", "").strip()
                or descriptor.get("descriptor_error", "").strip()
                or "Molecule is invalid or has no canonical SMILES."
            )
            results.append(
                placeholder_result(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    False,
                    source_database="not_available",
                    match_type="no_match",
                    lookup_status="invalid_molecule",
                    evidence_note="Public lookup was skipped for an invalid molecule.",
                    error_message=error,
                )
            )
            continue

        if offline:
            results.append(
                placeholder_result(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    True,
                    source_database="offline",
                    match_type="no_match",
                    lookup_status="offline",
                    evidence_note=(
                        "Offline mode enabled; no public API requests were made."
                    ),
                )
            )
            continue

        if max_molecules is not None and queried >= max_molecules:
            for source_database, match_type, note in (
                (
                    "PubChem",
                    "not_queried",
                    "PubChem exact lookup was not queried because of the configured max-molecules limit.",
                ),
                (
                    "ChEMBL",
                    "not_queried",
                    "ChEMBL similarity lookup was not queried because of the configured max-molecules limit.",
                ),
            ):
                results.append(
                    placeholder_result(
                        molecule_id,
                        canonical_smiles,
                        inchi_key,
                        True,
                        source_database=source_database,
                        match_type=match_type,
                        lookup_status="not_queried",
                        evidence_note=note,
                    )
                )
            continue

        queried += 1
        results.append(
            lookup_pubchem(
                molecule_id,
                canonical_smiles,
                inchi_key,
                active_client,
                timeout,
            )
        )
        results.append(
            lookup_chembl(
                molecule_id,
                canonical_smiles,
                inchi_key,
                active_client,
                timeout,
            )
        )
    return results


def write_output_csv(
    output_path: Path, records: Iterable[LookupResult]
) -> None:
    """Write public lookup evidence rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            row["smiles"] = row.get("smiles") or record.canonical_smiles
            row["standardized_smiles"] = row.get("standardized_smiles") or record.canonical_smiles
            row["source"] = row.get("source") or record.source_database
            row["source_available"] = row.get("source_available") or (
                "false" if record.lookup_status in {"lookup_error", "unavailable_network_error"} else "true"
            )
            row["public_match_found"] = row.get("public_match_found") or (
                "true" if record.lookup_status == "match_found" else "false"
            )
            row["error_type"] = row.get("error_type") or (
                "lookup_error" if record.lookup_status in {"lookup_error", "unavailable_network_error"} else ""
            )
            row["fallback_used"] = row.get("fallback_used") or (
                "true" if record.lookup_status in {"lookup_error", "unavailable_network_error"} else "false"
            )
            writer.writerow(row)


def public_lookup_csv(
    standardized_path: Path,
    descriptor_path: Path,
    output_path: Path,
    *,
    offline: bool = False,
    max_molecules: int | None = None,
    client: JsonClient | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Run public lookups and return the output row count."""
    standardized_rows = read_csv_with_columns(
        standardized_path, STANDARDIZED_COLUMNS, "Standardized"
    )
    descriptor_rows = read_csv_with_columns(
        descriptor_path, DESCRIPTOR_COLUMNS, "Descriptor"
    )
    results = lookup_rows(
        standardized_rows,
        descriptor_rows,
        offline=offline,
        max_molecules=max_molecules,
        client=client,
        timeout=timeout,
    )
    temp_path = output_path.with_suffix(".tmp.csv")
    try:
        write_output_csv(temp_path, results)
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Look up generated molecules in public compound databases."
    )
    parser.add_argument("--standardized", required=True, type=Path)
    parser.add_argument("--descriptors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public lookup command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        count = public_lookup_csv(
            args.standardized,
            args.descriptors,
            args.output,
            offline=args.offline,
            max_molecules=args.max_molecules,
            timeout=args.timeout,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {count} public lookup records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
