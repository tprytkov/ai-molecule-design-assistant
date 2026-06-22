"""Search public US patent metadata for compound-related text evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PATENTSVIEW_URL = "https://search.patentsview.org/api/v1/patent/"
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_RESULTS = 10
API_KEY_ENV = "PATENTSVIEW_API_KEY"

STANDARDIZED_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "inchi_key",
    "valid_smiles",
)
PUBLIC_LOOKUP_COLUMNS = (
    "molecule_id",
    "source_database",
    "public_name",
    "lookup_status",
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "inchi_key",
    "search_terms",
    "source_database",
    "patent_id",
    "patent_title",
    "patent_date",
    "assignee",
    "inventors",
    "abstract_excerpt",
    "public_url",
    "evidence_note",
    "search_status",
    "error_message",
)
OFFLINE_DEMO_PATENT_EVIDENCE = {
    "demo_aspirin": {
        "patent_id": "DEMO-PATENT-ASPIRIN-001",
        "patent_title": (
            "Synthetic demonstration record for small-molecule "
            "pharmaceutical composition"
        ),
        "patent_date": "2026-01-01",
        "abstract_excerpt": (
            "Public-safe demo text only. This synthetic patent-style evidence "
            "record illustrates how a small-molecule composition mention could "
            "be summarized in the local workflow. It is not a real patent "
            "claim and does not imply real patent coverage."
        ),
    },
    "demo_caffeine": {
        "patent_id": "DEMO-PATENT-CAFFEINE-001",
        "patent_title": (
            "Synthetic demonstration record for small-molecule bioactivity "
            "context"
        ),
        "patent_date": "2026-01-01",
        "abstract_excerpt": (
            "Public-safe demo text only. This synthetic patent-style evidence "
            "record shows how text mentioning a well-known small molecule and "
            "general bioactivity context can be retained for reporting. It is "
            "not a real patent claim and does not imply real patent coverage."
        ),
    },
    "demo_benzene": {
        "patent_id": "DEMO-PATENT-BENZENE-001",
        "patent_title": (
            "Synthetic low-relevance demonstration record for a simple "
            "aromatic structure"
        ),
        "patent_date": "2026-01-01",
        "abstract_excerpt": (
            "Public-safe demo text only. This low-relevance synthetic "
            "patent-style evidence record is included to demonstrate reporting "
            "behavior for a simple reference molecule. It is not a real patent "
            "claim and does not imply real patent coverage."
        ),
    },
}


class PatentClient(Protocol):
    """Minimal client interface for PatentsView and mocked tests."""

    def search(
        self,
        query: Mapping[str, object],
        fields: Sequence[str],
        max_results: int,
        timeout: float,
    ) -> Mapping[str, object]:
        """Return a decoded PatentsView response."""


class PatentsViewClient:
    """PatentsView client using an API key supplied at runtime."""

    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError(
                f"PatentsView API key is required in {API_KEY_ENV}."
            )
        self.api_key = api_key.strip()

    def search(
        self,
        query: Mapping[str, object],
        fields: Sequence[str],
        max_results: int,
        timeout: float,
    ) -> Mapping[str, object]:
        parameters = urlencode(
            {
                "q": json.dumps(query, separators=(",", ":")),
                "f": json.dumps(list(fields), separators=(",", ":")),
                "s": json.dumps(
                    [{"patent_date": "desc"}], separators=(",", ":")
                ),
                "o": json.dumps(
                    {"size": max_results}, separators=(",", ":")
                ),
            }
        )
        request = Request(
            f"{PATENTSVIEW_URL}?{parameters}",
            headers={
                "Accept": "application/json",
                "X-Api-Key": self.api_key,
                "User-Agent": "molecule-intelligence-demo/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(
                f"PatentsView returned HTTP {exc.code}."
            ) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise RuntimeError(f"PatentsView request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("PatentsView returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("PatentsView returned an unexpected response.")
        return payload


@dataclass(frozen=True)
class PatentEvidence:
    """One patent-search evidence row."""

    molecule_id: str
    canonical_smiles: str
    inchi_key: str
    search_terms: str
    source_database: str = "PatentsView"
    patent_id: str = ""
    patent_title: str = ""
    patent_date: str = ""
    assignee: str = ""
    inventors: str = ""
    abstract_excerpt: str = ""
    public_url: str = ""
    evidence_note: str = ""
    search_status: str = ""
    error_message: str = ""


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def collect_public_names(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Collect unique successful PubChem names by molecule ID."""
    names: dict[str, list[str]] = {}
    for row in rows:
        if row.get("lookup_status", "").strip() != "match_found":
            continue
        if row.get("source_database", "").strip() != "PubChem":
            continue
        molecule_id = row.get("molecule_id", "").strip()
        public_name = row.get("public_name", "").strip()
        if public_name and public_name not in names.setdefault(molecule_id, []):
            names[molecule_id].append(public_name)
    return names


def build_search_terms(
    molecule_id: str,
    inchi_key: str,
    public_names: Mapping[str, Sequence[str]],
) -> list[str]:
    """Build public-safe text terms from public names and an InChIKey."""
    terms = [
        name.strip()
        for name in public_names.get(molecule_id, ())
        if name.strip()
    ]
    if inchi_key.strip():
        terms.append(inchi_key.strip())
    return list(dict.fromkeys(terms))


def build_patent_query(terms: Sequence[str]) -> dict[str, object]:
    """Build a PatentsView title/abstract phrase query."""
    criteria: list[dict[str, object]] = []
    for term in terms:
        criteria.extend(
            [
                {"_text_phrase": {"patent_title": term}},
                {"_text_phrase": {"patent_abstract": term}},
            ]
        )
    if not criteria:
        raise ValueError("At least one patent search term is required.")
    return {"_or": criteria}


def flatten_names(value: object, keys: Sequence[str]) -> str:
    """Flatten nested assignee or inventor names from API records."""
    if not isinstance(value, list):
        return ""
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        parts = [str(item.get(key) or "").strip() for key in keys]
        name = " ".join(part for part in parts if part)
        if name and name not in names:
            names.append(name)
    return "; ".join(names)


def parse_patents(
    payload: Mapping[str, object],
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    terms: Sequence[str],
) -> list[PatentEvidence]:
    """Parse PatentsView patent records."""
    records = payload.get("patents")
    if not isinstance(records, list):
        return []

    results: list[PatentEvidence] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        patent_id = str(record.get("patent_id") or "").strip()
        if not patent_id:
            continue
        abstract = str(record.get("patent_abstract") or "").strip()
        excerpt = abstract[:500] + ("..." if len(abstract) > 500 else "")
        results.append(
            PatentEvidence(
                molecule_id=molecule_id,
                canonical_smiles=canonical_smiles,
                inchi_key=inchi_key,
                search_terms="; ".join(terms),
                patent_id=patent_id,
                patent_title=str(record.get("patent_title") or "").strip(),
                patent_date=str(record.get("patent_date") or "").strip(),
                assignee=flatten_names(
                    record.get("assignees"),
                    ("assignee_organization", "assignee_first_name", "assignee_last_name"),
                ),
                inventors=flatten_names(
                    record.get("inventors"),
                    ("inventor_name_first", "inventor_name_last"),
                ),
                abstract_excerpt=excerpt,
                public_url=f"https://patents.google.com/patent/US{patent_id}",
                evidence_note=(
                    "Text search hit in public US patent metadata or abstract. "
                    "A hit does not establish that the molecular structure is claimed."
                ),
                search_status="match_found",
            )
        )
    return results


def placeholder(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    terms: Sequence[str],
    status: str,
    note: str,
    error: str = "",
) -> PatentEvidence:
    """Create an offline, invalid, no-match, or error row."""
    return PatentEvidence(
        molecule_id=molecule_id,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        search_terms="; ".join(terms),
        evidence_note=note,
        search_status=status,
        error_message=error,
    )


def offline_demo_evidence(
    molecule_id: str,
    canonical_smiles: str,
    inchi_key: str,
    terms: Sequence[str],
) -> PatentEvidence | None:
    """Return synthetic patent-style evidence for public demo molecules."""
    demo = OFFLINE_DEMO_PATENT_EVIDENCE.get(molecule_id)
    if demo is None:
        return None
    return PatentEvidence(
        molecule_id=molecule_id,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        search_terms="; ".join(terms),
        patent_id=demo["patent_id"],
        patent_title=demo["patent_title"],
        patent_date=demo["patent_date"],
        abstract_excerpt=demo["abstract_excerpt"],
        evidence_note=(
            "Public-safe demonstration patent-style evidence generated in "
            "offline mode. This is synthetic demo text and not a legal "
            "determination."
        ),
        search_status="match_found",
    )


def search_rows(
    standardized_rows: Iterable[Mapping[str, str]],
    public_lookup_rows: Iterable[Mapping[str, str]],
    *,
    offline: bool,
    max_molecules: int | None,
    max_results: int,
    client: PatentClient | None,
    timeout: float,
) -> list[PatentEvidence]:
    """Search patent text for each selected generated molecule."""
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("max-molecules must be at least 1.")
    if max_results < 1 or max_results > 1000:
        raise ValueError("max-results must be between 1 and 1000.")

    names = collect_public_names(public_lookup_rows)
    selected = list(standardized_rows)
    if max_molecules is not None:
        selected = selected[:max_molecules]

    results: list[PatentEvidence] = []
    for row in selected:
        molecule_id = row.get("molecule_id", "").strip()
        canonical_smiles = row.get("canonical_smiles", "").strip()
        inchi_key = row.get("inchi_key", "").strip()
        terms = build_search_terms(molecule_id, inchi_key, names)

        if not parse_boolean(row.get("valid_smiles", "")):
            results.append(
                placeholder(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    terms,
                    "invalid_molecule",
                    "Patent search was skipped for an invalid molecule.",
                    row.get("error_message", "").strip(),
                )
            )
            continue
        if not terms:
            results.append(
                placeholder(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    terms,
                    "no_search_terms",
                    "No public compound name or InChIKey was available.",
                )
            )
            continue
        if offline:
            demo_evidence = offline_demo_evidence(
                molecule_id,
                canonical_smiles,
                inchi_key,
                terms,
            )
            if demo_evidence is not None:
                results.append(demo_evidence)
                continue
            results.append(
                placeholder(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    terms,
                    "offline",
                    "Offline mode enabled; no patent API request was made.",
                )
            )
            continue
        if client is None:
            raise ValueError("A patent API client is required in online mode.")

        try:
            payload = client.search(
                build_patent_query(terms),
                (
                    "patent_id",
                    "patent_title",
                    "patent_date",
                    "patent_abstract",
                    "assignees.assignee_organization",
                    "assignees.assignee_first_name",
                    "assignees.assignee_last_name",
                    "inventors.inventor_name_first",
                    "inventors.inventor_name_last",
                ),
                max_results,
                timeout,
            )
            matches = parse_patents(
                payload,
                molecule_id,
                canonical_smiles,
                inchi_key,
                terms,
            )
        except RuntimeError as exc:
            results.append(
                placeholder(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    terms,
                    "lookup_error",
                    "Public patent search could not be completed.",
                    str(exc),
                )
            )
            continue

        results.extend(
            matches
            or [
                placeholder(
                    molecule_id,
                    canonical_smiles,
                    inchi_key,
                    terms,
                    "no_match",
                    "No PatentsView title or abstract matches were returned.",
                )
            ]
        )
    return results


def write_output_csv(
    output_path: Path, records: Iterable[PatentEvidence]
) -> None:
    """Write patent evidence rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def patent_search_csv(
    standardized_path: Path,
    public_lookup_path: Path,
    output_path: Path,
    *,
    offline: bool = False,
    max_molecules: int | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: float = DEFAULT_TIMEOUT,
    client: PatentClient | None = None,
    api_key: str | None = None,
) -> int:
    """Run patent searches and return the number of output rows."""
    standardized_rows = read_csv_with_columns(
        standardized_path, STANDARDIZED_COLUMNS, "Standardized"
    )
    public_lookup_rows = read_csv_with_columns(
        public_lookup_path, PUBLIC_LOOKUP_COLUMNS, "Public lookup"
    )
    active_client = client
    if not offline and active_client is None:
        active_client = PatentsViewClient(
            api_key if api_key is not None else os.getenv(API_KEY_ENV, "")
        )
    results = search_rows(
        standardized_rows,
        public_lookup_rows,
        offline=offline,
        max_molecules=max_molecules,
        max_results=max_results,
        client=active_client,
        timeout=timeout,
    )
    write_output_csv(output_path, results)
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Search public US patent text for compound evidence."
    )
    parser.add_argument("--standardized", required=True, type=Path)
    parser.add_argument("--public-lookup", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the patent-search command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        count = patent_search_csv(
            args.standardized,
            args.public_lookup,
            args.output,
            offline=args.offline,
            max_molecules=args.max_molecules,
            max_results=args.max_results,
            timeout=args.timeout,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {count} patent evidence records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
