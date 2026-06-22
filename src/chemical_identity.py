"""Identify standardized molecules using exact public chemical records."""

from __future__ import annotations

import csv
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem

from src.public_lookup import PublicLookupHttpError


OUTPUT_COLUMNS = (
    "molecule_id",
    "smiles",
    "valid_smiles",
    "inchikey",
    "identity_status",
    "pubchem_cid",
    "chembl_id",
    "exact_public_name",
    "iupac_name",
    "preferred_name",
    "synonyms",
    "name_source",
    "identity_confidence",
    "lookup_status",
    "error_message",
)
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
CACTUS_BASE_URL = "https://cactus.nci.nih.gov"
DEFAULT_TIMEOUT = 15.0


class IdentityClient(Protocol):
    """HTTP operations used by online identity lookup."""

    def get_json(self, url: str, timeout: float) -> Mapping[str, object]:
        """Return a decoded JSON object."""

    def get_text(self, url: str, timeout: float) -> str:
        """Return response text."""


class UrllibIdentityClient:
    """Small standard-library client for public chemical-name services."""

    @staticmethod
    def _request(url: str, timeout: float, accept: str) -> bytes:
        request = Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "molecule-intelligence-demo/1.0",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            raise PublicLookupHttpError(
                exc.code, f"HTTP {exc.code} returned by public name service."
            ) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise RuntimeError(f"Public name service request failed: {exc}") from exc

    def get_json(self, url: str, timeout: float) -> Mapping[str, object]:
        try:
            payload = json.loads(self._request(url, timeout, "application/json"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Public name service returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Public name service returned an unexpected response.")
        return payload

    def get_text(self, url: str, timeout: float) -> str:
        try:
            return self._request(url, timeout, "text/plain").decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Public name service returned invalid text."
            ) from exc


@dataclass(frozen=True)
class ChemicalIdentity:
    """One chemical identity output row."""

    molecule_id: str
    smiles: str
    valid_smiles: bool
    inchikey: str = ""
    identity_status: str = "no_public_identity"
    pubchem_cid: str = ""
    chembl_id: str = ""
    exact_public_name: str = ""
    iupac_name: str = ""
    preferred_name: str = ""
    synonyms: str = ""
    name_source: str = ""
    identity_confidence: str = "none"
    lookup_status: str = "no_match"
    error_message: str = ""


def clean(value: object) -> str:
    """Normalize a possibly missing CSV value."""
    return str(value or "").strip()


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return clean(value).lower() in {"true", "1", "yes", "y"}


def unique_join(values: Iterable[object], limit: int = 20) -> str:
    """Join unique nonempty values without inventing or transforming names."""
    seen: set[str] = set()
    names: list[str] = []
    for value in values:
        text = clean(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            names.append(text)
        if len(names) >= limit:
            break
    return "; ".join(names)


def generate_inchikey(smiles: str) -> str:
    """Generate an InChIKey with RDKit for a valid SMILES string."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return Chem.MolToInchiKey(molecule)


def is_exact_lookup(row: Mapping[str, str]) -> bool:
    """Return whether a public lookup row represents exact identity."""
    match_type = clean(row.get("match_type")).lower()
    similarity = clean(row.get("similarity"))
    return clean(row.get("lookup_status")) == "match_found" and (
        match_type in {"exact", "exact_match", "exact_inchikey", "exact_smiles"}
        or similarity in {"1", "1.0", "1.00", "1.000"}
    )


def index_public_rows(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    """Index existing public lookup evidence by molecule ID."""
    indexed: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        indexed.setdefault(clean(row.get("molecule_id")), []).append(row)
    return indexed


def parse_pubchem_properties(
    payload: Mapping[str, object],
) -> tuple[str, str, str]:
    """Return CID, preferred title, and IUPAC name from PubChem properties."""
    table = payload.get("PropertyTable")
    if not isinstance(table, dict):
        return "", "", ""
    properties = table.get("Properties")
    if not isinstance(properties, list) or not properties:
        return "", "", ""
    record = properties[0]
    if not isinstance(record, dict):
        return "", "", ""
    return (
        clean(record.get("CID")),
        clean(record.get("Title")),
        clean(record.get("IUPACName")),
    )


def parse_pubchem_synonyms(payload: Mapping[str, object]) -> list[str]:
    """Return PubChem synonyms from the first information record."""
    information_list = payload.get("InformationList")
    if not isinstance(information_list, dict):
        return []
    information = information_list.get("Information")
    if not isinstance(information, list) or not information:
        return []
    record = information[0]
    if not isinstance(record, dict):
        return []
    synonyms = record.get("Synonym")
    return [clean(value) for value in synonyms] if isinstance(synonyms, list) else []


def pubchem_identity_lookup(
    inchikey: str,
    client: IdentityClient,
    timeout: float,
    known_cid: str = "",
) -> tuple[str, str, str, list[str], str, str]:
    """Fetch exact PubChem identity fields, preserving 404 as a no-match."""
    identifier = known_cid or inchikey
    namespace = "cid" if known_cid else "inchikey"
    encoded = quote(identifier, safe="")
    properties = quote("Title,IUPACName,InChIKey", safe=",")
    property_url = (
        f"{PUBCHEM_BASE_URL}/rest/pug/compound/{namespace}/{encoded}"
        f"/property/{properties}/JSON"
    )
    try:
        cid, preferred_name, iupac_name = parse_pubchem_properties(
            client.get_json(property_url, timeout)
        )
    except PublicLookupHttpError as exc:
        if exc.status_code == 404:
            return "", "", "", [], "no_match", ""
        return "", "", "", [], "lookup_error", str(exc)
    except RuntimeError as exc:
        return "", "", "", [], "lookup_error", str(exc)
    if not cid:
        return "", "", "", [], "no_match", ""

    synonyms_url = (
        f"{PUBCHEM_BASE_URL}/rest/pug/compound/cid/{quote(cid, safe='')}"
        "/synonyms/JSON"
    )
    try:
        synonyms = parse_pubchem_synonyms(client.get_json(synonyms_url, timeout))
    except (PublicLookupHttpError, RuntimeError):
        synonyms = []
    return cid, preferred_name, iupac_name, synonyms, "match_found", ""


def cactus_iupac_lookup(
    smiles: str, client: IdentityClient, timeout: float
) -> tuple[str, str]:
    """Return a Cactus-generated systematic name, if available."""
    url = (
        f"{CACTUS_BASE_URL}/chemical/structure/{quote(smiles, safe='')}"
        "/iupac_name"
    )
    try:
        name = clean(client.get_text(url, timeout))
    except PublicLookupHttpError as exc:
        if exc.status_code == 404:
            return "", ""
        return "", str(exc)
    except RuntimeError as exc:
        return "", str(exc)
    return name, ""


def identify_rows(
    standardized_rows: Iterable[Mapping[str, str]],
    public_rows: Iterable[Mapping[str, str]] = (),
    *,
    online: bool = False,
    max_molecules: int | None = None,
    client: IdentityClient | None = None,
    use_cactus: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[ChemicalIdentity]:
    """Identify standardized molecules without assigning unsupported names."""
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("max-molecules must be at least 1.")
    active_client = client or UrllibIdentityClient()
    public_by_id = index_public_rows(public_rows)
    results: list[ChemicalIdentity] = []
    queried = 0

    for row in standardized_rows:
        molecule_id = clean(row.get("molecule_id"))
        smiles = clean(row.get("canonical_smiles")) or clean(row.get("smiles"))
        valid = parse_boolean(row.get("valid_smiles"))
        if not valid or not smiles:
            results.append(
                ChemicalIdentity(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    valid_smiles=False,
                    identity_status="invalid_smiles",
                    lookup_status="invalid_smiles",
                    error_message=clean(row.get("error_message")),
                )
            )
            continue

        inchikey = generate_inchikey(smiles)
        existing = public_by_id.get(molecule_id, [])
        exact_pubchem = [
            item
            for item in existing
            if clean(item.get("source_database")).casefold() == "pubchem"
            and is_exact_lookup(item)
        ]
        exact_chembl = [
            item
            for item in existing
            if clean(item.get("source_database")).casefold() == "chembl"
            and is_exact_lookup(item)
        ]
        pubchem_id = clean(exact_pubchem[0].get("public_id")) if exact_pubchem else ""
        pubchem_cid = pubchem_id.removeprefix("CID:")
        pubchem_name = clean(exact_pubchem[0].get("public_name")) if exact_pubchem else ""
        chembl_id = clean(exact_chembl[0].get("public_id")) if exact_chembl else ""
        chembl_name = clean(exact_chembl[0].get("public_name")) if exact_chembl else ""

        if online and (max_molecules is None or queried < max_molecules):
            queried += 1
            cid, preferred, iupac, synonyms, status, error = pubchem_identity_lookup(
                inchikey, active_client, timeout, pubchem_cid
            )
            pubchem_cid = cid or pubchem_cid
            pubchem_name = preferred or pubchem_name
            if pubchem_cid:
                exact_name = pubchem_name or (synonyms[0] if synonyms else "")
                results.append(
                    ChemicalIdentity(
                        molecule_id=molecule_id,
                        smiles=smiles,
                        valid_smiles=True,
                        inchikey=inchikey,
                        identity_status="exact_public_identity",
                        pubchem_cid=pubchem_cid,
                        chembl_id=chembl_id,
                        exact_public_name=exact_name,
                        iupac_name=iupac,
                        preferred_name=pubchem_name or chembl_name,
                        synonyms=unique_join(synonyms),
                        name_source="PubChem",
                        identity_confidence="high",
                        lookup_status="match_found",
                    )
                )
                continue
            if chembl_id and chembl_name:
                results.append(
                    ChemicalIdentity(
                        molecule_id=molecule_id,
                        smiles=smiles,
                        valid_smiles=True,
                        inchikey=inchikey,
                        identity_status="exact_public_identity",
                        chembl_id=chembl_id,
                        exact_public_name=chembl_name,
                        preferred_name=chembl_name,
                        name_source="ChEMBL",
                        identity_confidence="high",
                        lookup_status="match_found",
                    )
                )
                continue
            if status == "lookup_error":
                results.append(
                    ChemicalIdentity(
                        molecule_id=molecule_id,
                        smiles=smiles,
                        valid_smiles=True,
                        inchikey=inchikey,
                        identity_status="lookup_error",
                        lookup_status="lookup_error",
                        error_message=error,
                    )
                )
                continue
            if use_cactus:
                generated_name, cactus_error = cactus_iupac_lookup(
                    smiles, active_client, timeout
                )
                if generated_name:
                    results.append(
                        ChemicalIdentity(
                            molecule_id=molecule_id,
                            smiles=smiles,
                            valid_smiles=True,
                            inchikey=inchikey,
                            identity_status="generated_iupac_name_only",
                            iupac_name=generated_name,
                            name_source="NCI Cactus",
                            identity_confidence="generated",
                            lookup_status="no_match",
                        )
                    )
                    continue
                error = cactus_error
            results.append(
                ChemicalIdentity(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    valid_smiles=True,
                    inchikey=inchikey,
                    identity_status="no_public_identity",
                    lookup_status="no_match",
                    error_message=error,
                )
            )
            continue

        if pubchem_cid or (chembl_id and chembl_name):
            source = "PubChem" if pubchem_cid else "ChEMBL"
            name = pubchem_name if pubchem_cid else chembl_name
            results.append(
                ChemicalIdentity(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    valid_smiles=True,
                    inchikey=inchikey,
                    identity_status="exact_public_identity",
                    pubchem_cid=pubchem_cid,
                    chembl_id=chembl_id,
                    exact_public_name=name,
                    preferred_name=name,
                    name_source=source,
                    identity_confidence="high",
                    lookup_status="match_found",
                )
            )
        else:
            status = (
                "not_queried"
                if online and max_molecules is not None and queried >= max_molecules
                else "offline"
            )
            results.append(
                ChemicalIdentity(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    valid_smiles=True,
                    inchikey=inchikey,
                    identity_status="no_public_identity",
                    lookup_status=status,
                )
            )
    return results


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def write_identity_csv(
    output_path: Path, rows: Iterable[ChemicalIdentity]
) -> None:
    """Write chemical identity rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in rows:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def chemical_identity_csv(
    standardized_path: Path,
    output_path: Path,
    *,
    public_lookup_path: Path | None = None,
    online: bool = False,
    max_molecules: int | None = None,
    client: IdentityClient | None = None,
    use_cactus: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Create ``chemical_identity.csv`` and return its row count."""
    public_rows = (
        read_csv(public_lookup_path)
        if public_lookup_path is not None and public_lookup_path.exists()
        else []
    )
    results = identify_rows(
        read_csv(standardized_path),
        public_rows,
        online=online,
        max_molecules=max_molecules,
        client=client,
        use_cactus=use_cactus,
        timeout=timeout,
    )
    write_identity_csv(output_path, results)
    return len(results)
