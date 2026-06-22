import csv
import subprocess
import sys
from pathlib import Path

from src.public_lookup import (
    OUTPUT_COLUMNS,
    PublicLookupHttpError,
    lookup_rows,
    parse_chembl_similarity_match,
    parse_pubchem_exact_match,
    public_lookup_csv,
)


class MockClient:
    def __init__(self, responses=None, error: str = ""):
        self.responses = responses or {}
        self.error = error
        self.urls: list[str] = []

    def get_json(self, url: str, timeout: float):
        self.urls.append(url)
        if self.error:
            raise RuntimeError(self.error)
        for key, response in self.responses.items():
            if key in url:
                return response
        return {}


class PubChemStatusClient:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def get_json(self, url: str, timeout: float):
        if "pubchem" in url.lower():
            raise PublicLookupHttpError(
                self.status_code,
                f"HTTP {self.status_code} returned by public database.",
            )
        return {"molecules": []}


def standardized_rows():
    return [
        {
            "molecule_id": "valid",
            "canonical_smiles": "CCO",
            "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "valid_smiles": "True",
            "error_message": "",
        },
        {
            "molecule_id": "invalid",
            "canonical_smiles": "",
            "inchi_key": "",
            "valid_smiles": "False",
            "error_message": "Invalid structure.",
        },
    ]


def descriptor_rows():
    return [
        {
            "molecule_id": "valid",
            "canonical_smiles": "CCO",
            "valid_smiles": "True",
        },
        {
            "molecule_id": "invalid",
            "canonical_smiles": "",
            "valid_smiles": "False",
        },
    ]


def test_invalid_molecule_is_retained() -> None:
    results = lookup_rows(
        standardized_rows(),
        descriptor_rows(),
        offline=True,
        max_molecules=None,
    )

    invalid = [row for row in results if row.molecule_id == "invalid"][0]
    assert invalid.lookup_status == "invalid_molecule"
    assert invalid.error_message == "Invalid structure."


def test_offline_mode_makes_no_api_calls() -> None:
    client = MockClient(error="should not be called")

    results = lookup_rows(
        standardized_rows()[:1],
        descriptor_rows()[:1],
        offline=True,
        max_molecules=None,
        client=client,
    )

    assert client.urls == []
    assert results[0].source_database == "offline"
    assert results[0].lookup_status == "offline"


def test_pubchem_exact_match_parsing() -> None:
    payload = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 702,
                    "Title": "Ethanol",
                    "ConnectivitySMILES": "CCO",
                }
            ]
        }
    }

    result = parse_pubchem_exact_match(
        payload, "valid", "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    )

    assert result is not None
    assert result.match_type == "exact_inchikey"
    assert result.public_id == "CID:702"
    assert result.public_name == "Ethanol"
    assert result.similarity == "1.000"


def test_chembl_similarity_parsing_selects_closest() -> None:
    payload = {
        "molecules": [
            {
                "molecule_chembl_id": "CHEMBL1",
                "pref_name": "First",
                "similarity": "72.5",
                "molecule_structures": {"canonical_smiles": "CC"},
            },
            {
                "molecule_chembl_id": "CHEMBL2",
                "pref_name": "Closest",
                "similarity": "91.2",
                "molecule_structures": {"canonical_smiles": "CCO"},
            },
        ]
    }

    result = parse_chembl_similarity_match(
        payload, "valid", "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    )

    assert result is not None
    assert result.public_id == "CHEMBL2"
    assert result.similarity == "0.912"
    assert result.match_type == "similarity"


def test_api_errors_become_output_rows() -> None:
    results = lookup_rows(
        standardized_rows()[:1],
        descriptor_rows()[:1],
        offline=False,
        max_molecules=None,
        client=MockClient(error="service unavailable"),
    )

    assert len(results) == 2
    assert all(row.lookup_status == "lookup_error" for row in results)
    assert all("service unavailable" in row.error_message for row in results)


def test_pubchem_404_is_no_exact_match() -> None:
    results = lookup_rows(
        standardized_rows()[:1],
        descriptor_rows()[:1],
        offline=False,
        max_molecules=None,
        client=PubChemStatusClient(404),
    )

    pubchem = [row for row in results if row.source_database == "PubChem"][0]
    assert pubchem.lookup_status == "no_match"
    assert pubchem.match_type == "no_exact_pubchem_match"
    assert pubchem.evidence_note == "No exact PubChem record found for this molecule."


def test_pubchem_server_error_is_lookup_error() -> None:
    results = lookup_rows(
        standardized_rows()[:1],
        descriptor_rows()[:1],
        offline=False,
        max_molecules=None,
        client=PubChemStatusClient(500),
    )

    pubchem = [row for row in results if row.source_database == "PubChem"][0]
    assert pubchem.lookup_status == "lookup_error"
    assert pubchem.match_type == "lookup_error"


def write_inputs(standardized: Path, descriptors: Path) -> None:
    standardized.write_text(
        "molecule_id,canonical_smiles,inchi_key,valid_smiles,error_message\n"
        "valid,CCO,LFQSCWFLJHTTHZ-UHFFFAOYSA-N,True,\n"
        "invalid,,,False,Invalid structure.\n",
        encoding="utf-8",
    )
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles\n"
        "valid,CCO,True\n"
        "invalid,,False\n",
        encoding="utf-8",
    )


def test_output_columns_exist(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    descriptors = tmp_path / "descriptors.csv"
    output = tmp_path / "lookup.csv"
    write_inputs(standardized, descriptors)

    public_lookup_csv(
        standardized, descriptors, output, offline=True
    )

    with output.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_offline_output(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    descriptors = tmp_path / "descriptors.csv"
    output = tmp_path / "lookup.csv"
    write_inputs(standardized, descriptors)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.public_lookup",
            "--standardized",
            str(standardized),
            "--descriptors",
            str(descriptors),
            "--output",
            str(output),
            "--offline",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
