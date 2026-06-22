import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.patent_search import (
    OUTPUT_COLUMNS,
    build_patent_query,
    parse_patents,
    patent_search_csv,
    search_rows,
)


class MockPatentClient:
    def __init__(self, payload=None, error: str = ""):
        self.payload = payload or {}
        self.error = error
        self.calls = 0

    def search(self, query, fields, max_results, timeout):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return self.payload


def standardized_rows():
    return [
        {
            "molecule_id": "demo",
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


def public_lookup_rows():
    return [
        {
            "molecule_id": "demo",
            "source_database": "PubChem",
            "public_name": "Ethanol",
            "lookup_status": "match_found",
        }
    ]


def test_query_uses_public_name_and_inchikey() -> None:
    query = build_patent_query(
        ["Ethanol", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]
    )
    criteria = query["_or"]
    assert isinstance(criteria, list)
    assert len(criteria) == 4


def test_patent_response_parsing() -> None:
    payload = {
        "patents": [
            {
                "patent_id": "1234567",
                "patent_title": "Ethanol composition",
                "patent_date": "2025-01-01",
                "patent_abstract": "A public patent abstract mentioning ethanol.",
                "assignees": [
                    {"assignee_organization": "Example Organization"}
                ],
                "inventors": [
                    {
                        "inventor_name_first": "Alex",
                        "inventor_name_last": "Smith",
                    }
                ],
            }
        ]
    }
    rows = parse_patents(
        payload,
        "demo",
        "CCO",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        ["Ethanol"],
    )
    assert len(rows) == 1
    assert rows[0].patent_id == "1234567"
    assert rows[0].assignee == "Example Organization"
    assert rows[0].inventors == "Alex Smith"
    assert "does not establish" in rows[0].evidence_note


def test_offline_and_invalid_rows_are_retained() -> None:
    client = MockPatentClient(error="must not be called")
    rows = search_rows(
        standardized_rows(),
        public_lookup_rows(),
        offline=True,
        max_molecules=None,
        max_results=5,
        client=client,
        timeout=1,
    )
    assert client.calls == 0
    assert rows[0].search_status == "offline"
    assert rows[1].search_status == "invalid_molecule"


def test_offline_demo_patent_evidence_rows_created() -> None:
    rows = search_rows(
        [
            {
                "molecule_id": "demo_aspirin",
                "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "valid_smiles": "True",
                "error_message": "",
            },
            {
                "molecule_id": "demo_caffeine",
                "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "inchi_key": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                "valid_smiles": "True",
                "error_message": "",
            },
        ],
        [],
        offline=True,
        max_molecules=None,
        max_results=5,
        client=MockPatentClient(error="must not be called"),
        timeout=1,
    )

    assert [row.search_status for row in rows] == ["match_found", "match_found"]
    assert rows[0].patent_id == "DEMO-PATENT-ASPIRIN-001"
    assert "demonstration patent-style evidence" in rows[0].evidence_note.lower()
    assert "not a real patent claim" in rows[0].abstract_excerpt


def test_api_error_is_retained() -> None:
    rows = search_rows(
        standardized_rows()[:1],
        public_lookup_rows(),
        offline=False,
        max_molecules=None,
        max_results=5,
        client=MockPatentClient(error="service unavailable"),
        timeout=1,
    )
    assert rows[0].search_status == "lookup_error"
    assert "service unavailable" in rows[0].error_message


def write_inputs(standardized: Path, public_lookup: Path) -> None:
    standardized.write_text(
        "molecule_id,canonical_smiles,inchi_key,valid_smiles,error_message\n"
        "demo,CCO,LFQSCWFLJHTTHZ-UHFFFAOYSA-N,True,\n"
        "invalid,,,False,Invalid structure.\n",
        encoding="utf-8",
    )
    public_lookup.write_text(
        "molecule_id,source_database,public_name,lookup_status\n"
        "demo,PubChem,Ethanol,match_found\n",
        encoding="utf-8",
    )


def test_output_columns_and_mocked_online_search(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    public_lookup = tmp_path / "lookup.csv"
    output = tmp_path / "patents.csv"
    write_inputs(standardized, public_lookup)
    client = MockPatentClient(
        payload={
            "patents": [
                {
                    "patent_id": "123",
                    "patent_title": "Demo",
                    "patent_date": "2025-01-01",
                    "patent_abstract": "Demo abstract.",
                }
            ]
        }
    )
    count = patent_search_csv(
        standardized,
        public_lookup,
        output,
        client=client,
        max_molecules=1,
    )
    assert count == 1
    with output.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_offline_output(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    public_lookup = tmp_path / "lookup.csv"
    output = tmp_path / "patents.csv"
    write_inputs(standardized, public_lookup)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.patent_search",
            "--standardized",
            str(standardized),
            "--public-lookup",
            str(public_lookup),
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
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["search_status"] == "offline"


def test_online_mode_requires_api_key(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    public_lookup = tmp_path / "lookup.csv"
    output = tmp_path / "patents.csv"
    write_inputs(standardized, public_lookup)
    with pytest.raises(ValueError, match="PATENTSVIEW_API_KEY"):
        patent_search_csv(
            standardized, public_lookup, output, api_key=""
        )
