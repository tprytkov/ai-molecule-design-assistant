import csv
import subprocess
import sys
from pathlib import Path

from src.surechembl_lookup import (
    OUTPUT_COLUMNS,
    SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL,
    SURECHEMBL_SEARCH_TYPE,
    SURECHEMBL_STRUCTURE_SEARCH_URL,
    build_structure_search_payload,
    prepare_patent_compounds,
    surechembl_lookup_csv,
)


class MockSurechemblClient:
    """Track and mock online SureChEMBL API calls."""

    def __init__(self, *, fail: bool = False, metadata: bool = True) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.get_calls: list[str] = []
        self.fail = fail
        self.metadata = metadata

    def post_json(self, url: str, payload, timeout: float, debug: bool = False):
        self.calls.append((url, dict(payload)))
        if self.fail:
            raise RuntimeError("mock timeout")
        if url.startswith(SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL):
            if not self.metadata:
                return {"status": "SUCCESS", "data": {"documents": []}}
            return {
                "status": "SUCCESS",
                "data": {
                    "documents": [
                        {
                            "chemical_id": "SCHEMBL-MOCK-ASPIRIN",
                            "patent_id": "MOCK-PATENT-001",
                            "patent_title": "Mock patent chemistry hit",
                            "publication_date": "2026-01-01",
                            "source_section": "claims",
                        }
                    ]
                },
            }
        return {
            "status": "SUCCESS",
            "data": {
                "results": {
                    "structures": [
                        {
                            "id": "SCHEMBL-MOCK-ASPIRIN",
                            "name": "aspirin",
                            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                            "similarity": 1.0,
                            "source_section": "api_mock",
                        }
                    ]
                }
            },
        }

    def get_json(self, url: str, timeout: float, debug: bool = False):
        self.get_calls.append(url)
        if self.fail:
            raise RuntimeError("mock timeout")
        return {"status": "SUCCESS", "data": {"results": []}}


def write_inputs(descriptors: Path, surechembl: Path) -> None:
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,descriptor_error\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,True,\n"
        "invalid,,False,Invalid molecule.\n",
        encoding="utf-8",
    )
    surechembl.write_text(
        "surechembl_id,patent_id,patent_title,patent_date,compound_name,"
        "smiles,source_section,evidence_note\n"
        "SC1,PAT1,Demo aspirin,2026-01-01,aspirin,"
        "CC(=O)Oc1ccccc1C(=O)O,claims_demo,Demo evidence.\n"
        "SC2,PAT2,Demo benzene,2026-01-01,benzene,c1ccccc1,"
        "abstract_demo,Demo evidence.\n"
        "BAD,PAT3,Bad,2026-01-01,bad,C1CC,abstract_demo,Invalid.\n",
        encoding="utf-8",
    )


def test_lookup_output_created_and_sorted(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    count = surechembl_lookup_csv(descriptors, surechembl, output, top_k=2)

    assert count == 3
    assert output.exists()
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    valid_rows = [row for row in rows if row["molecule_id"] == "aspirin"]
    scores = [float(row["tanimoto_similarity"]) for row in valid_rows]
    assert scores == sorted(scores, reverse=True)


def test_default_mode_remains_local_and_does_not_call_api(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)
    client = MockSurechemblClient()

    surechembl_lookup_csv(
        descriptors,
        surechembl,
        output,
        top_k=1,
        client=client,
    )

    assert client.calls == []
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["surechembl_id"] == "SC1"


def test_exact_local_structure_evidence_hit(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    count = surechembl_lookup_csv(descriptors, surechembl, output, top_k=5)

    assert count >= 1
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    exact = [row for row in rows if row["molecule_id"] == "aspirin"]
    assert exact[0]["compound_name"] == "aspirin"
    assert exact[0]["tanimoto_similarity"] == "1.000"


def test_invalid_generated_molecule_handled_clearly(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    surechembl_lookup_csv(descriptors, surechembl, output, top_k=1)

    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    invalid = [row for row in rows if row["molecule_id"] == "invalid"][0]
    assert invalid["lookup_status"] == "invalid_molecule"
    assert invalid["error_message"] == "Invalid molecule."


def test_online_surechembl_uses_mocked_api_client(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)
    client = MockSurechemblClient()

    count = surechembl_lookup_csv(
        descriptors,
        None,
        output,
        top_k=5,
        online_surechembl=True,
        max_molecules=1,
        client=client,
    )

    assert count == 2
    assert len(client.calls) == 2
    assert client.calls[0][0] == SURECHEMBL_STRUCTURE_SEARCH_URL
    assert client.calls[1][0].startswith(SURECHEMBL_DOCUMENTS_FOR_STRUCTURES_URL)
    assert client.calls[0][1] == build_structure_search_payload(
        "CC(=O)Oc1ccccc1C(=O)O", 5
    )
    assert (
        client.calls[0][1]["StructureSearchRequest"]["structSearchType"]
        == SURECHEMBL_SEARCH_TYPE
    )
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert rows[0]["surechembl_id"] == "SCHEMBL-MOCK-ASPIRIN"
    assert rows[0]["patent_id"] == "MOCK-PATENT-001"
    assert rows[0]["patent_title"] == "Mock patent chemistry hit"
    assert rows[0]["lookup_status"] == "match_found"
    assert rows[1]["lookup_status"] == "invalid_molecule"


def test_missing_follow_up_metadata_does_not_crash(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    surechembl_lookup_csv(
        descriptors,
        None,
        output,
        top_k=5,
        online_surechembl=True,
        max_molecules=1,
        client=MockSurechemblClient(metadata=False),
    )

    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert rows[0]["lookup_status"] == "match_found"
    assert rows[0]["patent_id"] == "not_available"
    assert rows[0]["patent_title"] == "not_available"


def test_online_error_produces_lookup_error(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    surechembl_lookup_csv(
        descriptors,
        None,
        output,
        top_k=5,
        online_surechembl=True,
        client=MockSurechemblClient(fail=True),
    )

    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    error = [row for row in rows if row["molecule_id"] == "aspirin"][0]
    assert error["lookup_status"] == "lookup_error"
    assert "mock timeout" in error["error_message"]


def test_max_molecules_limits_online_queries(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,descriptor_error\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,True,\n"
        "benzene,c1ccccc1,True,\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.csv"
    client = MockSurechemblClient()

    surechembl_lookup_csv(
        descriptors,
        None,
        output,
        top_k=1,
        online_surechembl=True,
        max_molecules=1,
        client=client,
    )

    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(client.calls) == 2
    assert rows[1]["molecule_id"] == "benzene"
    assert rows[1]["lookup_status"] == "not_queried"


def test_invalid_surechembl_smiles_skipped(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    write_inputs(descriptors, surechembl)

    compounds = prepare_patent_compounds(
        list(csv.DictReader(surechembl.open(encoding="utf-8")))
    )

    assert [compound.surechembl_id for compound in compounds] == ["SC1", "SC2"]


def test_cli_writes_output(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    surechembl = tmp_path / "surechembl.csv"
    output = tmp_path / "output.csv"
    write_inputs(descriptors, surechembl)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.surechembl_lookup",
            "--descriptors",
            str(descriptors),
            "--surechembl",
            str(surechembl),
            "--output",
            str(output),
            "--top-k",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
