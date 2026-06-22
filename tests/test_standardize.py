import csv
from pathlib import Path

import pytest

from src.standardize import standardize_csv, standardize_record, standardize_rows


@pytest.mark.parametrize(
    ("molecule_id", "smiles", "expected_formula"),
    [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "C8H10N4O2"),
        ("benzene", "c1ccccc1", "C6H6"),
    ],
)
def test_standardize_valid_public_molecules(
    molecule_id: str, smiles: str, expected_formula: str
) -> None:
    result = standardize_record(molecule_id, smiles)

    assert result.valid_smiles is True
    assert result.canonical_smiles
    assert len(result.inchi_key) == 27
    assert result.molecular_formula == expected_formula
    assert float(result.molecular_weight) > 0
    assert result.error_message == ""


def test_invalid_smiles_is_retained_with_error() -> None:
    result = standardize_record("invalid", "C1CC")

    assert result.valid_smiles is False
    assert result.canonical_smiles == ""
    assert result.inchi_key == ""
    assert result.error_message


def test_duplicate_canonical_smiles_are_removed_after_standardization() -> None:
    rows = [
        {"molecule_id": "aspirin_1", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"molecule_id": "aspirin_2", "smiles": "O=C(O)c1ccccc1OC(C)=O"},
        {"molecule_id": "invalid_1", "smiles": "C1CC"},
        {"molecule_id": "invalid_2", "smiles": "C1CC"},
    ]

    results = standardize_rows(rows)

    assert [result.molecule_id for result in results] == [
        "aspirin_1",
        "invalid_1",
        "invalid_2",
    ]


def test_standardize_csv_writes_expected_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "molecule_id,smiles\n"
        "benzene,c1ccccc1\n"
        "benzene_duplicate,C1=CC=CC=C1\n"
        "invalid,C1CC\n",
        encoding="utf-8",
    )

    row_count = standardize_csv(input_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        output_rows = list(csv.DictReader(output_file))

    assert row_count == 2
    assert len(output_rows) == 2
    assert output_rows[0]["molecule_id"] == "benzene"
    assert output_rows[0]["valid_smiles"] == "True"
    assert output_rows[1]["molecule_id"] == "invalid"
    assert output_rows[1]["valid_smiles"] == "False"
    assert output_rows[1]["error_message"]
