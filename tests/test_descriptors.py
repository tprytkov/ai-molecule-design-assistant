import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.descriptors import (
    OUTPUT_COLUMNS,
    calculate_descriptors,
    count_lipinski_violations,
    descriptor_csv,
    passes_lipinski,
    summarize_druglikeness,
)


@pytest.mark.parametrize(
    ("molecule_id", "canonical_smiles", "expected_hbd", "expected_hba"),
    [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", "1", "4"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "0", "6"),
        ("benzene", "c1ccccc1", "0", "0"),
    ],
)
def test_descriptors_for_public_molecules(
    molecule_id: str,
    canonical_smiles: str,
    expected_hbd: str,
    expected_hba: str,
) -> None:
    result = calculate_descriptors(molecule_id, canonical_smiles)

    assert result.valid_smiles is True
    assert float(result.molecular_weight) > 0
    assert result.hbd == expected_hbd
    assert result.hba == expected_hba
    assert result.lipinski_violations == "0"
    assert result.lipinski_pass == "True"
    assert result.druglikeness_category in {
        "favorable",
        "borderline",
        "unfavorable",
    }
    assert result.druglikeness_score
    assert result.descriptor_error == ""


def test_invalid_molecule_is_retained_with_error() -> None:
    result = calculate_descriptors(
        molecule_id="invalid",
        canonical_smiles="",
        valid_smiles=False,
        upstream_error="RDKit could not parse the SMILES.",
    )

    assert result.molecule_id == "invalid"
    assert result.valid_smiles is False
    assert result.molecular_weight == ""
    assert result.lipinski_pass == ""
    assert result.druglikeness_category == "invalid"
    assert result.mw_status == "invalid"
    assert result.descriptor_error == "RDKit could not parse the SMILES."


def test_missing_smiles_does_not_crash() -> None:
    result = calculate_descriptors("missing", "", valid_smiles=True)

    assert result.valid_smiles is False
    assert result.descriptor_error == "Canonical SMILES is missing."


def test_lipinski_pass_and_fail_logic() -> None:
    passing_violations = count_lipinski_violations(499.9, 5.0, 5, 10)
    failing_violations = count_lipinski_violations(501.0, 5.1, 6, 11)

    assert passing_violations == 0
    assert passes_lipinski(passing_violations) is True
    assert failing_violations == 4
    assert passes_lipinski(failing_violations) is False


def test_druglikeness_categories_follow_property_statuses() -> None:
    favorable = summarize_druglikeness(
        molecular_weight=350,
        logp=2.5,
        tpsa=80,
        hbd=2,
        hba=5,
        rotatable_bonds=4,
        qed=0.75,
        lipinski_pass=True,
    )
    borderline = summarize_druglikeness(
        molecular_weight=550,
        logp=5.5,
        tpsa=160,
        hbd=2,
        hba=5,
        rotatable_bonds=4,
        qed=0.50,
        lipinski_pass=True,
    )
    unfavorable = summarize_druglikeness(
        molecular_weight=700,
        logp=7,
        tpsa=200,
        hbd=7,
        hba=12,
        rotatable_bonds=12,
        qed=0.20,
        lipinski_pass=False,
    )

    assert favorable[0] == "favorable"
    assert favorable[1] == pytest.approx(1.0)
    assert borderline[0] == "borderline"
    assert "MW: borderline" in borderline[2]
    assert unfavorable[0] == "unfavorable"
    assert unfavorable[3]["Lipinski"] == "unfavorable"


def test_output_columns_exist(tmp_path: Path) -> None:
    input_path = tmp_path / "standardized.csv"
    output_path = tmp_path / "descriptors.csv"
    input_path.write_text(
        "molecule_id,canonical_smiles,valid_smiles,error_message\n"
        "benzene,c1ccccc1,True,\n",
        encoding="utf-8",
    )

    descriptor_csv(input_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_expected_output_file(tmp_path: Path) -> None:
    input_path = tmp_path / "standardized.csv"
    output_path = tmp_path / "descriptors.csv"
    input_path.write_text(
        "molecule_id,canonical_smiles,valid_smiles,error_message\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,True,\n"
        "invalid,,False,Invalid input structure.\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.descriptors",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 2
    assert rows[0]["molecule_id"] == "aspirin"
    assert rows[1]["descriptor_error"] == "Invalid input structure."
