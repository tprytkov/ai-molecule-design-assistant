import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.similarity import (
    OUTPUT_COLUMNS,
    calculate_best_similarity,
    categorize_similarity,
    prepare_references,
    similarity_csv,
)


def public_references():
    return prepare_references(
        [
            {
                "reference_id": "ref_aspirin",
                "name": "aspirin",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            },
            {
                "reference_id": "ref_benzene",
                "name": "benzene",
                "smiles": "c1ccccc1",
            },
        ]
    )


def test_valid_molecule_similarity_calculation() -> None:
    result = calculate_best_similarity(
        molecule_id="aspirin",
        canonical_smiles="CC(=O)Oc1ccccc1C(=O)O",
        valid_smiles=True,
        references=public_references(),
    )

    assert result.valid_smiles is True
    assert float(result.tanimoto_similarity) == pytest.approx(1.0)
    assert result.similarity_category == "very_close_analog"
    assert result.similarity_error == ""


def test_invalid_generated_molecule_is_retained() -> None:
    result = calculate_best_similarity(
        molecule_id="invalid",
        canonical_smiles="",
        valid_smiles=False,
        references=public_references(),
        upstream_error="Invalid generated structure.",
    )

    assert result.molecule_id == "invalid"
    assert result.valid_smiles is False
    assert result.tanimoto_similarity == ""
    assert result.similarity_category == "not_available"
    assert result.similarity_error == "Invalid generated structure."


def test_invalid_reference_molecule_is_skipped() -> None:
    references = prepare_references(
        [
            {
                "reference_id": "invalid",
                "name": "invalid",
                "smiles": "C1CC",
            },
            {
                "reference_id": "benzene",
                "name": "benzene",
                "smiles": "c1ccccc1",
            },
        ]
    )

    assert len(references) == 1
    assert references[0].reference_id == "benzene"


def test_best_reference_molecule_is_selected_correctly() -> None:
    result = calculate_best_similarity(
        molecule_id="benzene",
        canonical_smiles="c1ccccc1",
        valid_smiles=True,
        references=public_references(),
    )

    assert result.best_reference_id == "ref_benzene"
    assert result.best_reference_name == "benzene"
    assert result.best_reference_smiles == "c1ccccc1"
    assert result.tanimoto_similarity == "1.000"


@pytest.mark.parametrize(
    ("score", "expected_category"),
    [
        (0.85, "very_close_analog"),
        (0.849, "related_chemotype"),
        (0.70, "related_chemotype"),
        (0.699, "moderate_similarity"),
        (0.50, "moderate_similarity"),
        (0.499, "structurally_distinct"),
        (None, "not_available"),
    ],
)
def test_similarity_category_thresholds(
    score: float | None, expected_category: str
) -> None:
    assert categorize_similarity(score) == expected_category


def write_test_inputs(
    generated_path: Path, reference_path: Path
) -> None:
    generated_path.write_text(
        "molecule_id,canonical_smiles,valid_smiles,descriptor_error\n"
        "aspirin,CC(=O)Oc1ccccc1C(=O)O,True,\n"
        "invalid,,False,Invalid generated structure.\n",
        encoding="utf-8",
    )
    reference_path.write_text(
        "reference_id,name,smiles\n"
        "ref_aspirin,aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
        "ref_invalid,invalid,C1CC\n",
        encoding="utf-8",
    )


def test_output_columns_exist(tmp_path: Path) -> None:
    generated_path = tmp_path / "descriptors.csv"
    reference_path = tmp_path / "references.csv"
    output_path = tmp_path / "similarity.csv"
    write_test_inputs(generated_path, reference_path)

    similarity_csv(generated_path, reference_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS


def test_cli_writes_expected_output_file(tmp_path: Path) -> None:
    generated_path = tmp_path / "descriptors.csv"
    reference_path = tmp_path / "references.csv"
    output_path = tmp_path / "similarity.csv"
    write_test_inputs(generated_path, reference_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.similarity",
            "--input",
            str(generated_path),
            "--references",
            str(reference_path),
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
    assert rows[0]["best_reference_id"] == "ref_aspirin"
    assert rows[1]["similarity_category"] == "not_available"
