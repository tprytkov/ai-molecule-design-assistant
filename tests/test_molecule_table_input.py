from pathlib import Path

from src.io.molecule_table_input import normalize_molecule_csv, normalize_molecule_rows
from src.standardize import standardize_csv


def test_simple_molecule_smiles_table_normalizes_to_existing_schema(tmp_path: Path) -> None:
    source = tmp_path / "molecules.csv"
    normalized = tmp_path / "normalized.csv"
    standardized = tmp_path / "standardized.csv"
    source.write_text(
        "molecule,smiles\ncompound_1,CCO\ncompound_2,c1ccccc1\n",
        encoding="utf-8",
    )

    assert normalize_molecule_csv(source, normalized) == 2
    assert standardize_csv(source, standardized) == 2

    content = standardized.read_text(encoding="utf-8")
    assert "compound_1" in content
    assert "c1ccccc1" in content


def test_molecule_table_accepts_name_and_smiles_synonyms() -> None:
    rows = normalize_molecule_rows(
        [
            {"ligand": " Lig A ", "generated_smiles": " CCO "},
            {"compound": "Lig B", "canonical_smiles": "c1ccccc1"},
        ]
    )

    assert rows[0]["molecule_id"] == "Lig A"
    assert rows[0]["smiles"] == "CCO"
    assert rows[1]["molecule_id"] == "Lig B"
    assert rows[1]["smiles"] == "c1ccccc1"


def test_invalid_smiles_are_preserved_with_error_status(tmp_path: Path) -> None:
    source = tmp_path / "molecules.csv"
    standardized = tmp_path / "standardized.csv"
    source.write_text(
        "molecule,smiles\nvalid,CCO\ninvalid,not_a_smiles\n",
        encoding="utf-8",
    )

    standardize_csv(source, standardized)

    content = standardized.read_text(encoding="utf-8")
    assert "valid" in content
    assert "invalid" in content
    assert "False" in content


def test_duplicate_molecule_names_are_marked_before_merge() -> None:
    rows = normalize_molecule_rows(
        [
            {"molecule": "Compound A", "smiles": "CCO"},
            {"molecule": " compound a ", "smiles": "CCN"},
        ]
    )

    assert {row["input_status"] for row in rows} == {"duplicate_molecule"}
