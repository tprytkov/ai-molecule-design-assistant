import csv
from pathlib import Path

from rdkit import Chem


EXAMPLE_DIR = Path("data/examples")
CANDIDATES = EXAMPLE_DIR / "druglike_candidate_demo.csv"
REFERENCES = EXAMPLE_DIR / "druglike_reference_panel.csv"
TEXT_EVIDENCE = EXAMPLE_DIR / "text_evidence_demo.csv"
DATASET_README = EXAMPLE_DIR / "README.md"
ALLOWED_REFERENCE_ROLES = {
    "desired_reference",
    "background_reference",
    "avoid_reference",
    "control_reference",
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader.fieldnames or []), list(reader)


def test_all_demo_files_exist() -> None:
    assert CANDIDATES.is_file()
    assert REFERENCES.is_file()
    assert TEXT_EVIDENCE.is_file()
    assert DATASET_README.is_file()


def test_candidate_dataset_size_columns_and_smiles() -> None:
    columns, rows = read_rows(CANDIDATES)

    assert {"molecule_id", "smiles", "notes"} <= set(columns)
    assert 30 <= len(rows) <= 40
    assert [row["molecule_id"] for row in rows] == [
        f"demo_{index:03d}" for index in range(1, len(rows) + 1)
    ]
    parsed = [Chem.MolFromSmiles(row["smiles"]) for row in rows]
    assert sum(molecule is None for molecule in parsed) >= 1
    assert sum(molecule is not None for molecule in parsed) >= 29


def test_reference_panel_size_columns_roles_and_smiles() -> None:
    columns, rows = read_rows(REFERENCES)

    assert {
        "reference_name",
        "smiles",
        "reference_role",
        "target_family",
        "notes",
    } <= set(columns)
    assert len(rows) >= 8
    assert all(Chem.MolFromSmiles(row["smiles"]) is not None for row in rows)
    assert {row["reference_role"] for row in rows} <= ALLOWED_REFERENCE_ROLES


def test_text_evidence_columns_and_size() -> None:
    columns, rows = read_rows(TEXT_EVIDENCE)

    assert {
        "evidence_id",
        "text",
        "source",
        "target_family",
        "notes",
    } <= set(columns)
    assert 8 <= len(rows) <= 12


def test_example_readme_uses_public_safe_neutral_language() -> None:
    text = DATASET_README.read_text(encoding="utf-8").lower()
    forbidden = [
        "nico" + "tinic",
        "alpha" + "7",
        "alz" + "heimer",
        "patent" + "ability",
        "f" + "to",
        "infringe" + "ment",
    ]

    for term in forbidden:
        assert term not in text
