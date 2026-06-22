import csv
import subprocess
import sys
from pathlib import Path

import pytest

from src.compound_search import (
    OUTPUT_COLUMNS,
    compound_report,
    prepare_search_references,
    top_hits_csv,
)


def write_inputs(descriptors: Path, references: Path) -> None:
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles\n"
        "benzene,c1ccccc1,True\n"
        "invalid,,False\n",
        encoding="utf-8",
    )
    references.write_text(
        "reference_id,reference_name,smiles,reference_source,"
        "reference_source_id,evidence_note\n"
        "ref_ethanol,ethanol,CCO,public_demo,PUB-001,Public-safe note.\n"
        "ref_benzene,benzene,c1ccccc1,public_demo,PUB-002,Exact demo match.\n"
        "ref_invalid,invalid,C1CC,public_demo,PUB-003,Invalid demo row.\n",
        encoding="utf-8",
    )


def test_top_k_output_created_and_sorted(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    references = tmp_path / "references.csv"
    output = tmp_path / "hits.csv"
    write_inputs(descriptors, references)

    count = top_hits_csv(descriptors, references, output, top_k=2)

    assert count == 2
    assert output.exists()
    with output.open("r", encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert [row["hit_rank"] for row in rows] == ["1", "2"]
    scores = [float(row["tanimoto_similarity"]) for row in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["reference_name"] == "benzene"
    assert all(row["molecule_id"] != "invalid" for row in rows)


def test_invalid_reference_is_skipped() -> None:
    references = prepare_search_references(
        [
            {
                "reference_id": "bad",
                "reference_name": "bad",
                "smiles": "C1CC",
            },
            {
                "reference_id": "good",
                "reference_name": "benzene",
                "smiles": "c1ccccc1",
            },
        ]
    )

    assert len(references) == 1
    assert references[0].reference_id == "good"


def test_molecule_specific_markdown_report_created(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    references = tmp_path / "references.csv"
    report = tmp_path / "report.md"
    write_inputs(descriptors, references)

    count = compound_report(
        "benzene", descriptors, references, report, top_k=2
    )

    assert count == 2
    content = report.read_text(encoding="utf-8")
    assert "# Compound Intelligence Report: benzene" in content
    assert "Top Similar Reference Compounds" in content
    assert "not a legal conclusion" in content


def test_missing_molecule_id_gives_clear_error(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    references = tmp_path / "references.csv"
    write_inputs(descriptors, references)

    with pytest.raises(ValueError, match="Molecule ID 'missing' was not found"):
        compound_report(
            "missing", descriptors, references, tmp_path / "report.md", 2
        )


def test_cli_writes_batch_output(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    references = tmp_path / "references.csv"
    output = tmp_path / "hits.csv"
    write_inputs(descriptors, references)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.compound_search",
            "--descriptors",
            str(descriptors),
            "--references",
            str(references),
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
