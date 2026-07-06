from pathlib import Path

from src.structural.simple_docking_input import (
    build_docking_merge,
    write_simple_docking_outputs_from_csv,
)


MOLECULES = [
    {"molecule_id": "compound_1", "canonical_smiles": "CCO"},
    {"molecule_id": "compound_2", "canonical_smiles": "CCN"},
    {"molecule_id": "compound_3", "canonical_smiles": "CCC"},
]


def test_simple_docking_table_merges_by_normalized_molecule_name() -> None:
    rows, report = build_docking_merge(
        molecule_rows=MOLECULES,
        docking_rows=[
            {"molecule": " Compound_1 ", "affinity": "-8.4"},
            {"molecule": "compound_2", "affinity": "-7.1"},
        ],
        docking_program="AutoDock Vina",
        docking_units="kcal/mol",
        selected_target_id="target_a",
    )

    assert rows[0]["docking_status"] == "available"
    assert rows[0]["docking_rank"] == "1"
    assert rows[0]["docking_program"] == "AutoDock Vina"
    assert rows[0]["docking_units"] == "kcal/mol"
    assert rows[0]["target_id"] == "target_a"
    assert rows[2]["docking_status"] == "no_docking_file_provided"
    assert any(row["report_type"] == "molecules_without_docking" for row in report)


def test_docking_table_accepts_affinity_and_name_synonyms() -> None:
    rows, _ = build_docking_merge(
        molecule_rows=MOLECULES[:1],
        docking_rows=[{"ligand": "compound_1", "binding_energy": "-6.5"}],
    )

    assert rows[0]["docking_score"] == "-6.5"
    assert rows[0]["docking_status"] == "available"


def test_docking_merge_reports_unmatched_duplicate_and_invalid_rows() -> None:
    rows, report = build_docking_merge(
        molecule_rows=[
            {"molecule_id": "Compound A", "canonical_smiles": "CCO"},
            {"molecule_id": "compound a", "canonical_smiles": "CCN"},
            {"molecule_id": "Compound B", "canonical_smiles": "CCC"},
        ],
        docking_rows=[
            {"molecule": "compound a", "affinity": "-8.0"},
            {"molecule": "Compound B", "affinity": "not_numeric"},
            {"molecule": "unmatched", "affinity": "-7.0"},
        ],
    )

    assert rows[0]["docking_status"] == "ambiguous_duplicate"
    assert rows[2]["docking_status"] == "invalid_score"
    report_types = {row["report_type"] for row in report}
    assert "duplicate_molecule_names" in report_types
    assert "invalid_affinity_rows" in report_types
    assert "unmatched_docking_rows" in report_types


def test_higher_is_better_rank_direction() -> None:
    rows, _ = build_docking_merge(
        molecule_rows=MOLECULES[:2],
        docking_rows=[
            {"molecule": "compound_1", "score": "2.0"},
            {"molecule": "compound_2", "score": "5.0"},
        ],
        docking_score_direction="higher is better",
    )

    ranks = {row["molecule_id"]: row["docking_rank"] for row in rows}
    assert ranks["compound_2"] == "1"
    assert ranks["compound_1"] == "2"


def test_simple_docking_outputs_are_written_with_user_or_demo_source(tmp_path: Path) -> None:
    molecules = tmp_path / "molecules.csv"
    docking = tmp_path / "docking.csv"
    normalized = tmp_path / "docking_results_normalized.csv"
    report = tmp_path / "docking_merge_report.csv"
    molecules.write_text("molecule,smiles\ncompound_1,CCO\n", encoding="utf-8")
    docking.write_text("molecule,affinity\ncompound_1,-8.4\n", encoding="utf-8")

    counts = write_simple_docking_outputs_from_csv(
        molecule_path=molecules,
        docking_path=docking,
        docking_output_path=normalized,
        merge_report_path=report,
        docking_source="illustrative_demo",
        selected_target_id="adora2a_xanthine_demo",
    )

    assert counts["docking_results_normalized"] == 1
    normalized_text = normalized.read_text(encoding="utf-8")
    assert "illustrative_demo" in normalized_text
    assert "not real docking validation" in normalized_text
    assert "matched_rows" in report.read_text(encoding="utf-8")
