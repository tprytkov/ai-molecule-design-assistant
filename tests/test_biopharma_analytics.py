from __future__ import annotations

import csv
from pathlib import Path

from src.analytics import generate_biopharma_outputs
from src.scoring import scoring_csv


def write_minimal_upstream_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "standardized.csv").write_text(
        "molecule_id,smiles,canonical_smiles,valid_smiles\n"
        "mol_a,CCO,CCO,True\n",
        encoding="utf-8",
    )
    (output_dir / "descriptors.csv").write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed,lipinski_violations,"
        "lipinski_pass,druglikeness_category,druglikeness_score,druglikeness_flags,"
        "mw_status,logp_status,tpsa_status,qed_status,lipinski_status,descriptor_error\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4,0,True,"
        "borderline,0.8,,borderline,favorable,favorable,borderline,favorable,\n",
        encoding="utf-8",
    )
    (output_dir / "similarity.csv").write_text(
        "molecule_id,best_reference_id,best_reference_name,tanimoto_similarity,"
        "similarity_category,reference_smiles,query_smiles\n"
        "mol_a,ref_a,reference,0.500,moderate_similarity,CCO,CCO\n",
        encoding="utf-8",
    )
    (output_dir / "public_lookup.csv").write_text(
        "molecule_id,pubchem_status,chembl_status,lookup_status\n"
        "mol_a,no_match,no_match,not_found\n",
        encoding="utf-8",
    )
    (output_dir / "biomedical_evidence.csv").write_text(
        "molecule_id,biomedical_model_name,biomedical_model_status,biomedical_evidence_status\n"
        "mol_a,fake,preferred_model_used,available\n",
        encoding="utf-8",
    )
    (output_dir / "patent_evidence_embeddings.csv").write_text(
        "molecule_id,patent_model_name,patent_model_status,patent_evidence_status\n"
        "mol_a,fake,preferred_model_used,available\n",
        encoding="utf-8",
    )
    (output_dir / "admet_summary.csv").write_text(
        "molecule_id,smiles,bbb_prediction_label,cns_property_flag,"
        "toxicity_risk_flag,admet_readiness_category,model_status,evidence_note\n"
        "mol_a,CCO,moderate,moderate,favorable,moderate,"
        "fallback_descriptor_rule,Heuristic baseline.\n",
        encoding="utf-8",
    )
    scoring_csv(
        output_dir / "descriptors.csv",
        output_dir / "similarity.csv",
        output_dir / "prioritization_results.csv",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_biopharma_outputs_are_generated_from_existing_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    write_minimal_upstream_outputs(output_dir)

    counts = generate_biopharma_outputs(output_dir=output_dir)

    assert counts["biopharma_positioning"] == 1
    assert counts["evidence_readiness"] == 1
    assert counts["mock_rwe_cohort_summary"] >= 1
    assert counts["trial_endpoint_map"] >= 1
    for filename in (
        "biopharma_positioning.csv",
        "evidence_readiness.csv",
        "mock_rwe_cohort_summary.csv",
        "trial_endpoint_map.csv",
        "biopharma_summary_report.md",
    ):
        assert (output_dir / filename).exists()

    positioning = read_rows(output_dir / "biopharma_positioning.csv")
    assert positioning[0]["indication"] == "Alzheimer's disease"
    assert "alpha7" in positioning[0]["target_context"]
    assert "biomedical_evidence_support" in positioning[0]["positioning_category"]


def test_biopharma_outputs_tolerate_missing_optional_upstream_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "standardized.csv").write_text(
        "molecule_id,smiles,canonical_smiles,valid_smiles\n"
        "mol_gap,CCN,CCN,True\n",
        encoding="utf-8",
    )

    counts = generate_biopharma_outputs(output_dir=output_dir)

    assert counts["biopharma_positioning"] == 1
    readiness = read_rows(output_dir / "evidence_readiness.csv")
    assert readiness[0]["translational_readiness_category"] == "translational_followup_needed"
    assert "evidence_gap" in readiness[0].values()


def test_mock_rwe_output_is_labeled_synthetic_not_real_patient_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    write_minimal_upstream_outputs(output_dir)

    generate_biopharma_outputs(output_dir=output_dir)

    mock_rows = read_rows(output_dir / "mock_rwe_cohort_summary.csv")
    notes = " ".join(row["note"] for row in mock_rows)
    assert "Mock/synthetic" in notes
    assert "not real patient data" in notes


def test_biopharma_outputs_do_not_change_scoring_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    write_minimal_upstream_outputs(output_dir)
    baseline = output_dir / "prioritized_baseline.csv"
    after_biopharma = output_dir / "prioritized_after_biopharma.csv"

    scoring_csv(output_dir / "descriptors.csv", output_dir / "similarity.csv", baseline)
    generate_biopharma_outputs(output_dir=output_dir)
    scoring_csv(output_dir / "descriptors.csv", output_dir / "similarity.csv", after_biopharma)

    assert baseline.read_text(encoding="utf-8") == after_biopharma.read_text(
        encoding="utf-8"
    )
