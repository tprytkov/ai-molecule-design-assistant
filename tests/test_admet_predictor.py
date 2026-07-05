import csv
from pathlib import Path

from src.admet.admet_predictor import (
    ENDPOINTS,
    admet_csv,
    build_admet_outputs,
    descriptor_records,
)
from src.admet.admet_schema import (
    ADMET_PREDICTION_COLUMNS,
    ADMET_SUMMARY_COLUMNS,
    MODEL_STATUS_FALLBACK,
)
from src.scoring import scoring_csv


def test_admet_schema_columns_are_fixed() -> None:
    assert ADMET_PREDICTION_COLUMNS == (
        "molecule_id",
        "smiles",
        "admet_endpoint",
        "prediction_value",
        "prediction_probability",
        "prediction_label",
        "model_id",
        "model_backend",
        "model_status",
        "model_cache_status",
        "training_dataset",
        "evidence_note",
    )
    assert ADMET_SUMMARY_COLUMNS == (
        "molecule_id",
        "smiles",
        "bbb_prediction_label",
        "cns_property_flag",
        "toxicity_risk_flag",
        "admet_readiness_category",
        "model_status",
        "evidence_note",
    )


def test_descriptor_rule_fallback_outputs_endpoint_rows(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4\n",
        encoding="utf-8",
    )

    records = descriptor_records(descriptors_path=descriptors)
    predictions, summary = build_admet_outputs(records)

    assert len(predictions) == len(ENDPOINTS)
    assert len(summary) == 1
    assert {row["prediction_label"] for row in predictions} <= {
        "favorable",
        "moderate",
        "caution",
        "unavailable",
    }
    assert {row["model_status"] for row in predictions} == {MODEL_STATUS_FALLBACK}
    assert "heuristic" in predictions[0]["evidence_note"].lower()


def test_invalid_smiles_is_retained_as_unavailable(tmp_path: Path) -> None:
    standardized = tmp_path / "standardized.csv"
    standardized.write_text(
        "molecule_id,canonical_smiles,valid_smiles,parse_error\n"
        "bad,C1CC,False,RDKit parse failed\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "admet_predictions.csv"
    summary = tmp_path / "admet_summary.csv"

    counts = admet_csv(
        standardized_path=standardized,
        predictions_path=predictions,
        summary_path=summary,
    )

    assert counts == {"predictions": len(ENDPOINTS), "summary": 1}
    with predictions.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["prediction_label"] for row in rows} == {"unavailable"}
    with summary.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["admet_readiness_category"] == "unavailable"


def test_admet_csv_generation_writes_required_outputs(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4\n"
        "mol_b,c1ccccc1,True,78.114,1.687,0.00,0,0,0,1,0.44\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "admet_predictions.csv"
    summary = tmp_path / "admet_summary.csv"

    counts = admet_csv(
        descriptors_path=descriptors,
        predictions_path=predictions,
        summary_path=summary,
    )

    assert counts == {"predictions": 2 * len(ENDPOINTS), "summary": 2}
    with predictions.open("r", encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == ADMET_PREDICTION_COLUMNS
    with summary.open("r", encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == ADMET_SUMMARY_COLUMNS


def test_admet_outputs_do_not_change_scoring_columns(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    similarity = tmp_path / "similarity.csv"
    baseline = tmp_path / "prioritized_baseline.csv"
    with_admet = tmp_path / "prioritized_with_admet.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed,lipinski_violations,"
        "lipinski_pass,druglikeness_category,druglikeness_score,druglikeness_flags,"
        "mw_status,logp_status,tpsa_status,qed_status,lipinski_status,descriptor_error\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4,0,True,"
        "borderline,0.8,,borderline,favorable,favorable,borderline,favorable,\n",
        encoding="utf-8",
    )
    similarity.write_text(
        "molecule_id,best_reference_id,best_reference_name,tanimoto_similarity,"
        "similarity_category,reference_smiles,query_smiles\n"
        "mol_a,ref_a,reference,0.500,moderate_similarity,CCO,CCO\n",
        encoding="utf-8",
    )

    scoring_csv(descriptors, similarity, baseline)
    admet_csv(
        descriptors_path=descriptors,
        predictions_path=tmp_path / "admet_predictions.csv",
        summary_path=tmp_path / "admet_summary.csv",
    )
    scoring_csv(descriptors, similarity, with_admet)

    assert baseline.read_text(encoding="utf-8") == with_admet.read_text(
        encoding="utf-8"
    )
