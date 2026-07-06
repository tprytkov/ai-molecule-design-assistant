import csv
from pathlib import Path

from src.admet.admet_predictor import (
    ENDPOINTS,
    ENDPOINT_BBB,
    MODEL_STATUS_TUNED_BBB,
    admet_csv,
    build_admet_outputs,
    descriptor_records,
)
from src.admet.admet_model_registry import ADMET_MODEL_REGISTRY
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


def test_admet_registry_includes_tuned_chemberta_bbb_candidate() -> None:
    registry = {entry.model_id: entry for entry in ADMET_MODEL_REGISTRY}

    assert "Yousuf7/ChemBERT-BBB-Permeability" in registry
    assert registry["Yousuf7/ChemBERT-BBB-Permeability"].endpoint == "bbb_permeability"
    assert registry["Yousuf7/ChemBERT-BBB-Permeability"].enabled is True
    assert "experimental" in registry["Yousuf7/ChemBERT-BBB-Permeability"].notes.lower()
    assert "DeepChem/ChemBERTa-77M-MLM" in registry
    assert "not treated as ADMET prediction" in registry["DeepChem/ChemBERTa-77M-MLM"].notes


class FakeBBBClassifier:
    model_id = "Yousuf7/ChemBERT-BBB-Permeability"

    def predict_label(self, smiles: str) -> tuple[str, str]:
        return "favorable", "0.870"


def test_tuned_bbb_classifier_updates_only_bbb_model_metadata(tmp_path: Path) -> None:
    descriptors = tmp_path / "descriptors.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4\n",
        encoding="utf-8",
    )
    records = descriptor_records(descriptors_path=descriptors)

    predictions, summary = build_admet_outputs(
        records,
        bbb_classifier=FakeBBBClassifier(),
    )

    bbb = next(row for row in predictions if row["admet_endpoint"] == ENDPOINT_BBB)
    non_bbb = [row for row in predictions if row["admet_endpoint"] != ENDPOINT_BBB]
    assert bbb["model_id"] == "Yousuf7/ChemBERT-BBB-Permeability"
    assert bbb["model_status"] == MODEL_STATUS_TUNED_BBB
    assert bbb["model_cache_status"] == "cached"
    assert bbb["prediction_probability"] == "0.870"
    assert {row["model_status"] for row in non_bbb} == {MODEL_STATUS_FALLBACK}
    assert summary[0]["model_status"] == "mixed_tuned_bbb_and_descriptor_rule"


def test_admet_falls_back_when_tuned_chemberta_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.admet.admet_predictor.model_is_cached", lambda model_id: False)
    descriptors = tmp_path / "descriptors.csv"
    predictions = tmp_path / "admet_predictions.csv"
    summary = tmp_path / "admet_summary.csv"
    descriptors.write_text(
        "molecule_id,canonical_smiles,valid_smiles,molecular_weight,logp,tpsa,"
        "hbd,hba,rotatable_bonds,aromatic_rings,qed\n"
        "mol_a,CCO,True,46.069,-0.001,20.23,1,1,0,0,0.4\n",
        encoding="utf-8",
    )

    admet_csv(
        descriptors_path=descriptors,
        predictions_path=predictions,
        summary_path=summary,
    )

    rows = list(csv.DictReader(predictions.open("r", encoding="utf-8", newline="")))
    assert {row["model_status"] for row in rows} == {MODEL_STATUS_FALLBACK}
    assert {row["model_cache_status"] for row in rows} == {"not_required"}


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
