import csv
from pathlib import Path

from src.admet.admet_model_evaluation import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    evaluate_classification_model,
    evaluate_regression_model,
    write_evaluation_outputs,
)


class MockClassificationPredictor:
    def predict_proba(self, smiles: str) -> float:
        return 0.9 if smiles == "CCO" else 0.1


class MockRegressionPredictor:
    def predict_value(self, smiles: str) -> float:
        return 1.0 if smiles == "CCO" else 2.0


def test_classification_evaluation_writes_summary_and_rdkit_baseline(tmp_path: Path) -> None:
    rows = [
        {"molecule_id": "m1", "smiles": "CCO", "label": "1", "split": "test"},
        {"molecule_id": "m2", "smiles": "c1ccccc1", "label": "0", "split": "test"},
    ]

    result = evaluate_classification_model(
        endpoint_name="bbb_permeability",
        model_id="mock/model",
        benchmark_dataset="local_mock",
        rows=rows,
        predictor=MockClassificationPredictor(),
        baseline_endpoint="bbb_permeability_cns_likeness",
        validation_threshold=0.5,
    )
    summary = result.summary_rows[0]

    assert summary["validation_status"] == "benchmark_passed"
    assert summary["metric_primary"] == "balanced_accuracy"
    assert summary["baseline_method"] == "RDKit descriptor/rule baseline"
    assert summary["baseline_metric_primary_value"] != ""
    assert set(result.detail_rows[0]) == set(DETAIL_COLUMNS)

    summary_path = tmp_path / "admet_model_evaluation_summary.csv"
    details_path = tmp_path / "admet_model_evaluation_details.csv"
    counts = write_evaluation_outputs(
        summary_path=summary_path,
        details_path=details_path,
        result=result,
    )

    assert counts == {"summary": 1, "details": 2}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == SUMMARY_COLUMNS


def test_regression_evaluation_reports_regression_metrics() -> None:
    rows = [
        {"molecule_id": "m1", "smiles": "CCO", "label": "1.0", "split": "test"},
        {"molecule_id": "m2", "smiles": "c1ccccc1", "label": "2.5", "split": "test"},
    ]

    result = evaluate_regression_model(
        endpoint_name="solubility",
        model_id="mock/regressor",
        benchmark_dataset="local_mock",
        rows=rows,
        predictor=MockRegressionPredictor(),
        validation_threshold=1.0,
    )
    summary = result.summary_rows[0]

    assert summary["task_type"] == "regression"
    assert summary["metric_primary"] == "rmse"
    assert summary["rmse"] != ""
    assert summary["mae"] != ""
    assert summary["spearman"] != ""


def test_classification_evaluation_marks_invalid_smiles_failed() -> None:
    rows = [
        {"molecule_id": "m1", "smiles": "not_smiles", "label": "1", "split": "test"},
    ]

    result = evaluate_classification_model(
        endpoint_name="bbb_permeability",
        model_id="mock/model",
        benchmark_dataset="local_mock",
        rows=rows,
        predictor=MockClassificationPredictor(),
        baseline_endpoint="bbb_permeability_cns_likeness",
    )

    assert result.summary_rows[0]["n_failed_smiles"] == "1"
    assert result.detail_rows[0]["inference_status"] == "failed_smiles"
