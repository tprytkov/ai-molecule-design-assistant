"""Evaluate endpoint-specific ADMET models against public benchmark rows."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from rdkit import Chem

from src.admet.admet_baselines import rdkit_rule_predictions

SUMMARY_COLUMNS = (
    "endpoint_name",
    "model_id",
    "benchmark_dataset",
    "split",
    "task_type",
    "metric_primary",
    "metric_primary_value",
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "rmse",
    "mae",
    "r2",
    "spearman",
    "n_train",
    "n_valid",
    "n_test",
    "n_failed_smiles",
    "baseline_method",
    "baseline_metric_primary_value",
    "validation_status",
    "limitation_note",
)
DETAIL_COLUMNS = (
    "endpoint_name",
    "model_id",
    "molecule_id",
    "smiles",
    "true_label_or_value",
    "predicted_label_or_value",
    "prediction_probability",
    "prediction_error",
    "split",
    "inference_status",
)
DEFAULT_LIMITATION_NOTE = (
    "Computational benchmark summary only; not experimental ADMET, toxicity, "
    "safety, or clinical evidence."
)


class ClassificationPredictor(Protocol):
    """Minimal binary classification predictor interface."""

    def predict_proba(self, smiles: str) -> float:
        """Return probability for the positive class."""


class RegressionPredictor(Protocol):
    """Minimal regression predictor interface."""

    def predict_value(self, smiles: str) -> float:
        """Return a numeric prediction."""


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation summary and detail rows."""

    summary_rows: list[dict[str, str]]
    detail_rows: list[dict[str, str]]


def _float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _is_valid_smiles(smiles: str) -> bool:
    return bool(str(smiles).strip()) and Chem.MolFromSmiles(str(smiles).strip()) is not None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][1] == order[index][1]:
            end += 1
        rank = (index + end + 1) / 2
        for original, _ in order[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_den = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_den = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    return numerator / (x_den * y_den) if x_den and y_den else float("nan")


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - last_recall)
        last_recall = recall
    return area


def classification_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | int]:
    """Return binary classification metrics."""
    predicted = [1 if probability >= 0.5 else 0 for probability in probabilities]
    tp = sum(1 for y, p in zip(labels, predicted) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predicted) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predicted) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predicted) if y == 1 and p == 0)
    total = len(labels)
    accuracy = (tp + tn) / total if total else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "auroc": _auc(labels, probabilities),
        "auprc": _auprc(labels, probabilities),
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def regression_metrics(labels: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    """Return regression metrics."""
    n = len(labels)
    if not n:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "spearman": float("nan")}
    errors = [pred - true for true, pred in zip(labels, predictions)]
    mse = sum(error**2 for error in errors) / n
    mae = sum(abs(error) for error in errors) / n
    mean_label = sum(labels) / n
    ss_tot = sum((true - mean_label) ** 2 for true in labels)
    ss_res = sum(error**2 for error in errors)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    spearman = _pearson(_rank(labels), _rank(predictions))
    return {"rmse": math.sqrt(mse), "mae": mae, "r2": r2, "spearman": spearman}


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def evaluate_classification_model(
    *,
    endpoint_name: str,
    model_id: str,
    benchmark_dataset: str,
    rows: Iterable[Mapping[str, str]],
    predictor: ClassificationPredictor,
    split: str = "test",
    baseline_endpoint: str,
    validation_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate a binary classification endpoint model."""
    details = []
    labels: list[int] = []
    probabilities: list[float] = []
    failed = 0
    materialized = [dict(row) for row in rows if str(row.get("split", split)).lower() == split.lower()]
    for row in materialized:
        smiles = str(row.get("smiles", "")).strip()
        label = int(float(str(row.get("label", "0")).strip()))
        status = "available"
        probability = None
        predicted = ""
        error = ""
        if not _is_valid_smiles(smiles):
            failed += 1
            status = "failed_smiles"
            error = "invalid_smiles"
        else:
            try:
                probability = float(predictor.predict_proba(smiles))
                predicted = str(1 if probability >= 0.5 else 0)
                labels.append(label)
                probabilities.append(probability)
            except Exception as exc:
                failed += 1
                status = "model_error"
                error = f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "endpoint_name": endpoint_name,
                "model_id": model_id,
                "molecule_id": str(row.get("molecule_id", "")),
                "smiles": smiles,
                "true_label_or_value": str(label),
                "predicted_label_or_value": predicted,
                "prediction_probability": _fmt(probability) if probability is not None else "",
                "prediction_error": error,
                "split": split,
                "inference_status": status,
            }
        )
    metrics = classification_metrics(labels, probabilities)
    baseline_rows = rdkit_rule_predictions(materialized, baseline_endpoint)
    baseline_pairs = [
        (int(float(input_row.get("label", 0))), float(baseline_row["prediction_binary"]))
        for input_row, baseline_row in zip(materialized, baseline_rows)
        if baseline_row["inference_status"] == "available"
    ]
    baseline_labels = [label for label, _ in baseline_pairs]
    baseline_scores = [score for _, score in baseline_pairs]
    baseline_metric = classification_metrics(baseline_labels, baseline_scores).get("balanced_accuracy", float("nan"))
    primary = float(metrics["balanced_accuracy"])
    validation_status = "benchmark_passed" if primary >= validation_threshold else "benchmark_failed"
    summary = _summary_row(
        endpoint_name=endpoint_name,
        model_id=model_id,
        benchmark_dataset=benchmark_dataset,
        split=split,
        task_type="binary_classification",
        metric_primary="balanced_accuracy",
        metric_primary_value=primary,
        metrics=metrics,
        n_test=len(materialized),
        n_failed_smiles=failed,
        baseline_metric_primary_value=float(baseline_metric),
        validation_status=validation_status,
    )
    return EvaluationResult([summary], details)


def evaluate_regression_model(
    *,
    endpoint_name: str,
    model_id: str,
    benchmark_dataset: str,
    rows: Iterable[Mapping[str, str]],
    predictor: RegressionPredictor,
    split: str = "test",
    validation_threshold: float = float("inf"),
) -> EvaluationResult:
    """Evaluate a regression endpoint model."""
    details = []
    labels: list[float] = []
    predictions: list[float] = []
    failed = 0
    materialized = [dict(row) for row in rows if str(row.get("split", split)).lower() == split.lower()]
    for row in materialized:
        smiles = str(row.get("smiles", "")).strip()
        label = _float(row.get("label"))
        status = "available"
        prediction = None
        error = ""
        if label is None or not _is_valid_smiles(smiles):
            failed += 1
            status = "failed_smiles"
            error = "invalid_label_or_smiles"
        else:
            try:
                prediction = float(predictor.predict_value(smiles))
                labels.append(label)
                predictions.append(prediction)
            except Exception as exc:
                failed += 1
                status = "model_error"
                error = f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "endpoint_name": endpoint_name,
                "model_id": model_id,
                "molecule_id": str(row.get("molecule_id", "")),
                "smiles": smiles,
                "true_label_or_value": _fmt(label) if label is not None else "",
                "predicted_label_or_value": _fmt(prediction) if prediction is not None else "",
                "prediction_probability": "",
                "prediction_error": error,
                "split": split,
                "inference_status": status,
            }
        )
    metrics = regression_metrics(labels, predictions)
    primary = float(metrics["rmse"])
    validation_status = "benchmark_passed" if primary <= validation_threshold else "benchmark_failed"
    summary = _summary_row(
        endpoint_name=endpoint_name,
        model_id=model_id,
        benchmark_dataset=benchmark_dataset,
        split=split,
        task_type="regression",
        metric_primary="rmse",
        metric_primary_value=primary,
        metrics=metrics,
        n_test=len(materialized),
        n_failed_smiles=failed,
        baseline_metric_primary_value=float("nan"),
        validation_status=validation_status,
    )
    return EvaluationResult([summary], details)


def _summary_row(
    *,
    endpoint_name: str,
    model_id: str,
    benchmark_dataset: str,
    split: str,
    task_type: str,
    metric_primary: str,
    metric_primary_value: float,
    metrics: Mapping[str, object],
    n_test: int,
    n_failed_smiles: int,
    baseline_metric_primary_value: float,
    validation_status: str,
) -> dict[str, str]:
    return {
        "endpoint_name": endpoint_name,
        "model_id": model_id,
        "benchmark_dataset": benchmark_dataset,
        "split": split,
        "task_type": task_type,
        "metric_primary": metric_primary,
        "metric_primary_value": _fmt(metric_primary_value),
        "auroc": _fmt(metrics.get("auroc", "")),
        "auprc": _fmt(metrics.get("auprc", "")),
        "accuracy": _fmt(metrics.get("accuracy", "")),
        "balanced_accuracy": _fmt(metrics.get("balanced_accuracy", "")),
        "f1": _fmt(metrics.get("f1", "")),
        "rmse": _fmt(metrics.get("rmse", "")),
        "mae": _fmt(metrics.get("mae", "")),
        "r2": _fmt(metrics.get("r2", "")),
        "spearman": _fmt(metrics.get("spearman", "")),
        "n_train": "",
        "n_valid": "",
        "n_test": str(n_test),
        "n_failed_smiles": str(n_failed_smiles),
        "baseline_method": "RDKit descriptor/rule baseline",
        "baseline_metric_primary_value": _fmt(baseline_metric_primary_value),
        "validation_status": validation_status,
        "limitation_note": DEFAULT_LIMITATION_NOTE,
    }


def write_evaluation_outputs(
    *,
    summary_path: Path,
    details_path: Path,
    result: EvaluationResult,
) -> dict[str, int]:
    """Write evaluation summary and detail CSV files."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.summary_rows)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.detail_rows)
    return {"summary": len(result.summary_rows), "details": len(result.detail_rows)}
