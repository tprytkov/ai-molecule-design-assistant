"""Evaluate cached endpoint-specific ADMET models against benchmark CSV rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.admet.admet_benchmarks import read_local_benchmark_csv
from src.admet.admet_model_evaluation import (
    EvaluationResult,
    evaluate_classification_model,
    evaluate_regression_model,
    write_evaluation_outputs,
)
from src.model_source_status import HUGGINGFACE_CACHE_DIR, huggingface_model_cache_candidates
from src.optional_domain_models import CHEMBERTA_BBB_MODEL_ID


class LocalSequenceClassificationPredictor:
    """Local cached Transformers sequence-classification predictor."""

    def __init__(self, model_id: str, *, positive_label: str = "") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for model evaluation.") from exc
        self._torch = torch
        model_source = local_model_source(model_id)
        self.positive_label = positive_label.strip().lower()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(HUGGINGFACE_CACHE_DIR),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            cache_dir=str(HUGGINGFACE_CACHE_DIR),
            local_files_only=True,
        )
        self.model.eval()
        labels = {
            int(index): str(label).lower()
            for index, label in getattr(self.model.config, "id2label", {}).items()
        }
        self.id2label = labels
        self.positive_label_index = self._resolve_positive_label_index(labels)
        self.label_mapping_status = (
            "documented"
            if self.positive_label
            and any(self.positive_label in label for label in labels.values())
            else "metadata_incomplete"
        )

    def predict_proba(self, smiles: str) -> float:
        """Return probability for the configured positive class."""
        encoded = self.tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self._torch.nn.functional.softmax(logits, dim=-1)[0]
        if self.positive_label_index is not None:
            return float(probabilities[self.positive_label_index].item())
        if self.positive_label:
            for index, label in self.id2label.items():
                if self.positive_label in label:
                    return float(probabilities[index].item())
        if len(probabilities) > 1:
            return float(probabilities[1].item())
        return float(probabilities[0].item())

    def _resolve_positive_label_index(self, labels: dict[int, str]) -> int | None:
        if self.positive_label:
            for index, label in labels.items():
                if self.positive_label in label:
                    return index
        return 1 if 1 in labels else None


class LocalRegressionPredictor:
    """Local cached Transformers regression predictor."""

    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for model evaluation.") from exc
        self._torch = torch
        model_source = local_model_source(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(HUGGINGFACE_CACHE_DIR),
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            cache_dir=str(HUGGINGFACE_CACHE_DIR),
            local_files_only=True,
        )
        self.model.eval()

    def predict_value(self, smiles: str) -> float:
        """Return a numeric model-head output."""
        encoded = self.tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
        return float(logits.reshape(-1)[0].item())


def local_model_source(model_id: str) -> str:
    """Return a local directory source when a non-symlink cache copy exists."""
    for path in huggingface_model_cache_candidates(model_id):
        if (path / "config.json").exists():
            return str(path)
        snapshots = path / "snapshots"
        if snapshots.exists():
            for snapshot in sorted(snapshots.iterdir(), reverse=True):
                if (snapshot / "config.json").exists():
                    return str(snapshot)
    return model_id


def unavailable_result(
    *,
    endpoint_name: str,
    model_id: str,
    benchmark_dataset: str,
    split: str,
    task_type: str,
    error_message: str,
) -> EvaluationResult:
    """Build a small failed-evaluation result when the model cannot load."""
    summary = {
        "endpoint_name": endpoint_name,
        "model_id": model_id,
        "benchmark_dataset": benchmark_dataset,
        "split": split,
        "task_type": task_type,
        "metric_primary": "",
        "metric_primary_value": "",
        "auroc": "",
        "auprc": "",
        "accuracy": "",
        "balanced_accuracy": "",
        "f1": "",
        "rmse": "",
        "mae": "",
        "r2": "",
        "spearman": "",
        "n_train": "",
        "n_valid": "",
        "n_test": "0",
        "n_failed_smiles": "0",
        "baseline_method": "RDKit descriptor/rule baseline",
        "baseline_metric_primary_value": "",
        "validation_status": "unavailable",
        "limitation_note": (
            "Model could not be loaded from the app-managed cache; app predictions "
            "must use the RDKit descriptor/rule fallback. "
            f"{type(error_message).__name__ if not isinstance(error_message, str) else ''}"
        ).strip(),
    }
    detail = {
        "endpoint_name": endpoint_name,
        "model_id": model_id,
        "molecule_id": "",
        "smiles": "",
        "true_label_or_value": "",
        "predicted_label_or_value": "",
        "prediction_probability": "",
        "prediction_error": str(error_message)[:300],
        "split": split,
        "inference_status": "model_unavailable",
    }
    return EvaluationResult([summary], [detail])


def mark_metadata_incomplete(result: EvaluationResult, note: str) -> EvaluationResult:
    """Return result rows with honest metadata-incomplete validation status."""
    summary_rows = []
    for row in result.summary_rows:
        updated = dict(row)
        updated["validation_status"] = "metadata_incomplete"
        existing_note = str(updated.get("limitation_note", "")).strip()
        updated["limitation_note"] = f"{existing_note} {note}".strip()
        summary_rows.append(updated)
    return EvaluationResult(summary_rows, result.detail_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a cached endpoint-specific ADMET model."
    )
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-name", default="bbb_permeability")
    parser.add_argument("--model-id", default=CHEMBERTA_BBB_MODEL_ID)
    parser.add_argument("--benchmark-dataset", default="local_admet_benchmark_csv")
    parser.add_argument(
        "--task-type",
        choices=("binary_classification", "regression"),
        default="binary_classification",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--positive-label", default="perme")
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--regression-rmse-threshold", type=float, default=float("inf"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run endpoint-specific model evaluation."""
    args = build_parser().parse_args(argv)
    rows = read_local_benchmark_csv(args.benchmark_csv)
    try:
        if args.task_type == "binary_classification":
            predictor = LocalSequenceClassificationPredictor(
                args.model_id,
                positive_label=args.positive_label,
            )
            result = evaluate_classification_model(
                endpoint_name=args.endpoint_name,
                model_id=args.model_id,
                benchmark_dataset=args.benchmark_dataset,
                rows=rows,
                predictor=predictor,
                split=args.split,
                baseline_endpoint="bbb_permeability_cns_likeness",
                validation_threshold=args.classification_threshold,
            )
            if predictor.label_mapping_status == "metadata_incomplete":
                result = mark_metadata_incomplete(
                    result,
                    "Model config uses generic LABEL_0/LABEL_1 labels; metrics were "
                    "computed with class index 1 treated as BBB-permeable, but the "
                    "model is not marked benchmark_passed until label mapping is documented.",
                )
        else:
            result = evaluate_regression_model(
                endpoint_name=args.endpoint_name,
                model_id=args.model_id,
                benchmark_dataset=args.benchmark_dataset,
                rows=rows,
                predictor=LocalRegressionPredictor(args.model_id),
                split=args.split,
                validation_threshold=args.regression_rmse_threshold,
            )
    except Exception as exc:
        result = unavailable_result(
            endpoint_name=args.endpoint_name,
            model_id=args.model_id,
            benchmark_dataset=args.benchmark_dataset,
            split=args.split,
            task_type=args.task_type,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    counts = write_evaluation_outputs(
        summary_path=args.output_dir / "admet_model_evaluation_summary.csv",
        details_path=args.output_dir / "admet_model_evaluation_details.csv",
        result=result,
    )
    status = result.summary_rows[0].get("validation_status", "not_evaluated")
    print(
        "Wrote "
        f"{counts['summary']} ADMET evaluation summary rows and "
        f"{counts['details']} details rows. validation_status={status}"
    )
    return 0 if status in {"benchmark_passed", "benchmark_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
