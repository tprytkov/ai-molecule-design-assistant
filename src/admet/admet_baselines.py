"""RDKit descriptor/rule baseline helpers for ADMET benchmarking."""

from __future__ import annotations

from typing import Iterable, Mapping

from src.admet.admet_predictor import descriptor_from_smiles, endpoint_label


def label_to_binary(label: str) -> int:
    """Map app triage labels to a conservative binary label."""
    return 1 if str(label).strip().lower() == "favorable" else 0


def rdkit_rule_predictions(rows: Iterable[Mapping[str, str]], endpoint: str) -> list[dict[str, object]]:
    """Return RDKit/rule baseline predictions for benchmark rows."""
    predictions = []
    for row in rows:
        molecule_id = str(row.get("molecule_id", "")).strip()
        smiles = str(row.get("smiles", "")).strip()
        record = descriptor_from_smiles(molecule_id, smiles)
        label, value = endpoint_label(record, endpoint)
        predictions.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "prediction_label": label,
                "prediction_value": value,
                "prediction_binary": label_to_binary(label),
                "inference_status": "available" if record.valid else "failed_smiles",
            }
        )
    return predictions
