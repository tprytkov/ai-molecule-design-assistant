"""Simple ADMET model registry.

The first implementation intentionally enables only the descriptor/rule
fallback. Future endpoint-specific entries describe the metadata required for
real model-based ADMET prediction without downloading or loading model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.admet.admet_schema import (
    MODEL_BACKEND_FALLBACK,
    MODEL_ID_FALLBACK,
    TRAINING_DATASET_FALLBACK,
)


@dataclass(frozen=True)
class ADMETModelRegistryEntry:
    """Metadata for one ADMET endpoint model or fallback."""

    endpoint: str
    model_id: str
    backend: str
    cache_required: bool
    training_dataset: str
    enabled: bool
    notes: str


ADMET_MODEL_REGISTRY = (
    ADMETModelRegistryEntry(
        endpoint="descriptor_rule_fallback",
        model_id=MODEL_ID_FALLBACK,
        backend=MODEL_BACKEND_FALLBACK,
        cache_required=False,
        training_dataset=TRAINING_DATASET_FALLBACK,
        enabled=True,
        notes=(
            "Enabled baseline using RDKit descriptors and conservative rules. "
            "This is not a validated endpoint-specific ADMET model."
        ),
    ),
    ADMETModelRegistryEntry(
        endpoint="bbb_permeability",
        model_id="future_endpoint_specific_bbb_model",
        backend="future_transformers_or_chemberta_finetuned_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes=(
            "Future use requires an endpoint-specific fine-tuned model cached "
            "under app_data/model_cache/huggingface. ChemBERTa pretraining or "
            "embeddings alone are not treated as ADMET prediction."
        ),
    ),
    ADMETModelRegistryEntry(
        endpoint="solubility",
        model_id="future_endpoint_specific_solubility_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific solubility model placeholder.",
    ),
    ADMETModelRegistryEntry(
        endpoint="cardiotoxicity_hERG",
        model_id="future_endpoint_specific_herg_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific hERG/cardiotoxicity model placeholder.",
    ),
    ADMETModelRegistryEntry(
        endpoint="cyp_inhibition",
        model_id="future_endpoint_specific_cyp_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific CYP inhibition model placeholder.",
    ),
    ADMETModelRegistryEntry(
        endpoint="general_toxicity",
        model_id="future_endpoint_specific_toxicity_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific toxicity model placeholder.",
    ),
)


def enabled_registry_entries() -> tuple[ADMETModelRegistryEntry, ...]:
    """Return currently enabled ADMET registry entries."""
    return tuple(entry for entry in ADMET_MODEL_REGISTRY if entry.enabled)
