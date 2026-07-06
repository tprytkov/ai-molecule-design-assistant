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
from src.optional_domain_models import CHEMBERTA_BBB_MODEL_ID, CHEMBERTA_MLM_MODEL_ID, CHEMBERTA_MTR_MODEL_ID


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
    endpoint_name: str = ""
    model_family: str = ""
    task_type: str = ""
    expected_input: str = "SMILES"
    expected_output: str = ""
    label_mapping: str = ""
    benchmark_dataset: str = ""
    model_status: str = ""
    validation_status: str = "not_evaluated"
    scientific_note: str = ""


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
        endpoint_name="descriptor_rule_fallback",
        model_family="RDKit rules",
        task_type="rule_based_triage",
        expected_output="favorable/moderate/caution/unavailable",
        label_mapping="rule labels",
        benchmark_dataset="not_applicable",
        model_status="fallback_descriptor_rule",
        validation_status="not_evaluated",
        scientific_note="Descriptor/rule baseline for computational triage only.",
    ),
    ADMETModelRegistryEntry(
        endpoint="bbb_permeability",
        model_id=CHEMBERTA_BBB_MODEL_ID,
        backend="transformers-sequence-classification",
        cache_required=True,
        training_dataset="public_hf_model_card_bbb_permeability",
        enabled=True,
        notes=(
            "Experimental public Hugging Face tuned ChemBERTa BBB classifier. "
            "Used only when cached locally; otherwise BBB falls back to descriptor "
            "rules. This is not experimental ADMET evidence."
        ),
        endpoint_name="bbb_permeability",
        model_family="ChemBERTa-like sequence classifier",
        task_type="binary_classification",
        expected_output="BBB permeable vs non-permeable class probability",
        label_mapping="model id2label; must be verified during evaluation",
        benchmark_dataset="tdc_admet_bbb_martins",
        model_status="experimental_public_model",
        validation_status="not_evaluated",
        scientific_note=(
            "Endpoint-specific public model candidate. Use app predictions only "
            "after benchmark_passed or explicit experimental override."
        ),
    ),
    ADMETModelRegistryEntry(
        endpoint="molecular_embeddings",
        model_id=CHEMBERTA_MLM_MODEL_ID,
        backend="transformers",
        cache_required=True,
        training_dataset="masked_language_model_pretraining",
        enabled=False,
        notes=(
            "ChemBERTa-77M-MLM is used for embeddings / representation only. It "
            "is not treated as ADMET prediction or represented as a validated "
            "ADMET endpoint predictor."
        ),
        endpoint_name="molecular_embeddings",
        model_family="ChemBERTa masked language model",
        task_type="embedding",
        expected_output="molecular representation only",
        label_mapping="not_applicable",
        benchmark_dataset="not_applicable_for_admet_prediction",
        model_status="embedding_pretraining_model",
        validation_status="unavailable",
        scientific_note="Generic ChemBERTa embeddings are not ADMET predictors.",
    ),
    ADMETModelRegistryEntry(
        endpoint="experimental_molecular_property_regression",
        model_id=CHEMBERTA_MTR_MODEL_ID,
        backend="transformers",
        cache_required=True,
        training_dataset="public_hf_model_card_multiple_tasks",
        enabled=False,
        notes=(
            "Experimental regression candidate with status "
            "experimental_no_detailed_model_card unless separately validated."
        ),
        endpoint_name="experimental_molecular_property_regression",
        model_family="ChemBERTa-like molecular property model",
        task_type="regression",
        expected_output="model-specific regression output",
        label_mapping="not_applicable",
        benchmark_dataset="metadata_required_before_use",
        model_status="experimental_public_model",
        validation_status="metadata_incomplete",
        scientific_note="Not enabled for ADMET predictions until endpoint metadata is complete.",
    ),
    ADMETModelRegistryEntry(
        endpoint="solubility",
        model_id="future_endpoint_specific_solubility_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific solubility model placeholder.",
        endpoint_name="solubility",
        model_family="future endpoint-specific model",
        task_type="regression",
        expected_output="solubility value or label",
        benchmark_dataset="tdc_admet_solubility_or_moleculenet_esol",
        model_status="unavailable",
        validation_status="unavailable",
        scientific_note="Placeholder only.",
    ),
    ADMETModelRegistryEntry(
        endpoint="cardiotoxicity_hERG",
        model_id="future_endpoint_specific_herg_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific hERG/cardiotoxicity model placeholder.",
        endpoint_name="cardiotoxicity_hERG",
        model_family="future endpoint-specific model",
        task_type="binary_classification",
        expected_output="hERG/cardiotoxicity risk label",
        benchmark_dataset="tdc_admet_herg",
        model_status="unavailable",
        validation_status="unavailable",
        scientific_note="Placeholder only.",
    ),
    ADMETModelRegistryEntry(
        endpoint="cyp_inhibition",
        model_id="future_endpoint_specific_cyp_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific CYP inhibition model placeholder.",
        endpoint_name="cyp_inhibition",
        model_family="future endpoint-specific model",
        task_type="binary_classification",
        expected_output="CYP inhibition risk label",
        benchmark_dataset="tdc_admet_cyp",
        model_status="unavailable",
        validation_status="unavailable",
        scientific_note="Placeholder only.",
    ),
    ADMETModelRegistryEntry(
        endpoint="general_toxicity",
        model_id="future_endpoint_specific_toxicity_model",
        backend="future_transformers_or_classical_endpoint_model",
        cache_required=True,
        training_dataset="future_endpoint_specific_training_dataset_required",
        enabled=False,
        notes="Future endpoint-specific toxicity model placeholder.",
        endpoint_name="general_toxicity",
        model_family="future endpoint-specific model",
        task_type="binary_classification",
        expected_output="toxicity risk label",
        benchmark_dataset="tdc_admet_tox_or_moleculenet_tox21",
        model_status="unavailable",
        validation_status="unavailable",
        scientific_note="Placeholder only.",
    ),
)


def enabled_registry_entries() -> tuple[ADMETModelRegistryEntry, ...]:
    """Return currently enabled ADMET registry entries."""
    return tuple(entry for entry in ADMET_MODEL_REGISTRY if entry.enabled)
