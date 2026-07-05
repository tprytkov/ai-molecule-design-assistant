"""Schemas for ADMET descriptor/rule fallback outputs."""

ADMET_PREDICTION_COLUMNS = (
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

ADMET_SUMMARY_COLUMNS = (
    "molecule_id",
    "smiles",
    "bbb_prediction_label",
    "cns_property_flag",
    "toxicity_risk_flag",
    "admet_readiness_category",
    "model_status",
    "evidence_note",
)

MODEL_STATUS_FALLBACK = "fallback_descriptor_rule"
MODEL_BACKEND_FALLBACK = "RDKit descriptor/rule baseline"
MODEL_ID_FALLBACK = "rdkit_descriptor_rule_fallback_v1"
MODEL_CACHE_STATUS_FALLBACK = "not_required"
TRAINING_DATASET_FALLBACK = "not_applicable_descriptor_rules"
ADMET_EVIDENCE_NOTE = (
    "Heuristic RDKit descriptor/rule baseline for research triage only; this is "
    "not a validated ADMET prediction, experimental ADMET evidence, safety "
    "assessment, toxicity assessment, or clinical evidence."
)
