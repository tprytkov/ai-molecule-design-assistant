"""Schema for target profile metadata."""

TARGET_PROFILE_COLUMNS = (
    "target_id",
    "target_name",
    "gene_symbol",
    "organism",
    "uniprot_id",
    "pdb_id",
    "protein_structure_source",
    "binding_site_description",
    "disease_context",
    "mechanism_context",
    "reference_ligands",
    "docking_protocol_note",
    "target_relevance_note",
    "disclaimer",
)

TARGET_SOURCE_DEMO = "demo_provided"
TARGET_SOURCE_USER = "user_provided"
TARGET_SOURCE_MISSING = "missing"

TARGET_DISCLAIMER = (
    "Target metadata is project setup context for computational triage only; "
    "it is not experimental target validation, clinical rationale, or evidence "
    "of efficacy or safety."
)
