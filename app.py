"""Streamlit dashboard for local molecule-intelligence outputs."""

from __future__ import annotations

import html
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem

try:
    import streamlit_shadcn_ui as shadcn_ui
except Exception:
    shadcn_ui = None

from src.biomedical_evidence import biomedical_evidence_csv
from src.compound_qa import compound_qa
from src.chemberta_embeddings import (
    chemberta_embeddings_csv,
    merge_chemberta_into_prioritized,
    visualization_coordinates_csv,
)
from src.chemical_identity import UrllibIdentityClient, chemical_identity_csv
from src.compound_context import compound_context_csv
from src.compound_search import top_hits_csv
from src.descriptors import descriptor_csv
from src.patent_evidence_embeddings import patent_evidence_embeddings_csv
from src.optional_domain_models import (
    ALLOW_LOCAL_MODEL_DOWNLOADS_ENV,
    BIOMEDICAL_MODEL_OPTIONS,
    CLOUD_SAFE_FALLBACK_LABEL,
    CUSTOM_MODEL_LABEL,
    FALLBACK_MODEL_ID,
    PATENT_MODEL_OPTIONS,
    DomainModelUnavailableError,
    encoder_metadata,
    load_optional_model,
    resolve_model_selection,
)
from src.pipeline import (
    PipelinePaths,
    build_paths,
    csv_has_data_rows,
    run_pipeline,
    write_surechembl_not_run_csv,
)
from src.public_lookup import (
    UrllibJsonClient,
    lookup_chembl,
    lookup_pubchem,
    placeholder_result,
    write_output_csv as write_public_lookup_csv,
)
from src.scoring import scoring_csv
from src.similarity import similarity_csv
from src.standardize import standardize_csv
from src.surechembl_lookup import (
    UrllibSurechemblClient,
    lookup_online_rows,
    write_output_csv as write_surechembl_output_csv,
)
from src.text_nlp import text_nlp_csv


OUTPUT_FILES = {
    "prioritization": "prioritization_results.csv",
    "descriptors": "descriptors.csv",
    "similarity": "similarity.csv",
    "public_lookup": "public_lookup.csv",
    "chemical_identity": "chemical_identity.csv",
    "text_nlp": "text_nlp.csv",
    "biomedical_evidence": "biomedical_evidence.csv",
    "patent_evidence_embeddings": "patent_evidence_embeddings.csv",
    "surechembl": "surechembl_evidence.csv",
    "visualization": "visualization_coordinates.csv",
    "standardized": "standardized.csv",
    "compound_context": "compound_context.csv",
    "similarity_top_hits": "similarity_top_hits.csv",
    "chemberta_embeddings": "chemberta_embeddings.csv",
}
APP_TITLE = "AI Molecule Design Assistant"
IMPORTANT_COLUMNS = [
    "molecule_id",
    "prioritization_score",
    "prioritization_score_with_nlp",
    "best_reference_name",
    "tanimoto_similarity",
    "pubchem_status",
    "chembl_status",
    "surechembl_query_status",
    "chemberta_status",
    "nlp_status",
    "known_public_match",
]
STATUS_COLUMNS = [
    "pubchem_status",
    "chembl_status",
    "surechembl_query_status",
    "chemberta_status",
    "nlp_status",
]
DESCRIPTOR_COLUMNS = ["molecular_weight", "logp", "tpsa", "qed"]
DISPLAY_LABELS = {
    "molecule_id": "Molecule ID",
    "smiles": "SMILES",
    "canonical_smiles": "Canonical SMILES",
    "prioritization_score_with_nlp": "Research-prioritization score",
    "prioritization_score": "Base prioritization score",
    "prioritization_category_with_nlp": "Research-prioritization category",
    "prioritization_category": "Base prioritization category",
    "tanimoto_similarity": "Best reference similarity",
    "similarity_category": "Structure match category",
    "best_reference_name": "Closest reference compound",
    "source_type": "Source type",
    "reference_id": "Reference ID",
    "reference_name": "Reference name",
    "reference_role": "Reference role",
    "target": "Target",
    "cluster_display": "Chemical-space cluster",
    "nearest_reference_id": "Nearest reference ID",
    "nearest_reference_name": "Nearest reference",
    "nearest_reference_similarity": "Nearest reference similarity",
    "nearest_reference_interpretation": "Nearest reference interpretation",
    "known_public_match": "Exact public match",
    "pubchem_status": "PubChem status",
    "chembl_status": "ChEMBL status",
    "surechembl_query_status": "SureChEMBL status",
    "chemberta_status": "ChemBERTa status",
    "nlp_status": "Text-evidence status",
    "biomedical_model_name": "Biomedical model",
    "biomedical_model_status": "Biomedical model status",
    "biomedical_evidence_status": "Biomedical evidence status",
    "biomedical_similarity_score": "Biomedical similarity score",
    "biomedical_relevance_category": "Biomedical relevance",
    "biomedical_evidence_count": "Biomedical evidence count",
    "top_biomedical_evidence_id": "Top biomedical evidence ID",
    "top_biomedical_evidence_text": "Top biomedical evidence text",
    "patent_model_name": "Patent/IP model",
    "patent_model_status": "Patent/IP model status",
    "patent_evidence_status": "Patent-text embedding status",
    "patent_similarity_score": "Patent/IP similarity score",
    "patent_relevance_category": "Patent/IP relevance",
    "patent_evidence_count": "Patent evidence count",
    "top_patent_evidence_id": "Top patent evidence ID",
    "top_patent_evidence_text": "Top patent evidence text",
    "surechembl_structure_status": "SureChEMBL structure status",
    "patent_document_metadata_status": "Patent document metadata status",
    "molecular_weight": "Molecular weight",
    "logp": "LogP",
    "tpsa": "TPSA",
    "qed": "QED",
    "lipinski_pass": "Lipinski",
    "druglikeness_category": "Drug-likeness category",
    "druglikeness_score": "Drug-likeness score",
    "druglikeness_flags": "Drug-likeness flags",
    "mw_status": "MW status",
    "logp_status": "LogP status",
    "tpsa_status": "TPSA status",
    "qed_status": "QED status",
    "lipinski_status": "Lipinski status",
    "hbd": "Hydrogen-bond donors",
    "hba": "Hydrogen-bond acceptors",
    "rotatable_bonds": "Rotatable bonds",
    "identity_status": "Identity status",
    "identity_confidence": "Identity confidence",
    "exact_public_name": "Exact public name",
    "preferred_name": "Preferred name",
    "iupac_name": "IUPAC name",
    "synonyms": "Synonyms",
    "name_source": "Name source",
    "inchikey": "InChIKey",
    "inchi_key": "InChIKey",
    "pubchem_cid": "PubChem CID",
    "chembl_id": "ChEMBL ID",
    "surechembl_id": "SureChEMBL ID",
    "compound_name": "Compound name",
    "patent_id": "Patent document ID",
    "patent_number": "Patent number",
    "patent_title": "Patent title",
    "patent_date": "Patent date",
    "patent_section": "Metadata source",
    "patent_metadata_status": "Document metadata status",
    "patent_metadata_source": "Patent metadata source",
    "Molecule ID": "Molecule ID",
    "Report file": "Report file",
    "Priority/design category": "Priority/design category",
    "Exact public identity": "Exact public identity",
    "Drug-likeness category": "Drug-likeness category",
    "Text-evidence status": "Text-evidence status",
    "Download report button": "Download report button",
    "lookup_status": "Lookup status",
    "source_database": "Public database",
    "public_id": "Public ID",
    "public_name": "Public name",
    "evidence_note": "Evidence note",
    "cluster_id": "Chemical-space cluster",
    "nlp_relevance_category": "Text-evidence relevance",
    "max_relevance_score": "Top text-evidence score",
    "prioritization_category_with_nlp": "Final design category",
    "prioritization_category": "Design category",
    "valid_smiles": "Valid SMILES",
    "novelty_flag": "Public-database differentiation",
    "x": "Chemical space dimension 1",
    "y": "Chemical space dimension 2",
}
CHEMICAL_SPACE_EXPLANATION = (
    "Each point is one molecule. Molecules close together have similar "
    "ChemBERTa molecular embeddings. Use this plot to identify clusters, "
    "outliers, and chemically distinct candidates."
)
SCORE_SIMILARITY_EXPLANATION = (
    "High score with low reference similarity suggests a property-favorable "
    "but structurally differentiated molecule. High score with high reference "
    "similarity suggests a candidate closer to the uploaded reference panel."
)
PROPERTY_DISTRIBUTIONS_EXPLANATION = (
    "These histograms summarize molecular properties such as molecular weight, "
    "LogP, TPSA, and QED."
)
DRUGLIKENESS_EXPLANATION = (
    "This step helps identify molecules with favorable, borderline, or "
    "unfavorable drug-like property profiles before final prioritization."
)
DRUGLIKENESS_COLORS = {
    "favorable": "#2e7d32",
    "borderline": "#f9a825",
    "unfavorable": "#c62828",
    "invalid": "#616161",
}
MOLECULE_STATUS_COLORS = {
    "green": "#2e7d32",
    "yellow": "#f9a825",
    "red": "#c62828",
    "gray": "#757575",
}
MOLECULE_STATUS_ORDER = ["green", "yellow", "red", "gray"]
IDENTITY_DISPLAY_COLORS = {
    "Exact public identity": MOLECULE_STATUS_COLORS["green"],
    "Generated IUPAC name only": MOLECULE_STATUS_COLORS["yellow"],
    "No public identity": MOLECULE_STATUS_COLORS["red"],
    "Not queried": MOLECULE_STATUS_COLORS["gray"],
    "Invalid SMILES": MOLECULE_STATUS_COLORS["red"],
    "Lookup error": MOLECULE_STATUS_COLORS["red"],
}
IDENTITY_DISPLAY_ORDER = list(IDENTITY_DISPLAY_COLORS)
DRUGLIKENESS_ICONS = {
    "favorable": "✅ favorable",
    "borderline": "⚠️ borderline",
    "unfavorable": "❌ unfavorable",
    "invalid": "⛔ invalid",
}
ONLINE_LOOKUP_NOTE = (
    "Online lookup is enabled by default for new analyses. PubChem, ChEMBL, "
    "and SureChEMBL are called only after you click Run analysis."
)
RDKIT_DRAWING_UNAVAILABLE = False
LOAD_EXISTING_NOTE = (
    "This mode only loads an existing output folder. It does not rerun the "
    "pipeline or call online services."
)
WELCOME_TEXT = (
    "This app is designed for molecules generated by generative AI models. It "
    "helps researchers move from raw generated SMILES to an interpretable set "
    "of candidates by checking chemical validity, public identity, public "
    "database evidence, drug-like properties, chemical-space position, text "
    "evidence, and early IP-potential research signals. The IP-potential score "
    "is only a computational triage signal and is not a legal opinion or a "
    "determination of patentability, novelty, FTO, ownership, infringement "
    "risk, efficacy, safety, or clinical value."
)
START_GUIDANCE = (
    "Start with the guided example workflow to learn how each evidence stage "
    "supports molecule-design prioritization, then evaluate your own generated SMILES."
)
ABOUT_WORKFLOW_SECTIONS = (
    (
        "1. SMILES validation and standardization",
        "Generated SMILES are parsed and standardized before downstream analysis. "
        "This establishes a chemically interpretable representation for identifier "
        "generation, descriptor calculation, fingerprints, and molecular embeddings. "
        "[RDKit documentation](https://www.rdkit.org/docs/) describes the open-source "
        "cheminformatics functionality used for molecular parsing and representation.",
    ),
    (
        "2. Chemical identity lookup",
        "Standardized structures are assigned structure-derived identifiers such as "
        "InChIKey and are checked for exact public records when lookup services are "
        "enabled. [PubChem PUG-REST](https://pubmed.ncbi.nlm.nih.gov/27424744/) "
        "provides programmatic access to PubChem identifiers and properties, while "
        "[ChEMBL web services](https://academic.oup.com/nar/article/43/W1/W612/2467881) "
        "provide access to curated compound and bioactivity records.",
    ),
    (
        "3. Public database evidence",
        "Exact PubChem or ChEMBL matches indicate that a standardized structure is "
        "represented in those public resources. "
        "[SureChEMBL](https://academic.oup.com/nar/article/44/D1/D1220/2503102) "
        "can add structure-level evidence extracted from chemically annotated patent "
        "documents. These results are interpreted only as public structure evidence.",
    ),
    (
        "4. RDKit drug-likeness",
        "RDKit descriptors, fingerprint similarity, Lipinski-style property checks, "
        "and QED are presented as design heuristics. The "
        "[Lipinski framework](https://doi.org/10.1016/S0169-409X(00)00129-0) "
        "summarizes empirical property ranges, and "
        "[QED](https://doi.org/10.1038/nchem.1243) combines molecular-property "
        "distributions into a quantitative drug-likeness estimate. Neither constitutes "
        "evidence of biological activity or safety.",
    ),
    (
        "5. ChemBERTa chemical-space embeddings",
        "[ChemBERTa](https://arxiv.org/abs/2010.09885) uses transformer-based "
        "self-supervised learning on SMILES to construct molecular representations. "
        "Low-dimensional views may use [UMAP](https://arxiv.org/abs/1802.03426) to "
        "display clusters, outliers, and reference-like neighborhoods. These plots are "
        "exploratory representations rather than experimental validation.",
    ),
    (
        "6. Biomedical evidence and biological context",
        "After molecular identity and context are available, a lightweight general "
        "sentence-transformer baseline can compare molecule-context summaries with "
        "user-provided biomedical evidence text. BioBERT/PubMedBERT-style biomedical "
        "sentence embedding models are optional advanced local/cached models. If the "
        "configured model is unavailable in the public Streamlit Cloud app, this step "
        "writes a valid skipped output; skipped biomedical evidence is not an error. The "
        "[Sentence-BERT](https://arxiv.org/abs/1908.10084) approach and the "
        "[Sentence Transformers documentation](https://www.sbert.net/) describe the "
        "methodological and software basis for efficient semantic similarity matching. "
        "This stage organizes evidence for review and hypothesis generation; it does "
        "not establish biological activity.",
    ),
    (
        "7. Patent/IP-context evidence",
        "Patent/IP-context evidence is represented as a separate optional evidence "
        "embedding stage. PaECTER/patent-BERT-style encoders are optional advanced "
        "local/cached models for comparing molecule IP-context summaries with patent "
        "text signals. SureChEMBL structure evidence, patent document metadata, and "
        "patent-text embedding evidence are preserved separately. These outputs are "
        "research triage signals, not legal conclusions.",
    ),
    (
        "8. Final design prioritization",
        "The final ranking integrates chemical identity, public-database status, "
        "RDKit drug-likeness, reference similarity, ChemBERTa chemical-space context, "
        "text-evidence matching, and evidence completeness. It is a transparent "
        "research-prioritization aid for selecting candidates for further computational "
        "or experimental review, not a prediction of efficacy, safety, novelty, or "
        "clinical value.",
    ),
)
WORKFLOW_STEP_NAMES = (
    "Step 1: Load and validate SMILES",
    "Step 2: Chemical identity",
    "Step 3: Public database lookup",
    "Step 4: RDKit molecular properties",
    "Step 5: ChemBERTa chemical space",
    "Step 6: Biomedical evidence and biological context",
    "Step 7: Patent/IP-context evidence",
    "Step 8: Final prioritization",
    "Step 9: Reports",
)
WORKFLOW_STEP_PARAGRAPHS = (
    "This step converts input SMILES into a consistent molecule table using "
    "[RDKit](https://www.rdkit.org/) structure parsing and standardization "
    "utilities. The expected output is `standardized.csv`, with molecule "
    "identifiers, canonical SMILES, InChIKeys, validity status, and parse-error "
    "messages where structures cannot be interpreted. These records define the "
    "molecular input used by downstream identity, descriptor, similarity, "
    "public-evidence, and reporting steps; invalid or ambiguous structures should "
    "be reviewed before interpretation.",
    "This step derives chemical identity annotations from standardized structures "
    "using [RDKit](https://www.rdkit.org/) identifiers and optional online lookup "
    "of public names when enabled. The expected output is `chemical_identity.csv`, "
    "which includes SMILES-derived identifiers such as InChI/InChIKey, lookup "
    "status, source, and available public names. Identity annotations support "
    "exact-match lookup, molecule tracking, and reproducible comparison across "
    "workflow steps, but they can depend on standardization, stereochemistry, "
    "salts, and tautomer handling.",
    "This step checks whether each standardized molecule has exact or related "
    "records in public chemistry and patent-associated resources by querying "
    "[PubChem](https://pubchem.ncbi.nlm.nih.gov/), "
    "[ChEMBL](https://www.ebi.ac.uk/chembl/), and "
    "[SureChEMBL](https://www.surechembl.org/) when those lookup modes or local "
    "evidence files are available. The expected outputs are `public_lookup.csv` "
    "and `surechembl_evidence.csv`, with match status, database identifiers, "
    "query status, similarity or structure-evidence categories, and evidence "
    "counts when available. These results help identify molecules or related "
    "chemistry already represented in public resources, but absence of a match "
    "does not prove novelty, patentability, or freedom to operate.",
    "This step calculates molecular descriptors, rule-based drug-likeness "
    "indicators, and reference-ligand similarity using "
    "[RDKit](https://www.rdkit.org/) descriptors and molecular fingerprints. The "
    "expected outputs are `descriptors.csv`, `similarity.csv`, and "
    "`similarity_top_hits.csv`, containing physicochemical properties, "
    "drug-likeness flags, closest reference compounds, Tanimoto similarity, and "
    "top-hit summaries. These results support early triage of molecular quality "
    "and reference proximity, but they do not predict potency, selectivity, "
    "toxicity, or experimental activity.",
    "This step places generated and reference molecules into a shared "
    "chemical-space view using optional "
    "[ChemBERTa](https://arxiv.org/abs/2010.09885) molecular embeddings generated "
    "with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), "
    "followed by [UMAP](https://umap-learn.readthedocs.io/en/latest/) when "
    "appropriate or deterministic PCA fallback coordinates. The expected outputs "
    "are `chemberta_embeddings.csv` and `visualization_coordinates.csv`, with "
    "embedding availability, model metadata, source-type labels, two-dimensional "
    "coordinates, cluster annotations, and nearest-reference relationships. The "
    "map helps inspect whether generated molecules overlap with or separate from "
    "reference chemistry, but two-dimensional projections are qualitative and "
    "should be interpreted with molecule-level similarity and descriptor outputs.",
    "This step links molecule context records to biomedical text evidence using "
    "local semantic-similarity tools based on "
    "[Sentence Transformers](https://www.sbert.net/) when available, with "
    "BioBERT/PubMedBERT-style local cached models supported through the same "
    "sentence-transformer interface. The expected outputs include "
    "`compound_context.csv`, `text_nlp.csv`, and `biomedical_evidence.csv`, with "
    "model status, evidence status, similarity score, evidence count, top evidence "
    "text, and relevance category. These outputs provide literature-context "
    "triage signals and model-availability transparency, but they are not evidence "
    "of biological activity, mechanism, efficacy, or safety.",
    "This step summarizes patent-context evidence using public structure evidence "
    "from [SureChEMBL](https://www.surechembl.org/) and optional patent-text "
    "embedding tools configured through "
    "[Sentence Transformers](https://www.sbert.net/), including PaECTER- or "
    "patent-BERT-style local cached encoders when supplied. The expected output is "
    "`patent_evidence_embeddings.csv`, with model status, SureChEMBL structure "
    "status, patent similarity score, top evidence text, evidence counts, and "
    "evidence notes. These results support early IP-context triage, but they do "
    "not determine novelty, patentability, freedom to operate, ownership, "
    "infringement risk, or legal strategy.",
    "This step combines standardized identity, public-evidence signals, molecular "
    "descriptors, reference similarity, ChemBERTa availability, and available "
    "contextual evidence into an interpretable prioritization table using the "
    "app's scoring workflow. The expected output is `prioritization_results.csv`, "
    "with ranked candidates, score components, known-public-match flags, "
    "novelty/IP-context categories, evidence-stage statuses, and explanatory "
    "notes. Scores are intended for computational triage and hypothesis generation "
    "only; they are not predictions of activity, safety, synthesizability, "
    "patentability, or clinical value.",
    "This step converts available molecule-level outputs into downloadable "
    "Markdown summary reports and supporting report images for selected top "
    "candidates. The expected outputs are `compound_intelligence_report_*.md` "
    "files under `reports/` and generated structure images under `report_images/`, "
    "preserving the evidence used during prioritization. Reports are intended to "
    "support review, documentation, and follow-up analysis; they should be checked "
    "by domain experts before experimental, legal, or investment decisions.",
)
WORKFLOW_STEP_REFERENCES = (
    (
        ("RDKit", "https://www.rdkit.org/"),
        ("RDKit documentation", "https://www.rdkit.org/docs/"),
    ),
    (
        ("RDKit", "https://www.rdkit.org/"),
        ("InChI Trust / IUPAC InChI", "https://iupac.org/who-we-are/divisions/division-details/inchi/"),
    ),
    (
        ("PubChem", "https://pubchem.ncbi.nlm.nih.gov/"),
        ("PubChem PUG-REST documentation", "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest"),
        ("ChEMBL", "https://www.ebi.ac.uk/chembl/"),
        ("SureChEMBL", "https://www.surechembl.org/"),
    ),
    (
        ("RDKit", "https://www.rdkit.org/"),
        ("RDKit descriptors documentation", "https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors"),
        ("RDKit fingerprints documentation", "https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity"),
    ),
    (
        ("ChemBERTa", "https://arxiv.org/abs/2010.09885"),
        ("Hugging Face Transformers", "https://huggingface.co/docs/transformers/index"),
        ("UMAP", "https://umap-learn.readthedocs.io/en/latest/"),
    ),
    (
        ("Sentence Transformers", "https://www.sbert.net/"),
        ("Hugging Face Transformers", "https://huggingface.co/docs/transformers/index"),
        ("Optional embedding models", "README.md#optional-embedding-models"),
    ),
    (
        ("SureChEMBL", "https://www.surechembl.org/"),
        ("Sentence Transformers", "https://www.sbert.net/"),
        ("Optional embedding models", "README.md#optional-embedding-models"),
    ),
    (
        ("Project scoring overview", "README.md#about-the-workflow"),
        ("RDKit", "https://www.rdkit.org/"),
        ("PubChem", "https://pubchem.ncbi.nlm.nih.gov/"),
        ("ChEMBL", "https://www.ebi.ac.uk/chembl/"),
        ("SureChEMBL", "https://www.surechembl.org/"),
    ),
    (
        ("Guided example outputs", "README.md#run-the-guided-example-pipeline"),
        ("Project scope", "README.md#scope"),
    ),
)
WORKFLOW_MODE_OPTIONS = (
    "Start",
    "Guided demo",
    "Analyze my molecules",
    "Load previous run",
)
STEP_NAVIGATION_LABELS = (
    "Overview",
    "Input data",
    "Standardization",
    "Chemical identity",
    "Public evidence",
    "Descriptors",
    "Chemical-space map",
    "Biomedical evidence",
    "Patent/IP-context evidence",
    "Prioritization",
    "Reports",
    "Downloads",
    "Settings",
)
STEP_NAVIGATION_TO_WORKFLOW_STEP = {
    "Standardization": 1,
    "Chemical identity": 2,
    "Public evidence": 3,
    "Descriptors": 4,
    "Chemical-space map": 5,
    "Biomedical evidence": 6,
    "Patent/IP-context evidence": 7,
    "Prioritization": 8,
    "Reports": 9,
}
WORKFLOW_STEP_TO_NAVIGATION_LABEL = {
    value: key for key, value in STEP_NAVIGATION_TO_WORKFLOW_STEP.items()
}
SIDEBAR_STEP_NAVIGATION_WIDGET_KEY = "_sidebar_step_navigation_widget"
PENDING_ACTIVE_RUN_PAGE_KEY = "pending_active_run_page"
STEP_OUTPUT_KEYS = {
    1: ("standardized",),
    2: ("chemical_identity",),
    3: ("public_lookup", "surechembl"),
    4: ("descriptors", "similarity", "similarity_top_hits"),
    5: ("chemberta_embeddings", "visualization"),
    6: ("compound_context", "text_nlp", "biomedical_evidence"),
    7: ("patent_evidence_embeddings",),
    8: ("prioritization",),
    9: ("reports",),
}
STEP_STATUS_ICONS = {
    "completed": "✓",
    "not_run": "○",
    "skipped": "⚠",
    "missing_prerequisite": "✗",
}
CUSTOM_ANALYSIS_STEPS = tuple(STEP_NAVIGATION_TO_WORKFLOW_STEP)
CUSTOM_ANALYSIS_DEPENDENCIES = {
    "Standardization": (),
    "Chemical identity": ("Standardization",),
    "Public evidence": ("Standardization", "Chemical identity"),
    "Descriptors": ("Standardization",),
    "Chemical-space map": ("Standardization",),
    "Biomedical evidence": ("Standardization",),
    "Patent/IP-context evidence": ("Standardization", "Public evidence"),
    "Prioritization": (
        "Standardization",
        "Descriptors",
        "Chemical identity",
    ),
    "Reports": ("Prioritization",),
}
FINAL_RANKING_EXPLANATION = (
    "Final ranking combines evidence from chemical identity, public lookup, "
    "RDKit descriptors, ChemBERTa embeddings, text evidence, and evidence "
    "availability. Biomedical and patent embedding outputs remain separate "
    "review evidence in this version."
)
DEMO_INPUT = Path("data/examples/druglike_candidate_demo.csv")
DEMO_REFERENCES = Path("data/examples/druglike_reference_panel.csv")
DEMO_TEXT_EVIDENCE = Path("data/examples/text_evidence_demo.csv")
EXAMPLE_FILE_NOTES = {
    DEMO_INPUT: (
        "Candidate file",
        {
            "molecule_id": "unique molecule identifier",
            "smiles": "generated molecule SMILES",
            "notes": "optional notes or source context for the generated molecule",
        },
        (
            "This file supplies the generated molecules that move through SMILES "
            "validation, public lookup, descriptor calculation, and final ranking."
        ),
    ),
    DEMO_REFERENCES: (
        "Reference file",
        {
            "reference_id": "known or comparison molecule",
            "reference_name": "known or comparison molecule",
            "name": "known or comparison molecule",
            "smiles": "reference molecule SMILES",
            "reference_role": "optional annotation describing how the reference is used",
            "target": "optional target or assay context annotation",
            "target_family": "optional target or biological context annotation",
            "evidence": "optional evidence annotation",
            "evidence_note": "optional evidence annotation",
            "notes": "optional annotation or evidence context",
        },
        (
            "This file provides known or comparison molecules for reference "
            "similarity and chemical-context interpretation."
        ),
    ),
    DEMO_TEXT_EVIDENCE: (
        "Text evidence file",
        {
            "evidence_id": "evidence row identifier",
            "text": "notes, assay description, literature text, or target context",
            "source": "optional metadata column",
            "source_type": "optional metadata column",
            "target": "optional metadata column",
            "target_family": "optional metadata column",
            "molecule_id": "optional metadata column linking evidence to a molecule",
            "notes": "optional metadata column",
        },
        (
            "This file supplies short text evidence that the workflow compares "
            "against molecule context during the text-evidence stage."
        ),
    ),
}
GUIDED_EXAMPLE_MAX_MOLECULES = None
PUBCHEM_PREFLIGHT_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/TXT"
)
PUBCHEM_PREFLIGHT_TIMEOUT = 5.0
ONLINE_LOOKUP_UNAVAILABLE_MESSAGE = (
    "Online database lookup is unavailable from this process."
)
ONLINE_LOOKUP_RESTART_MESSAGE = (
    "Restart Streamlit from a normal conda terminal, then run the guided example again."
)
STEP3_CHEMBL_UNAVAILABLE_WARNING = (
    "ChEMBL lookup was unavailable or failed for this run. PubChem and available "
    "public evidence were preserved. You can rerun Step 3 later."
)
STEP3_CHEMBL_WARNING_CARD = (
    "ChEMBL lookup failed or was unavailable. This is usually an online-service "
    "or network issue, not a molecule-processing error."
)
STEP3_DEGRADED_LOOKUP_WARNING = (
    "Public lookup evidence is incomplete because one or more online services "
    "were unavailable. Valid partial results were activated so downstream steps "
    "can continue in degraded mode."
)
STEP3_SUMMARY_METADATA_KEYS = {
    "__completion_status",
    "__degraded_sources",
    "__chembl_unavailable",
}
APP_USAGE_STEPS = (
    "Upload generated SMILES.",
    "Upload optional reference molecules.",
    "Choose workflow options.",
    "Click Run analysis.",
    "Review evidence status.",
    "Explore chemical-space and score plots.",
    "Select molecules and generate reports.",
)
APP_RUNS_DIR = Path("app_runs")
GENERATED_REQUIRED_COLUMNS = ("molecule_id", "smiles")
REFERENCE_OUTPUT_COLUMNS = (
    "reference_id",
    "reference_name",
    "smiles",
    "reference_source",
    "reference_source_id",
    "evidence_note",
)
TEXT_EVIDENCE_COLUMNS = ("evidence_id", "molecule_id", "source_type", "title", "text")
STATUS_LABELS = {
    "pubchem_status": "PubChem",
    "chembl_status": "ChEMBL",
    "surechembl_query_status": "SureChEMBL",
    "chemberta_status": "ChemBERTa",
    "nlp_status": "NLP",
}
STATUS_MEANINGS = {
    "no_match": "queried; no match found",
    "hit": "queried; match found",
    "match_found": "queried; match found",
    "not_queried": "not checked in this run",
    "lookup_error": "query failed",
    "available": "available",
    "not_run": "workflow step was skipped",
    "not_available": "not available",
}
STATUS_ICONS = {
    "available": "✅",
    "hit": "✅",
    "match_found": "✅",
    "no_match": "⚪",
    "lookup_error": "⚠️",
    "not_queried": "⏭️",
    "not_run": "⏭️",
}


@dataclass(frozen=True)
class LoadedOutputs:
    """Loaded CSV outputs and useful output paths."""

    output_dir: Path
    reports_dir: Path
    images_dir: Path
    paths: dict[str, Path]
    tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class UploadedRunPaths:
    """Paths created for one app-submitted analysis run."""

    run_dir: Path
    input_dir: Path
    output_dir: Path
    generated_smiles: Path
    references: Path
    text_evidence: Path


@dataclass(frozen=True)
class ActiveRunPathStatus:
    """Resolved active-run paths plus read-only path status details."""

    run_type: str
    workflow_mode: str
    output_dir: Path
    pipeline_paths: PipelinePaths
    paths_resolved: bool
    missing_inputs: tuple[str, ...]
    existing_outputs: tuple[str, ...]
    unresolved_paths: tuple[str, ...]


@dataclass(frozen=True)
class OnlineLookupPreflight:
    """Result and process diagnostics for the PubChem connection check."""

    available: bool
    python_executable: str
    url: str
    exception_type: str = ""
    exception_message: str = ""


@dataclass(frozen=True)
class Step3Progress:
    """One visible progress update during guided public-database lookup."""

    database: str
    completed: int
    total: int
    elapsed_seconds: float
    output_dir: Path


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if present, otherwise return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_run_name(name: str | None, timestamp: datetime | None = None) -> str:
    """Return a filesystem-safe run name."""
    cleaned = (name or "").strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    if cleaned:
        return cleaned
    active_time = timestamp or datetime.now()
    return active_time.strftime("run_%Y%m%d_%H%M%S")


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Validate a dataframe has required columns."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {', '.join(missing)}")


def read_uploaded_csv(uploaded_file: object, label: str) -> pd.DataFrame:
    """Read an uploaded Streamlit CSV object into a dataframe."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"{label} could not be read as CSV: {exc}") from exc


def validate_generated_smiles(df: pd.DataFrame) -> None:
    """Validate generated SMILES upload columns."""
    validate_required_columns(df, GENERATED_REQUIRED_COLUMNS, "Generated SMILES CSV")


def validate_reference_csv(df: pd.DataFrame) -> None:
    """Validate uploaded reference columns."""
    validate_required_columns(df, ("smiles",), "Reference CSV")


def normalize_reference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize user reference uploads to the pipeline reference schema."""
    if df.empty:
        return pd.DataFrame(columns=REFERENCE_OUTPUT_COLUMNS)
    validate_reference_csv(df)
    rows: list[dict[str, str]] = []
    for index, row in df.fillna("").iterrows():
        reference_id = str(row.get("reference_id", "")).strip() or f"ref_{index + 1:04d}"
        reference_name = (
            str(row.get("reference_name", "")).strip()
            or str(row.get("name", "")).strip()
            or reference_id
        )
        notes = []
        for column in ("reference_role", "target", "notes", "evidence_note"):
            value = str(row.get(column, "")).strip()
            if value:
                notes.append(f"{column}: {value}")
        rows.append(
            {
                "reference_id": reference_id,
                "reference_name": reference_name,
                "smiles": str(row.get("smiles", "")).strip(),
                "reference_source": str(row.get("reference_source", "")).strip()
                or "uploaded_reference",
                "reference_source_id": str(row.get("reference_source_id", "")).strip()
                or reference_id,
                "evidence_note": "; ".join(notes) or "Uploaded reference molecule.",
            }
        )
    return pd.DataFrame(rows, columns=REFERENCE_OUTPUT_COLUMNS)


def empty_reference_dataframe() -> pd.DataFrame:
    """Return an empty but schema-valid reference table."""
    return pd.DataFrame(columns=REFERENCE_OUTPUT_COLUMNS)


def empty_text_evidence_dataframe() -> pd.DataFrame:
    """Return an empty but schema-valid text evidence table."""
    return pd.DataFrame(columns=TEXT_EVIDENCE_COLUMNS)


def save_uploaded_file(uploaded_file: object, path: Path) -> None:
    """Save a Streamlit uploaded file object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        if hasattr(uploaded_file, "getbuffer"):
            output_file.write(uploaded_file.getbuffer())
        elif hasattr(uploaded_file, "getvalue"):
            output_file.write(uploaded_file.getvalue())
        else:
            shutil.copyfileobj(uploaded_file, output_file)


def prepare_app_run_inputs(
    *,
    run_name: str,
    generated_upload: object,
    reference_upload: object | None = None,
    text_upload: object | None = None,
    app_runs_dir: Path = APP_RUNS_DIR,
) -> UploadedRunPaths:
    """Save uploaded files and normalized pipeline inputs under app_runs."""
    safe_name = safe_run_name(run_name)
    run_dir = app_runs_dir / safe_name
    input_dir = run_dir / "input"
    output_dir = run_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_raw = input_dir / "uploaded_generated_smiles.csv"
    save_uploaded_file(generated_upload, generated_raw)
    generated_df = pd.read_csv(generated_raw)
    validate_generated_smiles(generated_df)
    generated_path = input_dir / "generated_smiles.csv"
    generated_df.to_csv(generated_path, index=False)

    reference_raw = input_dir / "uploaded_references.csv"
    if reference_upload is not None:
        save_uploaded_file(reference_upload, reference_raw)
        reference_df = normalize_reference_dataframe(pd.read_csv(reference_raw))
    else:
        reference_df = empty_reference_dataframe()
    reference_path = input_dir / "references.csv"
    reference_df.to_csv(reference_path, index=False)

    text_raw = input_dir / "uploaded_text_evidence.csv"
    if text_upload is not None:
        save_uploaded_file(text_upload, text_raw)
        text_df = pd.read_csv(text_raw)
    else:
        text_df = empty_text_evidence_dataframe()
    text_path = input_dir / "text_evidence.csv"
    text_df.to_csv(text_path, index=False)

    return UploadedRunPaths(
        run_dir=run_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        generated_smiles=generated_path,
        references=reference_path,
        text_evidence=text_path,
    )


def default_workflow_options() -> dict[str, object]:
    """Return new-analysis form defaults."""
    return {
        "online_lookup": True,
        "online_surechembl": True,
        "use_chemberta": True,
        "generate_reports": True,
        "report_only_fully_analyzed": True,
        "max_molecules": 10,
        "report_top_n": 5,
    }


def usage_guide_markdown() -> str:
    """Return the user-facing app workflow guide."""
    lines = ["### How to use this app", ""]
    lines.extend(
        f"{index}. {step}" for index, step in enumerate(APP_USAGE_STEPS, start=1)
    )
    return "\n".join(lines)


def run_uploaded_analysis(
    paths: UploadedRunPaths,
    *,
    online_lookup: bool = False,
    online_surechembl: bool = False,
    use_chemberta: bool = False,
    report_top_n: int | None = 5,
    report_only_fully_analyzed: bool = False,
    max_molecules: int | None = None,
) -> Path:
    """Run the local pipeline for uploaded app inputs."""
    pipeline_paths = build_paths(
        input_path=paths.generated_smiles,
        references_path=paths.references,
        text_evidence_path=paths.text_evidence,
        output_dir=paths.output_dir,
    )
    return run_pipeline(
        paths=pipeline_paths,
        online_lookup=online_lookup,
        refresh_public_lookup=online_lookup,
        online_surechembl=online_surechembl,
        skip_surechembl=not online_surechembl,
        use_chemberta=use_chemberta,
        max_molecules=max_molecules,
        report_top_n=report_top_n,
        report_only_fully_analyzed=report_only_fully_analyzed,
        report_dir=paths.output_dir / "reports",
        clean_report_dir=True,
    )



def get_step_artifacts(step_number: int, output_dir: str | Path) -> list[Path]:
    """Return output artifacts that belong on one workflow step page."""
    root = Path(output_dir)
    if step_number == 9:
        return [root / "reports"]
    artifacts = []
    for key in STEP_OUTPUT_KEYS.get(step_number, ()):
        if key == "reports":
            artifacts.append(root / "reports")
            continue
        filename = OUTPUT_FILES.get(key)
        if filename:
            artifacts.append(root / filename)
    return artifacts

def load_output_directory(output_dir: str | Path) -> LoadedOutputs:
    """Load known pipeline outputs from an output directory."""
    root = Path(output_dir)
    paths = {name: root / filename for name, filename in OUTPUT_FILES.items()}
    tables = {name: read_optional_csv(path) for name, path in paths.items()}
    tables["prioritization"] = reconcile_nlp_status(
        tables["prioritization"], tables["text_nlp"]
    )
    return LoadedOutputs(
        output_dir=root,
        reports_dir=root / "reports",
        images_dir=root / "report_images",
        paths=paths,
        tables=tables,
    )


def reconcile_nlp_status(
    prioritization: pd.DataFrame, text_nlp: pd.DataFrame
) -> pd.DataFrame:
    """Infer NLP status only when prioritization does not provide the column."""
    if (
        prioritization.empty
        or text_nlp.empty
        or "nlp_status" in prioritization.columns
        or "molecule_id" not in prioritization.columns
        or "molecule_id" not in text_nlp.columns
    ):
        return prioritization
    result = prioritization.copy()
    usable = text_nlp["molecule_id"].fillna("").astype(str).str.strip()
    available_ids = set(usable[usable.ne("")])
    if not available_ids:
        return result
    result["nlp_status"] = result["molecule_id"].astype(str).map(
        lambda molecule_id: (
            "available" if molecule_id in available_ids else "no_match"
        )
    )
    return result


def nlp_output_note(text_nlp_path: Path, text_nlp: pd.DataFrame) -> str:
    """Explain NLP availability for the selected output folder."""
    if not text_nlp_path.exists():
        return (
            "NLP was not run for this output folder because no text_nlp.csv "
            "file was found. Rerun the pipeline with --text-evidence to enable "
            "NLP evidence matching."
        )
    if text_nlp.empty:
        return "NLP output exists but contains no evidence matches."
    if text_nlp_model_unavailable(text_nlp):
        return (
            "Text-evidence NLP was skipped because the embedding model is "
            "unavailable in this cloud environment."
        )
    return (
        "NLP evidence matching was run using molecule context and text evidence."
    )


def text_nlp_model_unavailable(text_nlp: pd.DataFrame) -> bool:
    """Return whether text NLP output was written by the model fallback path."""
    if text_nlp.empty or "nlp_status" not in text_nlp.columns:
        return False
    statuses = text_nlp["nlp_status"].fillna("").astype(str).str.strip()
    return statuses.eq("model_unavailable").any()


def biomedical_model_unavailable(biomedical: pd.DataFrame) -> bool:
    """Return whether biomedical evidence was skipped by the model fallback."""
    if biomedical.empty or "biomedical_model_status" not in biomedical.columns:
        return False
    statuses = biomedical["biomedical_model_status"].fillna("").astype(str).str.strip()
    return statuses.eq("model_unavailable").any()


def patent_model_unavailable(patent: pd.DataFrame) -> bool:
    """Return whether patent evidence was skipped by the model fallback."""
    if patent.empty or "patent_model_status" not in patent.columns:
        return False
    statuses = patent["patent_model_status"].fillna("").astype(str).str.strip()
    return statuses.eq("model_unavailable").any()


def score_column(df: pd.DataFrame) -> str:
    """Return the preferred score column."""
    if "prioritization_score_with_nlp" in df.columns:
        return "prioritization_score_with_nlp"
    return "prioritization_score"


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected columns converted to numeric when present."""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def apply_filters(
    df: pd.DataFrame,
    *,
    min_score: float = 0.0,
    known_public_match: str = "All",
    status_filters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Filter prioritization rows for dashboard display."""
    if df.empty:
        return df
    result = coerce_numeric(df, ["prioritization_score", "prioritization_score_with_nlp", "tanimoto_similarity"])
    active_score = score_column(result)
    if active_score in result.columns:
        result = result[result[active_score].fillna(0) >= min_score]
    if known_public_match != "All" and "known_public_match" in result.columns:
        result = result[
            result["known_public_match"].astype(str).str.lower()
            == known_public_match.lower()
        ]
    for column, selected in (status_filters or {}).items():
        if selected and column in result.columns:
            result = result[result[column].astype(str).isin(selected)]
    return result


def ordered_columns(df: pd.DataFrame) -> list[str]:
    """Put high-signal columns first while preserving all remaining columns."""
    first = [column for column in IMPORTANT_COLUMNS if column in df.columns]
    rest = [column for column in df.columns if column not in first]
    return first + rest


def display_label(column: str) -> str:
    """Return a readable UI label without changing the CSV schema."""
    return DISPLAY_LABELS.get(column, column.replace("_", " ").strip().title())


def display_labels(columns: Iterable[str]) -> dict[str, str]:
    """Build a Plotly-compatible label mapping."""
    return {column: display_label(column) for column in columns}


def display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-only dataframe with readable column labels."""
    return df.rename(columns=display_labels(df.columns))


def shorten_display_text(value: object, max_chars: int = 80) -> str:
    """Shorten long cell values for compact preview tables."""
    text = "" if pd.isna(value) else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def compact_preview_dataframe(
    frame: pd.DataFrame,
    *,
    preview_rows: int = 10,
    max_columns: int = 10,
    max_text_chars: int = 80,
) -> pd.DataFrame:
    """Return a short, readable dataframe for collapsed artifact previews."""
    preview = frame.head(preview_rows).copy()
    if len(preview.columns) > max_columns:
        preview = preview.iloc[:, :max_columns].copy()
    for column in preview.columns:
        preview[column] = preview[column].map(
            lambda value: shorten_display_text(value, max_text_chars)
        )
    return display_dataframe(preview)


def molecule_status_color(value: object) -> str:
    """Map workflow statuses to the shared molecule color vocabulary."""
    status = str(value or "").strip().lower().replace(" ", "_")
    if status == "model_unavailable":
        return "gray"
    if any(token in status for token in ("not_run", "offline", "skipped")):
        return "gray"
    if any(
        token in status
        for token in (
            "unfavorable",
            "unavailable",
            "not_available",
            "missing",
            "invalid",
            "error",
            "no_match",
            "low",
        )
    ):
        return "red"
    if any(token in status for token in ("borderline", "partial", "moderate", "medium")):
        return "yellow"
    if any(token in status for token in ("favorable", "available", "valid", "high")):
        return "green"
    if status in {
        "true",
        "valid",
        "available",
        "hit",
        "match_found",
        "exact_public_identity",
        "exact_public_match",
        "favorable",
        "high",
    }:
        return "green"
    if status in {
        "borderline",
        "partial",
        "moderate",
        "medium",
        "generated_iupac_name_only",
        "reference_context",
        "weak_reference_context",
        "not_queried",
    }:
        return "yellow"
    return "red"


def validation_molecule_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one validation row per molecule for plotting and selection."""
    if frame.empty or "molecule_id" not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result["validation_status"] = result.get("valid_smiles", False).map(
        lambda value: "Valid" if str(value).lower() == "true" else "Invalid"
    )
    result["status_color"] = result["validation_status"].map(molecule_status_color)
    result["molecule_position"] = range(1, len(result) + 1)
    return result


def identity_molecule_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one chemical-identity row per molecule."""
    if frame.empty or "molecule_id" not in frame.columns:
        return pd.DataFrame()
    result = frame.drop_duplicates("molecule_id").copy()
    result["display_status"] = result.apply(
        chemical_identity_display_status,
        axis=1,
    )
    result["status_color"] = result["display_status"].map(
        {
            label: color_name
            for label, color_name in (
                ("Exact public identity", "green"),
                ("Generated IUPAC name only", "yellow"),
                ("No public identity", "red"),
                ("Not queried", "gray"),
                ("Invalid SMILES", "red"),
                ("Lookup error", "red"),
            )
        }
    )
    result["molecule_position"] = range(1, len(result) + 1)
    return result


def chemical_identity_display_status(row: pd.Series) -> str:
    """Return the final Chemical Identity display category for one CSV row."""
    identity = normalize_status(row.get("identity_status", "")).lower()
    lookup = normalize_status(row.get("lookup_status", "")).lower()
    if identity in {"invalid_smiles", "invalid_molecule"} or lookup in {
        "invalid_smiles",
        "invalid_molecule",
    }:
        return "Invalid SMILES"
    if identity == "lookup_error" or lookup == "lookup_error":
        return "Lookup error"
    if lookup == "not_queried":
        return "Not queried"
    if identity in {
        "exact_public_identity",
        "exact_pubchem_match",
        "exact_chembl_match",
    }:
        return "Exact public identity"
    if identity == "generated_iupac_name_only":
        return "Generated IUPAC name only"
    if identity == "no_public_identity":
        return "No public identity"
    return "No public identity"


def best_status(values: Iterable[object]) -> str:
    """Choose the most informative status from repeated evidence rows."""
    statuses = [normalize_status(value) for value in values]
    priority = (
        "match_found",
        "hit",
        "available",
        "no_match",
        "not_queried",
        "model_unavailable",
        "lookup_error",
        "error",
        "not_run",
        "not_available",
    )
    return next((status for status in priority if status in statuses), "not_available")


def public_evidence_molecule_dataframe(
    public_lookup: pd.DataFrame,
    surechembl: pd.DataFrame,
    molecule_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Build one public-evidence status row per molecule."""
    ids = {str(value) for value in molecule_ids if str(value).strip()}
    if "molecule_id" in public_lookup.columns:
        ids.update(public_lookup["molecule_id"].dropna().astype(str))
    if "molecule_id" in surechembl.columns:
        ids.update(surechembl["molecule_id"].dropna().astype(str))
    rows = []
    for molecule_id in sorted(ids):
        lookup_rows = (
            public_lookup[public_lookup["molecule_id"].astype(str) == molecule_id]
            if not public_lookup.empty and "molecule_id" in public_lookup.columns
            else pd.DataFrame()
        )
        source_statuses = {}
        if not lookup_rows.empty:
            for source, group in lookup_rows.groupby(
                lookup_rows["source_database"].fillna("").astype(str)
            ):
                source_statuses[source.lower()] = best_status(group["lookup_status"])
        sure_rows = (
            surechembl[surechembl["molecule_id"].astype(str) == molecule_id]
            if not surechembl.empty and "molecule_id" in surechembl.columns
            else pd.DataFrame()
        )
        row = {
            "molecule_id": molecule_id,
            "pubchem_status": source_statuses.get("pubchem", "not_run"),
            "chembl_status": source_statuses.get("chembl", "not_run"),
            "surechembl_query_status": (
                best_status(sure_rows["lookup_status"])
                if not sure_rows.empty and "lookup_status" in sure_rows.columns
                else "not_run"
            ),
        }
        colors = [
            molecule_status_color(row[column])
            for column in (
                "pubchem_status",
                "chembl_status",
                "surechembl_query_status",
            )
        ]
        row["status_color"] = (
            "green"
            if "green" in colors
            else "yellow"
            if "yellow" in colors
            else "red"
            if "red" in colors
            else "gray"
        )
        row["evidence_status"] = {
            "green": "Available",
            "yellow": "Partial",
            "red": "Missing",
            "gray": "Not run",
        }[row["status_color"]]
        row["molecule_position"] = len(rows) + 1
        rows.append(row)
    return pd.DataFrame(rows)


def rdkit_molecule_dataframe(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Prepare molecule-level RDKit scatter data."""
    if descriptors.empty or "molecule_id" not in descriptors.columns:
        return pd.DataFrame()
    result = coerce_numeric(
        descriptors,
        (
            "molecular_weight",
            "logp",
            "tpsa",
            "hbd",
            "hba",
            "rotatable_bonds",
            "qed",
            "druglikeness_score",
        ),
    )
    if "druglikeness_category" not in result.columns:
        result["druglikeness_category"] = result.get(
            "valid_smiles", pd.Series(False, index=result.index)
        ).map(lambda value: "favorable" if str(value).lower() == "true" else "invalid")
    result["status_color"] = result["druglikeness_category"].map(molecule_status_color)
    return result


def text_evidence_molecule_dataframe(
    text_nlp: pd.DataFrame,
    molecule_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Summarize text evidence to one interactive row per molecule."""
    ids = {str(value) for value in molecule_ids if str(value).strip()}
    if "molecule_id" in text_nlp.columns:
        ids.update(text_nlp["molecule_id"].dropna().astype(str))
    rows = []
    for position, molecule_id in enumerate(sorted(ids), start=1):
        matches = (
            text_nlp[text_nlp["molecule_id"].astype(str) == molecule_id].copy()
            if not text_nlp.empty and "molecule_id" in text_nlp.columns
            else pd.DataFrame()
        )
        status = (
            best_status(matches["nlp_status"])
            if not matches.empty and "nlp_status" in matches.columns
            else "not_run"
        )
        score_column_name = next(
            (
                column
                for column in ("max_relevance_score", "similarity_score")
                if column in matches.columns
            ),
            "",
        )
        scores = (
            pd.to_numeric(matches[score_column_name], errors="coerce")
            if score_column_name
            else pd.Series(dtype="float64")
        )
        top_index = scores.idxmax() if not scores.dropna().empty else None
        rows.append(
            {
                "molecule_id": molecule_id,
                "nlp_status": status,
                "max_relevance_score": (
                    float(scores.loc[top_index]) if top_index is not None else None
                ),
                "top_evidence_title": (
                    str(matches.loc[top_index, "title"])
                    if top_index is not None and "title" in matches.columns
                    else ""
                ),
                "evidence_matches": int(len(matches)),
                "molecule_position": position,
                "status_color": molecule_status_color(status),
            }
        )
    return pd.DataFrame(rows)


def biomedical_evidence_molecule_dataframe(
    biomedical: pd.DataFrame,
    molecule_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare molecule-level biomedical evidence rows for Step 6."""
    ids = {str(value) for value in molecule_ids if str(value).strip()}
    if "molecule_id" in biomedical.columns:
        ids.update(biomedical["molecule_id"].dropna().astype(str))
    rows = []
    for position, molecule_id in enumerate(sorted(ids), start=1):
        matches = (
            biomedical[biomedical["molecule_id"].astype(str) == molecule_id].copy()
            if not biomedical.empty and "molecule_id" in biomedical.columns
            else pd.DataFrame()
        )
        selected = matches.iloc[0].to_dict() if not matches.empty else {}
        model_status = normalize_status(
            selected.get("biomedical_model_status", "not_run")
        )
        evidence_status = normalize_status(
            selected.get("biomedical_evidence_status", "not_run")
        )
        rows.append(
            {
                "molecule_id": molecule_id,
                "biomedical_model_status": model_status,
                "biomedical_evidence_status": evidence_status,
                "biomedical_similarity_score": selected.get(
                    "biomedical_similarity_score", "0.000"
                ),
                "biomedical_relevance_category": selected.get(
                    "biomedical_relevance_category", "not_run"
                ),
                "biomedical_evidence_count": selected.get(
                    "biomedical_evidence_count", "0"
                ),
                "top_biomedical_evidence_id": selected.get(
                    "top_biomedical_evidence_id", ""
                ),
                "top_biomedical_evidence_text": selected.get(
                    "top_biomedical_evidence_text", ""
                ),
                "molecule_position": position,
                "status_color": molecule_status_color(evidence_status),
            }
        )
    return pd.DataFrame(rows)


def patent_evidence_molecule_dataframe(
    patent: pd.DataFrame,
    molecule_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Prepare molecule-level patent/IP-context evidence rows for Step 7."""
    ids = {str(value) for value in molecule_ids if str(value).strip()}
    if "molecule_id" in patent.columns:
        ids.update(patent["molecule_id"].dropna().astype(str))
    rows = []
    for position, molecule_id in enumerate(sorted(ids), start=1):
        matches = (
            patent[patent["molecule_id"].astype(str) == molecule_id].copy()
            if not patent.empty and "molecule_id" in patent.columns
            else pd.DataFrame()
        )
        selected = matches.iloc[0].to_dict() if not matches.empty else {}
        rows.append(
            {
                "molecule_id": molecule_id,
                "surechembl_structure_status": normalize_status(
                    selected.get("surechembl_structure_status", "not_run")
                ),
                "patent_document_metadata_status": normalize_status(
                    selected.get("patent_document_metadata_status", "not_run")
                ),
                "patent_model_status": normalize_status(
                    selected.get("patent_model_status", "not_run")
                ),
                "patent_evidence_status": normalize_status(
                    selected.get("patent_evidence_status", "not_run")
                ),
                "patent_similarity_score": selected.get(
                    "patent_similarity_score", "0.000"
                ),
                "patent_relevance_category": selected.get(
                    "patent_relevance_category", "not_run"
                ),
                "patent_evidence_count": selected.get("patent_evidence_count", "0"),
                "top_patent_evidence_id": selected.get(
                    "top_patent_evidence_id", ""
                ),
                "top_patent_evidence_text": selected.get(
                    "top_patent_evidence_text", ""
                ),
                "molecule_position": position,
                "status_color": molecule_status_color(
                    selected.get("patent_evidence_status", "not_run")
                ),
            }
        )
    return pd.DataFrame(rows)


def final_priority_molecule_dataframe(prioritization: pd.DataFrame) -> pd.DataFrame:
    """Prepare final score-versus-similarity data for each molecule."""
    if prioritization.empty or "molecule_id" not in prioritization.columns:
        return pd.DataFrame()
    result = coerce_numeric(
        prioritization,
        ("prioritization_score_with_nlp", "prioritization_score", "tanimoto_similarity"),
    )
    category = next(
        (
            column
            for column in (
                "prioritization_category_with_nlp",
                "prioritization_category",
            )
            if column in result.columns
        ),
        "",
    )
    if not category:
        category = "design_category"
        result[category] = "not available"
    result["design_category"] = result[category].fillna("not available").astype(str)
    result["status_color"] = result["design_category"].map(molecule_status_color)
    return result


def selected_molecule_from_plot_event(event: object) -> str:
    """Extract a molecule ID from a Streamlit Plotly selection event."""
    if event is None:
        return ""
    selection = (
        event.get("selection", {})
        if isinstance(event, dict)
        else getattr(event, "selection", {})
    )
    points = (
        selection.get("points", [])
        if isinstance(selection, dict)
        else getattr(selection, "points", [])
    )
    if not points:
        return ""
    point = points[0]
    custom = point.get("customdata", []) if isinstance(point, dict) else []
    return str(custom[0]) if custom else ""


def status_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return value counts for a status column."""
    if column not in df.columns:
        return {}
    counts = df[column].fillna("not_available").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def normalize_status(value: object) -> str:
    """Normalize display status values."""
    text = str(value or "").strip()
    return text or "not_available"


def readable_status(value: object) -> str:
    """Convert a machine-readable workflow status to concise UI text."""
    status = normalize_status(value)
    labels = {
        "match_found": "Match found",
        "hit": "Match found",
        "no_match": "No match",
        "no_evidence": "No evidence",
        "lookup_error": "Lookup error",
        "error": "Error",
        "not_queried": "Not queried",
        "not_run": "Not run",
        "skipped": "Skipped",
        "model_unavailable": "Model unavailable",
        "not_available": "Not available",
        "available": "Available",
        "offline": "Not run",
        "invalid_molecule": "Invalid molecule",
        "structure_match_only": "Structure match only",
        "very_close_patent_analog": "Very close SureChEMBL structure match",
        "related_patent_chemotype": "Related SureChEMBL structure match",
        "moderate_patent_similarity": "Moderate SureChEMBL structure match",
        "structurally_distinct_from_patent_compound": (
            "Structurally distinct SureChEMBL match"
        ),
        "SureChEMBL API": "SureChEMBL structure lookup",
    }
    return labels.get(status, status.replace("_", " ").strip().capitalize())


def readable_ui_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with machine-readable status/category values humanized."""
    result = frame.copy()
    value_columns = {
        column
        for column in result.columns
        if column.endswith("_status")
        or column
        in {
            "identity_status",
            "lookup_status",
            "context_status",
            "nlp_status",
            "nlp_relevance_category",
            "biomedical_model_status",
            "biomedical_evidence_status",
            "biomedical_relevance_category",
            "patent_model_status",
            "patent_evidence_status",
            "patent_relevance_category",
            "surechembl_structure_status",
            "patent_document_metadata_status",
            "druglikeness_category",
            "prioritization_category",
            "prioritization_category_with_nlp",
            "design_category",
        }
    }
    for column in value_columns:
        result[column] = result[column].map(readable_status)
    return result


def compact_detail_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop detail fields whose selected-molecule value is empty or unavailable."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    keep = []
    for column in result.columns:
        value = result.iloc[0][column]
        text = "" if pd.isna(value) else str(value).strip()
        if text.lower() not in {"", "nan", "none", "null", "not_available"}:
            keep.append(column)
    return result[keep]


def status_display(value: object) -> str:
    """Return a compact styled status label."""
    status = normalize_status(value)
    icon = STATUS_ICONS.get(status, "")
    return f"{icon} {readable_status(status)}".strip()


def status_meaning(value: object) -> str:
    """Return a human-readable status meaning."""
    return STATUS_MEANINGS.get(normalize_status(value), "not available")


def evidence_completeness_rows(record: dict[str, object]) -> pd.DataFrame:
    """Build table-ready evidence completeness rows."""
    rows = []
    for column, label in STATUS_LABELS.items():
        status = normalize_status(record.get(column, "not_available"))
        rows.append(
            {
                "Evidence source": label,
                "Status": status_display(status),
                "Meaning": status_meaning(status),
            }
        )
    return pd.DataFrame(rows)


def reference_similarity_interpretation(value: object) -> str:
    """Interpret uploaded-reference Tanimoto similarity for UI display."""
    score = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(score):
        return "Reference similarity is not available."
    if float(score) < 0.30:
        return "Weak RDKit fingerprint similarity to the uploaded reference panel."
    if float(score) < 0.50:
        return "Moderate RDKit fingerprint similarity to the uploaded reference panel."
    return "Strong RDKit fingerprint similarity to the uploaded reference panel."


def molecule_detail_rows(record: dict[str, object], active_score: str) -> pd.DataFrame:
    """Build table-ready selected molecule details."""
    similarity = record.get("tanimoto_similarity", "")
    rows = [
        (display_label(active_score), record.get(active_score, "")),
        (display_label("best_reference_name"), record.get("best_reference_name", "")),
        (display_label("tanimoto_similarity"), similarity),
        ("Reference similarity interpretation", reference_similarity_interpretation(similarity)),
        (display_label("known_public_match"), record.get("known_public_match", "")),
        (display_label("pubchem_status"), status_display(record.get("pubchem_status", ""))),
        (display_label("chembl_status"), status_display(record.get("chembl_status", ""))),
        (display_label("surechembl_query_status"), status_display(record.get("surechembl_query_status", ""))),
        (display_label("chemberta_status"), status_display(record.get("chemberta_status", ""))),
        (display_label("nlp_status"), status_display(record.get("nlp_status", ""))),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"]).astype("string")


def molecule_image_message(image_path: Path) -> str:
    """Return the message shown when a molecule image is missing."""
    if RDKIT_DRAWING_UNAVAILABLE:
        return "2D structure image is unavailable in this environment."
    if image_path.exists():
        return ""
    return "2D structure image is not available for this molecule."


def molecule_structure_image(smiles: object) -> object | None:
    """Create an in-memory 2D structure image for a valid SMILES value."""
    global RDKIT_DRAWING_UNAVAILABLE

    RDKIT_DRAWING_UNAVAILABLE = False
    try:
        from rdkit.Chem import Draw, rdDepictor
    except ImportError:
        RDKIT_DRAWING_UNAVAILABLE = True
        return None

    text = str(smiles or "").strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    rdDepictor.Compute2DCoords(molecule)
    return Draw.MolToImage(molecule, size=(420, 320))


def molecule_smiles_from_outputs(loaded: LoadedOutputs, molecule_id: str) -> str:
    """Return the best available standardized SMILES for a selected molecule."""
    source_order = (
        ("standardized", ("canonical_smiles", "smiles")),
        ("descriptors", ("canonical_smiles",)),
        ("chemical_identity", ("smiles",)),
        ("prioritization", ("canonical_smiles", "smiles")),
    )
    for table_name, columns in source_order:
        frame = loaded.tables[table_name]
        if frame.empty or "molecule_id" not in frame.columns:
            continue
        row = frame[frame["molecule_id"].astype(str) == molecule_id]
        if row.empty:
            continue
        for column in columns:
            if column in row.columns:
                value = str(row.iloc[0][column] or "").strip()
                if value and value.lower() != "nan":
                    return value
    return ""


def generate_report(
    loaded: LoadedOutputs,
    molecule_id: str,
) -> Path:
    """Generate one local Markdown report without online API calls."""
    loaded.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = loaded.reports_dir / f"compound_intelligence_report_{molecule_id}.md"
    compound_qa(
        molecule_id,
        "full_report",
        output_path,
        prioritized_path=loaded.paths["prioritization"],
        similarity_path=loaded.output_dir / "similarity_top_hits.csv",
        public_lookup_path=loaded.paths["public_lookup"],
        nlp_path=loaded.paths["prioritization"].with_name("text_nlp.csv"),
        descriptor_path=loaded.paths["descriptors"],
        surechembl_path=loaded.paths["surechembl"],
        visualization_path=loaded.paths["visualization"],
        image_dir=loaded.images_dir,
    )
    return output_path


def existing_report_path(loaded: LoadedOutputs, molecule_id: str) -> Path:
    """Return the expected report path for a molecule."""
    return loaded.reports_dir / f"compound_intelligence_report_{molecule_id}.md"


def report_status_message(report_path: Path) -> tuple[str, str]:
    """Return concise report status text without exposing a filesystem path."""
    if report_path.exists():
        return "success", "A molecule report is available for this candidate."
    return "info", "No molecule report has been generated yet."


def summarize_run(
    df: pd.DataFrame,
    *,
    public_lookup_exists: bool | None = None,
) -> dict[str, int | str]:
    """Return top-line summary metrics without implying an unrun lookup found zero hits."""
    valid_count = (
        int(df["valid_smiles"].astype(str).str.lower().eq("true").sum())
        if "valid_smiles" in df.columns
        else 0
    )
    public_hits: int | str = "Not run"
    if public_lookup_exists is not False and "known_public_match" in df.columns:
        lookup_statuses = set()
        for column in ("pubchem_status", "chembl_status"):
            if column in df.columns:
                lookup_statuses.update(
                    df[column].fillna("not_available").astype(str).str.lower()
                )
        if (
            public_lookup_exists is True
            or (public_lookup_exists is None and not lookup_statuses)
            or lookup_statuses.difference({"not_run", "offline", "not_available"})
        ):
            public_hits = int(
                df["known_public_match"].astype(str).str.lower().eq("true").sum()
            )
    chemberta_available = (
        int(df["chemberta_status"].astype(str).eq("available").sum())
        if "chemberta_status" in df.columns
        else 0
    )
    return {
        "Total": int(len(df)),
        "Valid": valid_count,
        "Exact public matches": public_hits,
        "ChemBERTa": chemberta_available,
    }


def count_status_bucket(counts: dict[str, int], statuses: tuple[str, ...]) -> int:
    """Sum equivalent status values into a display bucket."""
    return sum(counts.get(status, 0) for status in statuses)


def build_external_public_evidence_table(
    df: pd.DataFrame,
    *,
    public_lookup_exists: bool | None = None,
    surechembl_exists: bool | None = None,
) -> pd.DataFrame:
    """Build external public evidence status counts."""
    rows = []
    sources = (
        ("PubChem", "pubchem_status", public_lookup_exists),
        ("ChEMBL", "chembl_status", public_lookup_exists),
        ("SureChEMBL", "surechembl_query_status", surechembl_exists),
    )
    for source, column, file_exists in sources:
        source_not_run = file_exists is False or column not in df.columns
        counts = {} if source_not_run else status_counts(df, column)
        rows.append(
            {
                "Source": source,
                "Hit": count_status_bucket(counts, ("hit", "match_found")),
                "No match": count_status_bucket(counts, ("no_match",)),
                "Not queried": count_status_bucket(counts, ("not_queried",)),
                "Not run": (
                    int(len(df))
                    if source_not_run
                    else count_status_bucket(
                        counts, ("not_run", "offline", "not_available")
                    )
                ),
                "Error": count_status_bucket(counts, ("lookup_error", "error")),
            }
        )
    return pd.DataFrame(rows)


def render_design_foundation_css() -> None:
    """Install a small, reusable visual foundation for Streamlit pages."""
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e8e4;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
        }
        .ui-card {
            background: #ffffff;
            border: 1px solid #d9e8e4;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.55rem 0 0.85rem;
        }
        .ui-card--subtle {
            background: #f4faf8;
        }
        .ui-card--warning {
            background: #fff8e6;
            border-color: #f0d38a;
        }
        .ui-card__title {
            color: #102a43;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .ui-card__body {
            color: #334e68;
            line-height: 1.5;
        }
        .ui-hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #eef8f6 54%, #e8f2fb 100%);
            border: 1px solid #cfe1dc;
            border-radius: 12px;
            padding: 1.35rem 1.45rem;
            margin: 0.75rem 0 1.1rem;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08);
        }
        .ui-hero-card__title {
            color: #102a43;
            font-size: 1.65rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .ui-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.85rem;
            margin: 0.75rem 0 1rem;
        }
        .ui-step-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.35rem 0 1rem;
        }
        .ui-step-chip {
            background: #ffffff;
            border: 1px solid #cfe1dc;
            border-radius: 999px;
            color: #264653;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.35rem 0.6rem;
            white-space: nowrap;
        }
        .ui-status-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid #c9d8d5;
            background: #eef6f4;
            color: #264653;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.28rem 0.55rem;
            margin: 0.1rem 0.2rem 0.1rem 0;
            white-space: nowrap;
        }
        .ui-status-badge--success {
            background: #e7f5ee;
            border-color: #a9d9c0;
            color: #146c43;
        }
        .ui-status-badge--warning {
            background: #fff3cd;
            border-color: #f0d38a;
            color: #7a5200;
        }
        .ui-status-badge--error {
            background: #fdecec;
            border-color: #f0b4b4;
            color: #9f1c1c;
        }
        .ui-section-note {
            color: #52616f;
            margin-top: -0.35rem;
            margin-bottom: 0.75rem;
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border: 1px solid #d9e8e4;
            border-radius: 8px;
            padding: 0.35rem;
            background: #ffffff;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]),
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTable"]) {
            margin-top: 0.35rem;
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ui_text(value: object) -> str:
    """Escape user-visible text before placing it in small HTML cards."""
    return html.escape(str(value or ""), quote=True)


def render_metric_card(
    label: str,
    value: object,
    *,
    help_text: str | None = None,
    container: object | None = None,
) -> None:
    """Render a native Streamlit metric styled by the shared visual foundation."""
    target = container or st
    target.metric(label, value)
    if help_text:
        target.caption(help_text)


def render_status_badge(
    label: str,
    *,
    status: str = "neutral",
    container: object | None = None,
) -> None:
    """Render a compact status badge without replacing Streamlit warnings/errors."""
    target = container or st
    status_class = slugify_key(status) or "neutral"
    target.markdown(
        f'<span class="ui-status-badge ui-status-badge--{status_class}">{_ui_text(label)}</span>',
        unsafe_allow_html=True,
    )
def render_info_card(
    title: str,
    body: str,
    *,
    container: object | None = None,
) -> None:
    """Render a subtle informational card for passive context."""
    target = container or st
    if hasattr(target, "markdown"):
        target.markdown(
            '<div class="ui-card ui-card--subtle">'
            f'<div class="ui-card__title">{_ui_text(title)}</div>'
            f'<div class="ui-card__body">{_ui_text(body)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif hasattr(target, "caption"):
        target.caption(f"{title}: {body}")

def render_warning_card(
    title: str,
    body: str,
    *,
    container: object | None = None,
) -> None:
    """Render a non-blocking warning card while leaving st.warning available."""
    target = container or st
    if hasattr(target, "markdown"):
        target.markdown(
            '<div class="ui-card ui-card--warning">'
            f'<div class="ui-card__title">{_ui_text(title)}</div>'
            f'<div class="ui-card__body">{_ui_text(body)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif hasattr(target, "caption"):
        target.caption(f"{title}: {body}")
def render_section_header(
    title: str,
    body: str | None = None,
    *,
    level: int = 2,
) -> None:
    """Render a consistent section heading with optional supporting text."""
    if level <= 1:
        st.header(title)
    elif level == 2:
        st.subheader(title)
    else:
        st.markdown(f"{'#' * min(level, 6)} {title}")
    if body:
        if hasattr(st, "caption"):
            st.caption(body)
        else:
            st.markdown(body)


def render_step_summary_card(
    title: str,
    steps: Iterable[str],
    *,
    empty_text: str = "None",
) -> None:
    """Render compact lists used by step summaries and path-status panels."""
    items = list(steps)
    body = empty_text if not items else "<br>".join(_ui_text(step) for step in items)
    st.markdown(
        '<div class="ui-card">'
        f'<div class="ui-card__title">{_ui_text(title)}</div>'
        f'<div class="ui-card__body">{body}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
def shadcn_ui_available() -> bool:
    """Return whether shadcn components can safely render in this Streamlit run."""
    return shadcn_ui is not None and getattr(st, "__name__", "") == "streamlit"

def render_modern_badge(
    label: str,
    *,
    variant: str = "secondary",
    status: str = "neutral",
    key: str | None = None,
) -> None:
    """Render a shadcn badge with a native fallback."""
    if shadcn_ui_available():
        shadcn_ui.badges([(label, variant)], key=key)
    render_status_badge(label, status=status)

def render_modern_card(
    title: str,
    body: str,
    *,
    description: str | None = None,
    badges: Iterable[tuple[str, str]] = (),
    key: str | None = None,
) -> None:
    """Render a shadcn card with a native card fallback companion."""
    badge_list = list(badges)
    if shadcn_ui_available():
        shadcn_ui.card(
            title=title,
            content=body,
            description=description,
            key=key,
        )
        if badge_list:
            shadcn_ui.badges(badge_list, key=f"{key}_badges" if key else None)
    render_info_card(title, body if description is None else f"{description} {body}")
    for badge, variant in badge_list:
        render_status_badge(
            badge,
            status="success" if variant == "default" else "neutral",
        )

def render_modern_metric_card(
    title: str,
    value: object,
    *,
    description: str | None = None,
    key: str | None = None,
    container: object | None = None,
) -> None:
    """Render a shadcn metric card with native metric fallback."""
    target = container or st
    if shadcn_ui_available() and container is None:
        shadcn_ui.metric_card(
            title=title,
            content=str(value),
            description=description,
            key=key,
        )
    render_metric_card(title, value, help_text=description, container=target)
def render_modern_step_card(
    title: str,
    steps: Iterable[Path | str],
    *,
    empty_text: str = "None",
    key: str | None = None,
) -> None:
    """Render a compact artifact/step list card."""
    values = [Path(step).name if isinstance(step, Path) else str(step) for step in steps]
    body = empty_text if not values else " | ".join(values)
    render_modern_card(title, body, key=key)


def render_modern_evidence_card(
    title: str,
    rows: dict[str, object],
    *,
    key: str | None = None,
) -> None:
    """Render evidence summary values in modern card form."""
    body = " | ".join(f"{label}: {value}" for label, value in rows.items())
    render_modern_card(title, body or "No evidence summary available.", key=key)


def render_timeline_chips(items: Iterable[str]) -> None:
    """Render compact workflow timeline chips."""
    chips = "".join(f'<span class="ui-step-chip">{_ui_text(item)}</span>' for item in items)
    st.markdown(
        f'<div class="ui-step-chip-row">{chips}</div>',
        unsafe_allow_html=True,
    )

def build_computed_analysis_status_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build computed-analysis status counts."""
    rows = []
    sources = (
        ("ChemBERTa", "chemberta_status"),
        ("Text-evidence matching", "nlp_status"),
        ("RDKit descriptors", "descriptor_status"),
    )
    for source, column in sources:
        if column == "descriptor_status":
            counts = {
                "available": int(len(df)) if "molecular_weight" in df.columns or "valid_smiles" in df.columns else 0,
                "not_available": 0 if ("molecular_weight" in df.columns or "valid_smiles" in df.columns) else int(len(df)),
            }
        else:
            counts = status_counts(df, column)
        rows.append(
            {
                "Source": source,
                "Available": count_status_bucket(counts, ("available", "match_found")),
                "Not run": count_status_bucket(counts, ("not_run", "offline")),
                "Not available": count_status_bucket(
                    counts,
                    (
                        "no_match",
                        "not_available",
                        "not_queried",
                        "invalid_molecule",
                    ),
                ),
                "Error": count_status_bucket(counts, ("lookup_error", "error")),
            }
        )
    return pd.DataFrame(rows)

def render_summary_cards(loaded: LoadedOutputs) -> None:
    """Render high-level run summary metrics."""
    df = loaded.tables["prioritization"]
    summary = summarize_run(
        df,
        public_lookup_exists=loaded.paths["public_lookup"].exists(),
    )
    metrics = st.columns(4)
    for index, (label, value) in enumerate(summary.items()):
        render_metric_card(label, value, container=metrics[index])

    render_section_header(
        "External Public Evidence",
        "Not queried means the molecule was not checked because of the molecule limit or workflow settings.",
        level=4,
    )
    st.table(
        build_external_public_evidence_table(
            df,
            public_lookup_exists=loaded.paths["public_lookup"].exists(),
            surechembl_exists=loaded.paths["surechembl"].exists(),
        )
    )
    render_section_header(
        "Computed Analysis Status",
        "ChemBERTa availability means molecular embeddings were generated; it is not a public database match.",
        level=4,
    )
    st.info(nlp_output_note(loaded.paths["text_nlp"], loaded.tables["text_nlp"]))
    st.caption("ChemBERTa is unavailable for invalid SMILES.")
    st.table(build_computed_analysis_status_table(df))

def chemical_space_dataframe(
    coords: pd.DataFrame,
    prioritization: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare coordinates with ranking metadata only when it exists."""
    if (
        not prioritization.empty
        and "molecule_id" in prioritization.columns
        and "molecule_id" in coords.columns
    ):
        plot_df = coords.merge(
            prioritization,
            on="molecule_id",
            how="left",
            suffixes=("", "_prioritized"),
        )
    else:
        plot_df = coords.copy()
    return coerce_numeric(
        plot_df,
        [
            "x",
            "y",
            "prioritization_score_with_nlp",
            "prioritization_score",
            "tanimoto_similarity",
            "nearest_reference_similarity",
        ],
    )


def chemical_space_summary_values(plot_df: pd.DataFrame) -> dict[str, object]:
    """Return compact generated/reference chemical-space summary metrics."""
    source = (
        plot_df["source_type"].fillna("generated").astype(str).str.strip()
        if "source_type" in plot_df.columns
        else pd.Series(["generated"] * len(plot_df), index=plot_df.index)
    )
    generated = plot_df[source.eq("generated")]
    references = plot_df[source.eq("reference")]
    similarities = pd.to_numeric(
        generated.get("nearest_reference_similarity", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    median = f"{similarities.median():.3f}" if not similarities.empty else "Not available"
    high = int((similarities >= 0.70).sum()) if not similarities.empty else 0
    far = int((similarities < 0.40).sum()) if not similarities.empty else 0
    return {
        "Generated plotted": int(len(generated)),
        "References plotted": int(len(references)),
        "Median nearest-reference similarity": median,
        "Generated high similarity": high,
        "Generated far from references": far,
    }


def render_chemical_space_summary(plot_df: pd.DataFrame) -> None:
    """Render compact chemical-space summary metrics."""
    values = chemical_space_summary_values(plot_df)
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values.items()):
        column.metric(label, value)


def nearest_reference_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return generated molecule nearest-reference rows for optional preview."""
    if "source_type" not in plot_df.columns:
        return pd.DataFrame()
    generated = plot_df[plot_df["source_type"].fillna("").astype(str) == "generated"]
    columns = available_columns(
        generated,
        (
            "molecule_id",
            "nearest_reference_id",
            "nearest_reference_name",
            "nearest_reference_similarity",
            "nearest_reference_interpretation",
            "best_reference_name",
            "tanimoto_similarity",
        ),
    )
    return display_dataframe(readable_ui_dataframe(generated[columns])) if columns else pd.DataFrame()


def nearest_similarity_distribution(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return compact nearest-reference similarity category counts."""
    if "source_type" not in plot_df.columns:
        return pd.DataFrame(columns=["Category", "Generated molecules"])
    generated = plot_df[plot_df["source_type"].fillna("").astype(str) == "generated"]
    values = pd.to_numeric(
        generated.get("nearest_reference_similarity", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    return pd.DataFrame(
        {
            "Category": ["High similarity", "Moderate similarity", "Low similarity"],
            "Generated molecules": [
                int((values >= 0.70).sum()),
                int(((values >= 0.40) & (values < 0.70)).sum()),
                int((values < 0.40).sum()),
            ],
        }
    )


def chemical_space_cluster_column(plot_df: pd.DataFrame) -> pd.Series:
    """Return display cluster labels while keeping reference points visible."""
    if "cluster_id" not in plot_df.columns:
        return pd.Series(["not_clustered"] * len(plot_df), index=plot_df.index)
    source = (
        plot_df["source_type"].fillna("").astype(str)
        if "source_type" in plot_df.columns
        else pd.Series(["generated"] * len(plot_df), index=plot_df.index)
    )
    clusters = plot_df["cluster_id"].fillna("").astype(str).str.strip()
    clusters = clusters.where(clusters.ne(""), "not_clustered")
    return clusters.where(~source.eq("reference") | clusters.ne("not_clustered"), "reference")


def chemical_space_cluster_summary(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return generated/reference counts per chemical-space cluster."""
    if plot_df.empty:
        return pd.DataFrame(columns=["Cluster", "Generated molecules", "Reference molecules"])
    frame = plot_df.copy()
    frame["cluster_display"] = chemical_space_cluster_column(frame)
    source = (
        frame["source_type"].fillna("generated").astype(str)
        if "source_type" in frame.columns
        else pd.Series(["generated"] * len(frame), index=frame.index)
    )
    rows = []
    for cluster in sorted(frame["cluster_display"].dropna().astype(str).unique()):
        members = frame[frame["cluster_display"].astype(str) == cluster]
        member_source = source.loc[members.index]
        rows.append(
            {
                "Cluster": cluster,
                "Generated molecules": int(member_source.eq("generated").sum()),
                "Reference molecules": int(member_source.eq("reference").sum()),
            }
        )
    return pd.DataFrame(rows)


def chemical_space_color_options(plot_df: pd.DataFrame) -> dict[str, str]:
    """Return available chemical-space color modes mapped to dataframe columns."""
    options = {"Source type": "source_type", "Chemical-space cluster": "cluster_display"}
    for label, candidates in (
        (
            "Priority category",
            (
                "prioritization_category_with_nlp",
                "prioritization_category",
                "design_category",
            ),
        ),
        ("Public identity status", ("known_public_match", "novelty_flag")),
    ):
        for column in candidates:
            if column in plot_df.columns:
                options[label] = column
                break
    return options


def nearest_reference_link_rows(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Select generated molecules for optional nearest-reference link overlays."""
    if "source_type" not in plot_df.columns:
        return pd.DataFrame()
    generated = plot_df[plot_df["source_type"].fillna("").astype(str) == "generated"].copy()
    references = plot_df[plot_df["source_type"].fillna("").astype(str) == "reference"].copy()
    if generated.empty or references.empty:
        return pd.DataFrame()
    reference_lookup = {}
    for _, reference in references.iterrows():
        for column in ("reference_id", "molecule_id", "reference_name"):
            value = str(reference.get(column, "")).strip()
            if value:
                reference_lookup[value] = reference
    generated["link_score"] = pd.to_numeric(
        generated.get("prioritization_score_with_nlp", pd.Series(dtype=float)),
        errors="coerce",
    )
    if generated["link_score"].notna().any():
        generated = generated.sort_values("link_score", ascending=False)
    else:
        generated["link_score"] = pd.to_numeric(
            generated.get("nearest_reference_similarity", pd.Series(dtype=float)),
            errors="coerce",
        )
        generated = generated.sort_values("link_score", ascending=False)
    rows = []
    for _, row in generated.head(10).iterrows():
        reference = reference_lookup.get(str(row.get("nearest_reference_id", "")).strip())
        if reference is None:
            reference = reference_lookup.get(str(row.get("nearest_reference_name", "")).strip())
        if reference is None:
            continue
        rows.append(
            {
                "generated_x": row.get("x"),
                "generated_y": row.get("y"),
                "reference_x": reference.get("x"),
                "reference_y": reference.get("y"),
            }
        )
    return pd.DataFrame(rows)


def chemical_space_hover_columns(plot_df: pd.DataFrame) -> list[str]:
    """Return hover columns for generated/reference chemical-space points."""
    return [
        column
        for column in (
            "molecule_id",
            "source_type",
            "reference_name",
            "reference_role",
            "target",
            "canonical_smiles",
            "nearest_reference_id",
            "nearest_reference_name",
            "nearest_reference_similarity",
            "nearest_reference_interpretation",
            "best_reference_name",
            "tanimoto_similarity",
            "cluster_id",
            "druglikeness_category",
            "novelty_flag",
        )
        if column in plot_df.columns
    ]


def build_chemical_space_figure(
    plot_df: pd.DataFrame,
    *,
    color_by: str = "Source type",
    show_links: bool = False,
) -> go.Figure:
    """Build the generated/reference chemical-space scatter plot."""
    figure_df = plot_df.copy()
    if "source_type" not in figure_df.columns:
        figure_df["source_type"] = "generated"
    figure_df["source_type"] = figure_df["source_type"].fillna("generated").astype(str)
    figure_df["marker_size"] = figure_df["source_type"].map(
        {"generated": 8, "reference": 12}
    ).fillna(8)
    if "cluster_id" in figure_df.columns:
        figure_df["cluster_id"] = figure_df["cluster_id"].fillna("not assigned").astype(str)
    figure_df["cluster_display"] = chemical_space_cluster_column(figure_df)
    color_options = chemical_space_color_options(figure_df)
    color_column = color_options.get(color_by, "source_type")
    color_map = (
        {
            "generated": "#1f77b4",
            "reference": "#d95f02",
        }
        if color_column == "source_type"
        else None
    )
    fig = px.scatter(
        figure_df,
        x="x",
        y="y",
        color=color_column,
        symbol="source_type",
        symbol_map={"generated": "circle", "reference": "diamond"},
        size="marker_size",
        size_max=13,
        hover_name="molecule_id",
        hover_data=chemical_space_hover_columns(figure_df),
        custom_data=["molecule_id"],
        labels=display_labels(figure_df.columns),
        title="Generated and Reference Molecules in Chemical Space",
        color_discrete_map=color_map,
    )
    fig.update_traces(marker={"opacity": 0.88, "line": {"width": 0.7, "color": "white"}})
    if show_links:
        for _, link in nearest_reference_link_rows(figure_df).iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[link["generated_x"], link["reference_x"]],
                    y=[link["generated_y"], link["reference_y"]],
                    mode="lines",
                    line={"color": "rgba(120, 120, 120, 0.35)", "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return fig


def category_color_map(frame: pd.DataFrame, column: str) -> dict[str, str]:
    """Return Plotly colors for each readable status/category value."""
    if column not in frame.columns:
        return {}
    return {
        str(value): MOLECULE_STATUS_COLORS[molecule_status_color(value)]
        for value in frame[column].dropna().astype(str).unique()
    }


def render_molecule_selector(
    frame: pd.DataFrame,
    *,
    key: str,
    plot_event: object = None,
) -> str:
    """Render a molecule selector synchronized with a Plotly point selection."""
    if frame.empty or "molecule_id" not in frame.columns:
        return ""
    molecule_ids = frame["molecule_id"].dropna().astype(str).drop_duplicates().tolist()
    if not molecule_ids:
        return ""
    clicked = selected_molecule_from_plot_event(plot_event)
    remembered = str(st.session_state.get(f"{key}_selected", ""))
    default = clicked if clicked in molecule_ids else remembered
    if default not in molecule_ids:
        default = molecule_ids[0]
    selected = st.selectbox(
        "Select molecule to inspect",
        molecule_ids,
        index=molecule_ids.index(default),
        key=f"{key}_selector",
    )
    active = clicked if clicked in molecule_ids else str(selected)
    st.session_state[f"{key}_selected"] = active
    return active


def render_validation_view(frame: pd.DataFrame, *, key: str) -> str:
    """Render molecule-level validation badges, points, and selection."""
    plot_df = validation_molecule_dataframe(frame)
    if plot_df.empty:
        st.info("SMILES validation output is unavailable.")
        return ""
    st.markdown("#### Valid and invalid SMILES")
    table_columns = available_columns(
        plot_df,
        ("molecule_id", "validation_status", "smiles", "canonical_smiles", "error_message"),
    )
    with st.expander("Preview table", expanded=False):
        st.dataframe(
            display_dataframe(readable_ui_dataframe(plot_df[table_columns])),
            width="stretch",
            hide_index=True,
            height=260,
        )
    figure = px.scatter(
        plot_df,
        x="molecule_position",
        y="validation_status",
        color="validation_status",
        hover_name="molecule_id",
        hover_data=available_columns(
            plot_df, ("smiles", "canonical_smiles", "error_message")
        ),
        custom_data=["molecule_id"],
        color_discrete_map={"Valid": MOLECULE_STATUS_COLORS["green"], "Invalid": MOLECULE_STATUS_COLORS["red"]},
        labels={
            "molecule_position": "Molecule",
            "validation_status": "Validation status",
        },
        title="SMILES validation by molecule",
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def chemical_identity_figure(plot_df: pd.DataFrame) -> object:
    """Build the Chemical Identity Plotly figure from mapped molecule rows."""
    figure = px.scatter(
        plot_df,
        x="molecule_position",
        y="display_status",
        color="display_status",
        hover_name="molecule_id",
        hover_data=available_columns(
            plot_df,
            (
                "exact_public_name",
                "iupac_name",
                "pubchem_cid",
                "lookup_status",
                "identity_confidence",
            ),
        ),
        custom_data=["molecule_id"],
        category_orders={"display_status": IDENTITY_DISPLAY_ORDER},
        color_discrete_map=IDENTITY_DISPLAY_COLORS,
        labels={
            **display_labels(plot_df.columns),
            "display_status": "Identity status",
        },
        title="Chemical identity evidence by molecule",
    )
    return figure


def chemical_identity_debug_dataframe(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return the requested row-level fields used to verify plot categories."""
    columns = available_columns(
        plot_df,
        (
            "molecule_id",
            "identity_status",
            "lookup_status",
            "identity_confidence",
            "display_status",
        ),
    )
    return plot_df[columns].copy()


def chemical_identity_sanity_summary(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    csv_path: Path,
) -> dict[str, object]:
    """Build browser-visible provenance and status counts from chemical_identity.csv."""
    plot_df = identity_molecule_dataframe(frame)

    def counts(column: str) -> dict[str, int]:
        if column not in frame.columns:
            return {}
        values = frame[column].fillna("not_available").astype(str).value_counts()
        return {str(value): int(count) for value, count in values.items()}

    display_counts = (
        {
            str(value): int(count)
            for value, count in plot_df["display_status"].value_counts().items()
        }
        if "display_status" in plot_df.columns
        else {}
    )
    return {
        "active_output_folder": str(output_dir),
        "chemical_identity_csv": str(csv_path),
        "identity_status_counts": counts("identity_status"),
        "lookup_status_counts": counts("lookup_status"),
        "display_status_counts": display_counts,
    }


def chemical_identity_summary_table(summary: dict[str, object]) -> pd.DataFrame:
    """Convert identity sanity counts to a compact browser table."""
    rows = []
    for source, label in (
        ("identity_status_counts", "Raw identity status"),
        ("lookup_status_counts", "Raw lookup status"),
        ("display_status_counts", "Mapped display status"),
    ):
        values = summary.get(source, {})
        if not isinstance(values, dict):
            continue
        rows.extend(
            {"Count source": label, "Status": status, "Molecules": count}
            for status, count in values.items()
        )
    return pd.DataFrame(rows)


def chemical_identity_lookup_warning(plot_df: pd.DataFrame) -> str:
    """Return a warning only for rows finally mapped to Lookup error."""
    if plot_df.empty or "display_status" not in plot_df.columns:
        return ""
    count = int(plot_df["display_status"].eq("Lookup error").sum())
    return f"External lookup failed for {count} molecules." if count else ""


def identity_lookup_failure_message(frame: pd.DataFrame) -> str:
    """Explain when every query-eligible identity lookup failed."""
    if frame.empty or "lookup_status" not in frame.columns:
        return ""
    statuses = frame["lookup_status"].fillna("").astype(str).str.lower()
    query_eligible = ~statuses.isin({"invalid_smiles", "invalid_molecule"})
    eligible_statuses = statuses[query_eligible]
    if eligible_statuses.empty or not eligible_statuses.eq("lookup_error").all():
        return ""
    error = ""
    if "error_message" in frame.columns:
        errors = (
            frame.loc[query_eligible, "error_message"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        nonempty = errors[errors.ne("")]
        error = nonempty.iloc[0] if not nonempty.empty else ""
    message = (
        f"Online chemical-identity lookup failed for all {len(eligible_statuses)} "
        "valid molecules. The structures remain valid, but no public identity "
        "result was returned."
    )
    if "WinError 10013" in error:
        return (
            message
            + " Windows blocked the outbound network connection (WinError 10013). "
            "Rerun the guided workflow from a process with network permission."
        )
    return message + (f" Service response: {error}" if error else "")


def render_identity_view(
    frame: pd.DataFrame,
    *,
    key: str,
    output_dir: Path,
    csv_path: Path,
) -> str:
    """Render one interactive identity point per molecule."""
    plot_df = identity_molecule_dataframe(frame)
    if plot_df.empty:
        st.info("Chemical identity output is unavailable.")
        return ""
    summary = chemical_identity_sanity_summary(
        frame,
        output_dir=output_dir,
        csv_path=csv_path,
    )
    st.markdown("#### Chemical identity sanity check")
    st.caption(f"Active output folder: {summary['active_output_folder']}")
    st.caption(f"chemical_identity.csv: {summary['chemical_identity_csv']}")
    st.dataframe(
        chemical_identity_summary_table(summary),
        width="stretch",
        hide_index=True,
    )
    warning = chemical_identity_lookup_warning(plot_df)
    if warning:
        st.warning(warning)
    figure = chemical_identity_figure(plot_df)
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    st.markdown("#### Chemical identity status details")
    st.dataframe(
        chemical_identity_debug_dataframe(plot_df),
        width="stretch",
        hide_index=True,
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_public_evidence_view(
    public_lookup: pd.DataFrame,
    surechembl: pd.DataFrame,
    molecule_ids: Iterable[str],
    *,
    key: str,
) -> str:
    """Render a molecule-by-database evidence grid and selector."""
    evidence = public_evidence_molecule_dataframe(
        public_lookup, surechembl, molecule_ids
    )
    if evidence.empty:
        st.info("Public database evidence is unavailable.")
        return ""
    status_fields = [
        "pubchem_status",
        "chembl_status",
        "surechembl_query_status",
    ]
    display_evidence = evidence[
        ["molecule_id", *status_fields]
    ].copy()
    for column in status_fields:
        display_evidence[column] = display_evidence[column].map(readable_status)
    readable = display_dataframe(display_evidence)
    status_columns = [
        display_label("pubchem_status"),
        display_label("chembl_status"),
        display_label("surechembl_query_status"),
    ]
    styled = readable.style.map(
        lambda value: (
            f"background-color: {MOLECULE_STATUS_COLORS[molecule_status_color(value)]}; "
            "color: white; font-weight: 700"
        ),
        subset=status_columns,
    )
    st.markdown("#### Molecule-level public evidence")
    st.caption("SureChEMBL public structure evidence is reported as structure evidence; patent document metadata is shown only when retrieved.")
    st.dataframe(styled, width="stretch", hide_index=True)
    plot_evidence = evidence.copy()
    for column in status_fields:
        plot_evidence[column] = plot_evidence[column].map(readable_status)
    figure = px.scatter(
        plot_evidence,
        x="molecule_position",
        y="evidence_status",
        color="evidence_status",
        hover_name="molecule_id",
        hover_data=[
            "pubchem_status",
            "chembl_status",
            "surechembl_query_status",
        ],
        custom_data=["molecule_id"],
        color_discrete_map={
            "Available": MOLECULE_STATUS_COLORS["green"],
            "Partial": MOLECULE_STATUS_COLORS["yellow"],
            "Missing": MOLECULE_STATUS_COLORS["red"],
            "Not run": MOLECULE_STATUS_COLORS["gray"],
        },
        labels={
            **display_labels(plot_evidence.columns),
            "molecule_position": "Molecule",
            "evidence_status": "Public evidence status",
        },
        title="Public database evidence by molecule",
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_evidence, key=key, plot_event=event)


def render_chemical_space(
    loaded: LoadedOutputs, prioritization: pd.DataFrame, *, key: str = "chemical_space"
) -> str:
    """Render chemical space before or after final prioritization exists."""
    st.caption(CHEMICAL_SPACE_EXPLANATION)
    coords = loaded.tables["visualization"]
    if coords.empty:
        st.info("visualization_coordinates.csv was not found; chemical-space plot is unavailable.")
        return ""
    plot_df = chemical_space_dataframe(coords, prioritization)
    if "source_type" not in plot_df.columns:
        plot_df["source_type"] = "generated"
    st.info(
        "Reference molecules are plotted in the same chemical-space projection "
        "as generated molecules. Distances are useful for visual triage but "
        "should be interpreted together with fingerprint similarity and other evidence."
    )
    render_chemical_space_summary(plot_df)
    if loaded.paths["visualization"].exists():
        st.download_button(
            "Download visualization_coordinates.csv",
            data=loaded.paths["visualization"].read_bytes(),
            file_name="visualization_coordinates.csv",
            mime="text/csv",
            key=f"{key}_download_visualization_coordinates",
        )
    nearest_table = nearest_reference_table(plot_df)
    if not nearest_table.empty:
        with st.expander("Nearest-reference table", expanded=False):
            st.dataframe(nearest_table.head(10), width="stretch", hide_index=True, height=260)
    cluster_summary = chemical_space_cluster_summary(plot_df)
    if not cluster_summary.empty:
        cluster_count = int(len(cluster_summary))
        st.metric("Chemical-space clusters", cluster_count)
        with st.expander("Cluster summary", expanded=False):
            st.dataframe(cluster_summary, width="stretch", hide_index=True, height=260)
    distribution = nearest_similarity_distribution(plot_df)
    if int(distribution["Generated molecules"].sum()) > 0:
        with st.expander("Nearest-reference similarity distribution", expanded=False):
            st.table(distribution)
            histogram_df = plot_df[
                plot_df["source_type"].fillna("").astype(str) == "generated"
            ].copy()
            histogram_df = coerce_numeric(histogram_df, ["nearest_reference_similarity"])
            histogram = px.histogram(
                histogram_df,
                x="nearest_reference_similarity",
                nbins=12,
                labels=display_labels(histogram_df.columns),
                title="Nearest-reference similarity distribution",
            )
            histogram.update_layout(height=230, margin={"l": 20, "r": 20, "t": 45, "b": 35})
            st.plotly_chart(histogram, width="stretch", key=f"{key}_similarity_histogram")
    color_options = chemical_space_color_options(plot_df)
    color_label = st.selectbox(
        "Color points by",
        list(color_options),
        index=0,
        key=f"{key}_color_points_by",
    )
    st.caption(
        "Use Source type to compare generated vs reference molecules. Use "
        "Chemical-space cluster to inspect local regions of chemical similarity."
    )
    show_links = st.checkbox(
        "Show nearest-reference links",
        value=False,
        key=f"{key}_show_nearest_reference_links",
    )
    fig = build_chemical_space_figure(
        plot_df,
        color_by=str(color_label),
        show_links=show_links,
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    st.caption(
        "Generated and reference molecules are projected together using the same "
        "fingerprint representation and dimensionality-reduction fit. The 2D map "
        "is intended for visual triage; nearest-reference Tanimoto similarity "
        "should be used for more reliable similarity interpretation. Use Source "
        "type view to compare generated and reference molecules. Use Chemical-space "
        "cluster view to inspect local regions of chemical similarity."
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_score_similarity(
    prioritization: pd.DataFrame, *, key: str = "score_similarity"
) -> str:
    """Render score versus local reference similarity."""
    st.caption(SCORE_SIMILARITY_EXPLANATION)
    if prioritization.empty or "tanimoto_similarity" not in prioritization.columns:
        st.info("Reference similarity data is unavailable.")
        return ""
    plot_df = final_priority_molecule_dataframe(prioritization)
    plot_df["display_design_category"] = plot_df["design_category"].map(
        readable_status
    )
    for column in (
        "pubchem_status",
        "chembl_status",
        "surechembl_query_status",
        "chemberta_status",
        "nlp_status",
    ):
        if column in plot_df.columns:
            plot_df[column] = plot_df[column].map(readable_status)
    active_score = score_column(plot_df)
    fig = px.scatter(
        plot_df,
        x="tanimoto_similarity",
        y=active_score,
        hover_name="molecule_id" if "molecule_id" in plot_df.columns else None,
        color="display_design_category",
        hover_data=available_columns(
            plot_df,
            (
                "best_reference_name",
                "pubchem_status",
                "chembl_status",
                "surechembl_query_status",
                "chemberta_status",
                "nlp_status",
            ),
        ),
        custom_data=["molecule_id"],
        title="Score vs RDKit Reference Similarity",
        color_discrete_map={
            readable_status(value): color
            for value, color in category_color_map(
                plot_df, "design_category"
            ).items()
        },
        labels={
            **display_labels(plot_df.columns),
            "display_design_category": "Final design category",
        },
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_descriptor_histograms(descriptors: pd.DataFrame) -> None:
    """Render descriptor distribution histograms."""
    st.caption(PROPERTY_DISTRIBUTIONS_EXPLANATION)
    if descriptors.empty:
        st.info("descriptors.csv was not found; property distributions are unavailable.")
        return
    cols = st.columns(2)
    for index, column in enumerate(DESCRIPTOR_COLUMNS):
        if column not in descriptors.columns:
            continue
        plot_df = coerce_numeric(descriptors, [column])
        readable = display_label(column)
        fig = px.histogram(
            plot_df,
            x=column,
            labels=display_labels(plot_df.columns),
            title=f"{readable} distribution",
        )
        cols[index % 2].plotly_chart(fig, width="stretch")


def druglikeness_counts(descriptors: pd.DataFrame) -> dict[str, int]:
    """Return stable counts for all drug-likeness categories."""
    values = descriptors.get(
        "druglikeness_category", pd.Series(dtype="string")
    )
    counts = values.fillna("invalid").astype(str).value_counts()
    return {
        category: int(counts.get(category, 0))
        for category in ("favorable", "borderline", "unfavorable", "invalid")
    }


def readable_druglikeness_status(value: object) -> str:
    """Return a readable icon and label for a property status."""
    status = str(value or "").strip().lower() or "invalid"
    return DRUGLIKENESS_ICONS.get(status, status)


def druglikeness_status_matrix(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Build a molecule-by-property matrix with readable status labels."""
    columns = {
        "mw_status": "MW",
        "logp_status": "LogP",
        "tpsa_status": "TPSA",
        "qed_status": "QED",
        "lipinski_status": "Lipinski",
    }
    if descriptors.empty or "molecule_id" not in descriptors.columns:
        return pd.DataFrame(columns=["Molecule ID", *columns.values()])
    available = [column for column in columns if column in descriptors.columns]
    matrix = descriptors[["molecule_id", *available]].copy()
    matrix = matrix.rename(
        columns={"molecule_id": "Molecule ID", **columns}
    )
    for column in columns.values():
        if column in matrix.columns:
            matrix[column] = matrix[column].map(readable_druglikeness_status)
    return matrix


def color_druglikeness_category(value: object) -> str:
    """Return CSS styling for a drug-likeness category cell."""
    color = DRUGLIKENESS_COLORS.get(str(value or "").strip().lower())
    return f"color: {color}; font-weight: 700" if color else ""


def render_druglikeness_views(
    descriptors: pd.DataFrame, *, key: str = "rdkit_properties"
) -> str:
    """Render molecule-level drug-likeness table, scatter, and selection."""
    st.info(DRUGLIKENESS_EXPLANATION)
    if descriptors.empty:
        st.info("descriptors.csv was not found; drug-likeness views are unavailable.")
        return ""

    plot = rdkit_molecule_dataframe(descriptors)
    table_columns = available_columns(
        plot,
        (
            "molecule_id",
            "molecular_weight",
            "logp",
            "tpsa",
            "hbd",
            "hba",
            "rotatable_bonds",
            "qed",
            "lipinski_pass",
            "druglikeness_score",
            "druglikeness_category",
        ),
    )
    st.markdown("#### Molecule-level drug-likeness")
    table = display_dataframe(plot[table_columns].copy())
    category_label = display_label("druglikeness_category")
    styled_table = (
        table.style.map(
            color_druglikeness_category,
            subset=[category_label],
        )
        if category_label in table.columns
        else table
    )
    st.dataframe(styled_table, width="stretch", hide_index=True)

    valid_plot = plot.dropna(subset=["logp", "qed"])
    event = None
    if not valid_plot.empty:
        figure = px.scatter(
            valid_plot,
            x="logp",
            y="qed",
            color="druglikeness_category",
            hover_name="molecule_id",
            hover_data=available_columns(
                valid_plot,
                (
                    "molecular_weight",
                    "tpsa",
                    "hbd",
                    "hba",
                    "rotatable_bonds",
                    "lipinski_pass",
                    "druglikeness_score",
                ),
            ),
            custom_data=["molecule_id"],
            category_orders={
                "druglikeness_category": [
                    "favorable",
                    "borderline",
                    "unfavorable",
                    "invalid",
                ]
            },
            color_discrete_map=DRUGLIKENESS_COLORS,
            labels=display_labels(valid_plot.columns),
            title="LogP vs QED by drug-likeness category",
        )
        event = st.plotly_chart(
            figure,
            width="stretch",
            key=f"{key}_plot",
            on_select="rerun",
            selection_mode="points",
        )
    return render_molecule_selector(plot, key=key, plot_event=event)


def render_text_evidence_view(
    text_nlp: pd.DataFrame,
    molecule_ids: Iterable[str],
    *,
    key: str,
) -> str:
    """Render one text-evidence point per molecule and a readable summary table."""
    plot_df = text_evidence_molecule_dataframe(text_nlp, molecule_ids)
    if plot_df.empty:
        st.info("Text evidence is unavailable.")
        return ""
    if text_nlp_model_unavailable(text_nlp):
        st.info(
            "Text-evidence NLP was skipped because the embedding model is "
            "unavailable in this cloud environment."
        )
    plot_df["display_evidence_score"] = pd.to_numeric(
        plot_df["max_relevance_score"], errors="coerce"
    ).fillna(0.0)
    plot_df["display_nlp_status"] = plot_df["nlp_status"].map(readable_status)
    render_modern_evidence_card(
        "Text evidence summary",
        {
            "Molecules": len(plot_df),
            "Available matches": int(plot_df["nlp_status"].astype(str).eq("available").sum()),
            "Evidence rows": int(pd.to_numeric(plot_df["evidence_matches"], errors="coerce").fillna(0).sum()),
        },
        key=f"{key}_text_evidence_summary_card",
    )
    table_columns = [
        "molecule_id",
        "nlp_status",
        "max_relevance_score",
        "top_evidence_title",
        "evidence_matches",
    ]
    st.dataframe(
        display_dataframe(plot_df[table_columns]),
        width="stretch",
        hide_index=True,
    )
    figure = px.scatter(
        plot_df,
        x="molecule_position",
        y="display_evidence_score",
        color="display_nlp_status",
        size="evidence_matches",
        hover_name="molecule_id",
        hover_data=["top_evidence_title", "evidence_matches"],
        custom_data=["molecule_id"],
        color_discrete_map={
            readable_status(value): color
            for value, color in category_color_map(plot_df, "nlp_status").items()
        },
        labels={
            **display_labels(plot_df.columns),
            "display_evidence_score": "Text-evidence score",
            "display_nlp_status": "Text-evidence status",
        },
        title="Text evidence by molecule",
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_biomedical_evidence_view(
    biomedical: pd.DataFrame,
    molecule_ids: Iterable[str],
    *,
    key: str,
) -> str:
    """Render molecule-level biomedical evidence status and matching results."""
    plot_df = biomedical_evidence_molecule_dataframe(biomedical, molecule_ids)
    if plot_df.empty:
        st.info("Biomedical evidence matching was not run.")
        return ""
    if biomedical_model_unavailable(biomedical):
        st.info(
            "Biomedical evidence matching was skipped because the embedding "
            "model is unavailable in this cloud environment."
        )
    model_available_count = int(
        plot_df["biomedical_model_status"].astype(str).str.strip().eq("available").sum()
    )
    model_skipped_count = int(
        plot_df["biomedical_model_status"]
        .astype(str)
        .str.strip()
        .eq("model_unavailable")
        .sum()
    )
    evidence_matched_count = int(
        plot_df["biomedical_evidence_status"].astype(str).str.strip().eq("available").sum()
    )
    card_cols = st.columns(2)
    with card_cols[0]:
        render_modern_evidence_card(
            "Biomedical evidence summary",
            {
                "Molecules": len(plot_df),
                "Evidence matched": evidence_matched_count,
                "Evidence skipped": int(len(plot_df) - evidence_matched_count),
            },
            key=f"{key}_biomedical_summary_card",
        )
    with card_cols[1]:
        render_modern_evidence_card(
            "Model/fallback status",
            {
                "Model available": model_available_count,
                "Model skipped": model_skipped_count,
                "Fallback preserved": "yes",
            },
            key=f"{key}_biomedical_model_status_card",
        )
    left, middle, right = st.columns(3)
    with left:
        st.metric(
            "Model available",
            int(
                plot_df["biomedical_model_status"]
                .astype(str)
                .str.strip()
                .eq("available")
                .sum()
            ),
        )
    with middle:
        st.metric(
            "Model skipped",
            int(
                plot_df["biomedical_model_status"]
                .astype(str)
                .str.strip()
                .eq("model_unavailable")
                .sum()
            ),
        )
    with right:
        st.metric(
            "Evidence matched",
            int(
                plot_df["biomedical_evidence_status"]
                .astype(str)
                .str.strip()
                .eq("available")
                .sum()
            ),
        )
    plot_df = coerce_numeric(
        plot_df,
        ("biomedical_similarity_score", "biomedical_evidence_count"),
    )
    table_columns = available_columns(
        plot_df,
        (
            "molecule_id",
            "biomedical_model_status",
            "biomedical_evidence_status",
            "biomedical_similarity_score",
            "biomedical_relevance_category",
            "biomedical_evidence_count",
            "top_biomedical_evidence_id",
        ),
    )
    with st.expander("Preview table", expanded=False):
        st.dataframe(
            display_dataframe(readable_ui_dataframe(plot_df[table_columns])),
            width="stretch",
            hide_index=True,
            height=260,
        )
    plot_df["display_biomedical_status"] = plot_df[
        "biomedical_evidence_status"
    ].map(readable_status)
    figure = px.scatter(
        plot_df,
        x="molecule_position",
        y="biomedical_similarity_score",
        color="display_biomedical_status",
        size="biomedical_evidence_count",
        hover_name="molecule_id",
        hover_data=available_columns(
            plot_df,
            (
                "biomedical_model_status",
                "biomedical_relevance_category",
                "top_biomedical_evidence_id",
            ),
        ),
        custom_data=["molecule_id"],
        color_discrete_map={
            readable_status(value): color
            for value, color in category_color_map(
                plot_df, "biomedical_evidence_status"
            ).items()
        },
        labels={
            **display_labels(plot_df.columns),
            "display_biomedical_status": "Biomedical evidence status",
        },
        title="Biomedical evidence by molecule",
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_patent_evidence_view(
    patent: pd.DataFrame,
    molecule_ids: Iterable[str],
    *,
    key: str,
) -> str:
    """Render patent/IP-context evidence without making legal conclusions."""
    plot_df = patent_evidence_molecule_dataframe(patent, molecule_ids)
    if plot_df.empty:
        st.info("Patent/IP-context evidence matching was not run.")
        return ""
    if patent_model_unavailable(patent):
        st.info(
            "Patent/IP-context evidence matching was skipped because the "
            "embedding model is unavailable in this cloud environment."
        )
    st.caption(
        "These outputs separate public structure evidence, patent document "
        "metadata, and optional patent-text embedding evidence. They are "
        "research triage signals, not legal conclusions."
    )
    structure_available = (
        plot_df["surechembl_structure_status"]
        .astype(str)
        .str.strip()
        .isin(["match_found", "available", "hit"])
        .sum()
    )
    metadata_available = (
        plot_df["patent_document_metadata_status"]
        .astype(str)
        .str.strip()
        .isin(["available", "match_found", "hit"])
        .sum()
    )
    embedding_available = (
        plot_df["patent_evidence_status"]
        .astype(str)
        .str.strip()
        .eq("available")
        .sum()
    )
    patent_card_cols = st.columns(2)
    with patent_card_cols[0]:
        render_modern_evidence_card(
            "Patent/IP-context evidence summary",
            {
                "Molecules": len(plot_df),
                "Structure evidence": int(structure_available),
                "Patent metadata": int(metadata_available),
            },
            key=f"{key}_patent_summary_card",
        )
    with patent_card_cols[1]:
        render_modern_evidence_card(
            "Patent-text evidence and model status",
            {
                "Patent-text evidence": int(embedding_available),
                "Model unavailable": int(
                    plot_df["patent_model_status"]
                    .astype(str)
                    .str.strip()
                    .eq("model_unavailable")
                    .sum()
                ),
                "Fallback preserved": "yes",
            },
            key=f"{key}_patent_model_status_card",
        )
    render_modern_evidence_card(
        "Structure evidence",
        {
            "SureChEMBL structure evidence": int(structure_available),
            "Patent document metadata": int(metadata_available),
        },
        key=f"{key}_patent_structure_card",
    )
    left, middle, right = st.columns(3)
    with left:
        st.metric("SureChEMBL structure evidence", int(structure_available))
    with middle:
        st.metric("Patent document metadata", int(metadata_available))
    with right:
        st.metric("Patent-text embeddings", int(embedding_available))
    plot_df = coerce_numeric(
        plot_df,
        ("patent_similarity_score", "patent_evidence_count"),
    )
    table_columns = available_columns(
        plot_df,
        (
            "molecule_id",
            "surechembl_structure_status",
            "patent_document_metadata_status",
            "patent_model_status",
            "patent_evidence_status",
            "patent_similarity_score",
            "patent_relevance_category",
            "patent_evidence_count",
            "top_patent_evidence_id",
        ),
    )
    with st.expander("Preview table", expanded=False):
        st.dataframe(
            display_dataframe(readable_ui_dataframe(plot_df[table_columns])),
            width="stretch",
            hide_index=True,
            height=260,
        )
    plot_df["display_patent_status"] = plot_df["patent_evidence_status"].map(
        readable_status
    )
    figure = px.scatter(
        plot_df,
        x="molecule_position",
        y="patent_similarity_score",
        color="display_patent_status",
        size="patent_evidence_count",
        hover_name="molecule_id",
        hover_data=available_columns(
            plot_df,
            (
                "surechembl_structure_status",
                "patent_document_metadata_status",
                "patent_relevance_category",
                "top_patent_evidence_id",
            ),
        ),
        custom_data=["molecule_id"],
        color_discrete_map={
            readable_status(value): color
            for value, color in category_color_map(
                plot_df, "patent_evidence_status"
            ).items()
        },
        labels={
            **display_labels(plot_df.columns),
            "display_patent_status": "Patent/IP evidence status",
        },
        title="Patent/IP-context evidence by molecule",
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
    )
    return render_molecule_selector(plot_df, key=key, plot_event=event)


def render_detail_panel(loaded: LoadedOutputs, molecule_id: str) -> None:
    """Render the shared cross-workflow detail panel for one molecule."""
    prioritization = loaded.tables["prioritization"]
    descriptors = loaded.tables["descriptors"]
    if not molecule_id:
        return
    st.subheader(f"Molecule detail: {molecule_id}")
    image_path = loaded.images_dir / f"{molecule_id}.png"
    if image_path.exists():
        st.image(str(image_path), caption=f"2D structure: {molecule_id}", width=360)
    else:
        structure_image = molecule_structure_image(
            molecule_smiles_from_outputs(loaded, molecule_id)
        )
        if structure_image is not None:
            st.image(
                structure_image,
                caption=f"2D structure: {molecule_id}",
                width=360,
            )
        else:
            st.info(molecule_image_message(image_path))

    standardized = loaded.tables["standardized"]
    if not standardized.empty and "molecule_id" in standardized.columns:
        structure_row = standardized[
            standardized["molecule_id"].astype(str) == molecule_id
        ]
        if not structure_row.empty:
            st.markdown("#### Structure and validation")
            columns = available_columns(
                structure_row,
                (
                    "molecule_id",
                    "smiles",
                    "canonical_smiles",
                    "valid_smiles",
                    "error_message",
                ),
            )
            compact = compact_detail_dataframe(structure_row[columns])
            st.table(display_dataframe(compact).T)

    row = (
        prioritization[prioritization["molecule_id"].astype(str) == molecule_id]
        if not prioritization.empty and "molecule_id" in prioritization.columns
        else pd.DataFrame()
    )
    if not row.empty:
        selected = row.iloc[0].to_dict()
        st.markdown("#### Evidence completeness")
        st.table(evidence_completeness_rows(selected))

    if not descriptors.empty and "molecule_id" in descriptors.columns:
        descriptor_row = descriptors[
            descriptors["molecule_id"].astype(str) == molecule_id
        ]
        if not descriptor_row.empty:
            st.markdown("#### RDKit drug-likeness")
            columns = available_columns(
                descriptor_row,
                (
                    "molecular_weight",
                    "logp",
                    "tpsa",
                    "hbd",
                    "hba",
                    "rotatable_bonds",
                    "qed",
                    "lipinski_pass",
                    "druglikeness_category",
                    "druglikeness_flags",
                ),
            )
            compact = compact_detail_dataframe(descriptor_row[columns])
            st.table(
                display_dataframe(readable_ui_dataframe(compact))
                .T.rename(columns={descriptor_row.index[0]: "Value"})
            )

    identities = loaded.tables["chemical_identity"]
    if not identities.empty and "molecule_id" in identities.columns:
        identity_row = identities[
            identities["molecule_id"].astype(str) == molecule_id
        ]
        if not identity_row.empty:
            st.markdown("#### Chemical identity")
            columns = [
                column
                for column in (
                    "identity_status",
                    "exact_public_name",
                    "preferred_name",
                    "iupac_name",
                    "synonyms",
                    "inchikey",
                    "pubchem_cid",
                    "chembl_id",
                    "name_source",
                    "identity_confidence",
                    "lookup_status",
                )
                if column in identity_row.columns
            ]
            compact = compact_detail_dataframe(identity_row[columns])
            st.table(
                display_dataframe(readable_ui_dataframe(compact))
                .T.rename(columns={identity_row.index[0]: "Value"})
            )

    public_lookup = loaded.tables["public_lookup"]
    public_rows = (
        public_lookup[public_lookup["molecule_id"].astype(str) == molecule_id]
        if not public_lookup.empty and "molecule_id" in public_lookup.columns
        else pd.DataFrame()
    )
    surechembl = loaded.tables["surechembl"]
    sure_rows = (
        surechembl[surechembl["molecule_id"].astype(str) == molecule_id]
        if not surechembl.empty and "molecule_id" in surechembl.columns
        else pd.DataFrame()
    )
    if not public_rows.empty or not sure_rows.empty:
        st.markdown("#### Public database evidence")
        if not public_rows.empty:
            columns = available_columns(
                public_rows,
                (
                    "source_database",
                    "lookup_status",
                    "match_type",
                    "public_id",
                    "public_name",
                    "similarity",
                    "evidence_note",
                    "error_message",
                ),
            )
            st.dataframe(
                display_dataframe(readable_ui_dataframe(public_rows[columns])),
                width="stretch",
                hide_index=True,
            )
        if not sure_rows.empty:
            render_surechembl_public_structure_evidence(sure_rows)

    visualization = loaded.tables["visualization"]
    visualization_row = (
        visualization[visualization["molecule_id"].astype(str) == molecule_id]
        if not visualization.empty and "molecule_id" in visualization.columns
        else pd.DataFrame()
    )
    if not visualization_row.empty:
        st.markdown("#### ChemBERTa context")
        columns = available_columns(
            visualization_row,
            (
                "cluster_id",
                "coordinate_method",
                "best_reference_name",
                "tanimoto_similarity",
                "chemberta_status",
            ),
        )
        st.table(
            display_dataframe(visualization_row[columns])
            .T.rename(columns={visualization_row.index[0]: "Value"})
        )

    text_nlp = loaded.tables["text_nlp"]
    text_rows = (
        text_nlp[text_nlp["molecule_id"].astype(str) == molecule_id].copy()
        if not text_nlp.empty and "molecule_id" in text_nlp.columns
        else pd.DataFrame()
    )
    if not text_rows.empty:
        st.markdown("#### Text evidence")
        ranking_column = next(
            (
                column
                for column in ("max_relevance_score", "similarity_score")
                if column in text_rows.columns
            ),
            "",
        )
        if ranking_column:
            text_rows[ranking_column] = pd.to_numeric(
                text_rows[ranking_column], errors="coerce"
            )
            text_rows = text_rows.sort_values(ranking_column, ascending=False)
        columns = available_columns(
            text_rows,
            (
                "nlp_status",
                "title",
                "source_type",
                "max_relevance_score",
                "similarity_score",
                "nlp_relevance_category",
                "nlp_notes",
            ),
        )
        st.dataframe(
            display_dataframe(
                readable_ui_dataframe(text_rows[columns].head(5))
            ),
            width="stretch",
            hide_index=True,
        )

    if not row.empty:
        st.markdown("#### Final design prioritization")
        st.write(
            "The final view combines structure validity, public evidence, RDKit "
            "properties, reference similarity, ChemBERTa availability, and text "
            "evidence while preserving unavailable-stage statuses."
        )
        st.table(molecule_detail_rows(row.iloc[0].to_dict(), score_column(prioritization)))

def render_new_analysis_form() -> Path | None:
    """Render upload-and-run form and return new output path after success."""
    defaults = default_workflow_options()
    st.header("Input Data / Run New Analysis")
    st.info(ONLINE_LOOKUP_NOTE)
    st.caption("Uploads and generated results are saved locally under app_runs.")
    with st.form("run_new_analysis"):
        run_name = st.text_input("Run name", value="")
        generated_upload = st.file_uploader(
            "Generated SMILES CSV (required: molecule_id, smiles)",
            type=["csv"],
        )
        reference_upload = st.file_uploader(
            "Reference CSV (optional; smiles column required if uploaded)",
            type=["csv"],
        )
        text_upload = st.file_uploader(
            "Text evidence CSV (optional)",
            type=["csv"],
        )

        st.markdown("#### Workflow options")
        col1, col2, col3 = st.columns(3)
        online_lookup = col1.checkbox(
            "Run PubChem/ChEMBL lookup",
            value=bool(defaults["online_lookup"]),
        )
        online_surechembl = col2.checkbox(
            "Run SureChEMBL structure evidence search",
            value=bool(defaults["online_surechembl"]),
        )
        use_chemberta = col3.checkbox(
            "Run ChemBERTa embeddings",
            value=bool(defaults["use_chemberta"]),
        )
        col4, col5, col6 = st.columns(3)
        generate_reports = col4.checkbox(
            "Generate top-N reports",
            value=bool(defaults["generate_reports"]),
        )
        report_only_fully_analyzed = col5.checkbox(
            "Report only fully analyzed molecules",
            value=bool(defaults["report_only_fully_analyzed"]),
        )
        max_molecules_value = col6.number_input(
            "Maximum molecules",
            min_value=0,
            max_value=100000,
            value=int(defaults["max_molecules"]),
            help="Use 0 to analyze all molecules.",
        )
        report_top_n = st.number_input(
            "Number of top reports",
            min_value=1,
            max_value=100,
            value=int(defaults["report_top_n"]),
        )
        submitted = st.form_submit_button("Run analysis")

    if not submitted:
        return None
    if generated_upload is None:
        st.error("Please upload a generated SMILES CSV with molecule_id and smiles columns.")
        return None

    try:
        paths = prepare_app_run_inputs(
            run_name=run_name,
            generated_upload=generated_upload,
            reference_upload=reference_upload,
            text_upload=text_upload,
        )
        output_path = run_uploaded_analysis(
            paths,
            online_lookup=online_lookup,
            online_surechembl=online_surechembl,
            use_chemberta=use_chemberta,
            report_top_n=int(report_top_n) if generate_reports else None,
            report_only_fully_analyzed=report_only_fully_analyzed,
            max_molecules=(
                int(max_molecules_value) if int(max_molecules_value) > 0 else None
            ),
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return None

    st.success(f"Analysis complete: {output_path.parent}")
    st.session_state["active_output_dir"] = str(output_path.parent)
    st.session_state["workflow_step"] = 1
    st.session_state["completed_workflow_steps"] = list(
        range(1, len(WORKFLOW_STEP_NAMES) + 1)
    )
    st.session_state["workflow_mode"] = "completed_run"
    return output_path.parent


def public_demo_input_paths() -> tuple[Path, Path, Path]:
    """Return the public-safe example files shown on the welcome workflow."""
    return DEMO_INPUT, DEMO_REFERENCES, DEMO_TEXT_EVIDENCE


def input_paths_for_active_run(
    output_dir: Path,
    workflow_mode: str,
) -> tuple[Path, Path, Path]:
    """Return expected generated/reference/text input paths for an active run."""
    if workflow_mode == "public_demo":
        return public_demo_input_paths()
    input_dir = output_dir.parent / "inputs"
    return (
        input_dir / "generated_smiles.csv",
        input_dir / "references.csv",
        input_dir / "text_evidence.csv",
    )


def existing_output_artifacts(output_dir: Path) -> tuple[str, ...]:
    """Return known output artifact names already present in an output folder."""
    existing = [
        filename
        for filename in OUTPUT_FILES.values()
        if (output_dir / filename).exists()
    ]
    reports_dir = output_dir / "reports"
    if reports_dir.exists() and any(
        reports_dir.glob("compound_intelligence_report_*.md")
    ):
        existing.append("reports")
    return tuple(sorted(existing))


def resolve_active_run_paths(
    output_dir: str | Path,
    workflow_mode: str,
) -> ActiveRunPathStatus:
    """Resolve active-run paths without executing or creating outputs."""
    root = Path(output_dir)
    mode = str(workflow_mode or "").strip() or "unknown"
    generated_path, references_path, text_evidence_path = input_paths_for_active_run(
        root,
        mode,
    )
    pipeline_paths = build_paths(
        input_path=generated_path,
        references_path=references_path,
        text_evidence_path=text_evidence_path,
        output_dir=root,
    )
    input_labels = {
        "generated_smiles": generated_path,
        "references": references_path,
        "text_evidence": text_evidence_path,
    }
    missing_inputs = tuple(
        label for label, path in input_labels.items() if not path.exists()
    )
    existing_outputs = existing_output_artifacts(root)
    if mode == "public_demo":
        run_type = "public_demo"
    elif mode in {"completed_run", "uploaded", "custom"}:
        run_type = "uploaded"
    elif mode == "existing_results":
        run_type = "loaded_previous"
    else:
        run_type = mode
    unresolved_paths = missing_inputs
    paths_resolved = not missing_inputs
    return ActiveRunPathStatus(
        run_type=run_type,
        workflow_mode=mode,
        output_dir=root,
        pipeline_paths=pipeline_paths,
        paths_resolved=paths_resolved,
        missing_inputs=missing_inputs,
        existing_outputs=existing_outputs,
        unresolved_paths=unresolved_paths,
    )


def format_file_size(path: Path) -> str:
    """Return a compact human-readable file size."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size} bytes"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
def render_output_artifact_card(
    label: str,
    dataframe: pd.DataFrame,
    filename: str,
    description: str,
    *,
    data: bytes,
    mime: str = "text/csv",
    preview_rows: int = 10,
    key_prefix: str,
    file_size: str | None = None,
    raw_text: str | None = None,
) -> None:
    """Render a compact artifact summary with hidden optional previews."""
    row_word = "row" if len(dataframe) == 1 else "rows"
    size_text = file_size or "size unavailable"
    legacy_size_text = f" - {file_size}" if file_size else ""
    st.markdown(f"#### {label}")
    st.write(description)
    st.write(
        f"Artifact: {filename} - {len(dataframe)} {row_word} - "
        f"{len(dataframe.columns)} columns{legacy_size_text}"
    )
    render_modern_card(
        label,
        description,
        description=f"{filename} | {len(dataframe)} {row_word} | {len(dataframe.columns)} columns | {size_text}",
        badges=(("Downloadable artifact", "secondary"),),
        key=f"{key_prefix}_artifact_card_{slugify_key(filename)}",
    )
    st.download_button(
        f"Download {filename}",
        data=data,
        file_name=filename,
        mime=mime,
        key=f"{key_prefix}_download_{filename}",
    )
    with st.expander("Preview table", expanded=False):
        preview = compact_preview_dataframe(dataframe, preview_rows=preview_rows)
        st.dataframe(preview, width="stretch", hide_index=True, height=260)
        if len(dataframe.columns) > len(preview.columns):
            hidden_count = len(dataframe.columns) - len(preview.columns)
            if hasattr(st, "caption"):
                st.caption(f"{hidden_count} additional column(s) hidden in preview.")
    if raw_text is not None:
        with st.expander(f"Show raw text: {filename}", expanded=False):
            st.text_area(
                "Raw text",
                value=raw_text,
                height=220,
                key=f"{key_prefix}_raw_{filename}",
            )
def example_column_meanings(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Return visible example-file column explanations for present columns."""
    _, meanings, _ = EXAMPLE_FILE_NOTES[path]
    rows = []
    for column in columns:
        meaning = meanings.get(column, "optional metadata column if present")
        rows.append({"Column": column, "Meaning": meaning})
    return pd.DataFrame(rows)


def render_example_file_preview(path: Path) -> None:
    """Show a compact public example CSV card with optional previews."""
    label, _, note = EXAMPLE_FILE_NOTES[path]
    data = path.read_bytes()
    raw_text = data.decode("utf-8")
    frame = pd.read_csv(path).fillna("")
    render_output_artifact_card(
        label,
        frame,
        path.name,
        (
            f"File purpose: {note} Use this file as a template for preparing "
            "your own input."
        ),
        data=data,
        preview_rows=10,
        key_prefix=f"example_{path.stem}",
        file_size=format_file_size(path),
        raw_text=raw_text,
    )
    with st.expander("Column guide", expanded=False):
        st.dataframe(
            example_column_meanings(path, frame.columns),
            width="stretch",
            hide_index=True,
            height=240,
        )
    return
    row_word = "row" if len(frame) == 1 else "rows"
    st.markdown(f"#### {label}")
    st.markdown(f"**File purpose:** {note}")
    st.markdown("Use this file as a template for preparing your own input.")
    st.write(
        f"{len(frame)} {row_word} · {format_file_size(path)} · "
        f"{len(frame.columns)} columns"
    )
    st.markdown("**Required/important columns**")
    st.dataframe(
        example_column_meanings(path, frame.columns),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download CSV",
        data=data,
        file_name=path.name,
        mime="text/csv",
        key=f"download_{path.stem}",
    )
    with st.expander(f"Show raw CSV text: {path.name}"):
        st.text_area(
            "Raw CSV text",
            value=raw_text,
            height=240,
            key=f"raw_csv_{path.stem}",
        )


def create_public_demo_workflow(
    *,
    timestamp: datetime | None = None,
) -> PipelinePaths:
    """Create a fresh public-demo workspace without running calculations."""
    active_time = timestamp or datetime.now()
    run_dir = APP_RUNS_DIR / active_time.strftime("public_demo_%Y%m%d_%H%M%S")
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return build_paths(
        input_path=DEMO_INPUT,
        references_path=DEMO_REFERENCES,
        text_evidence_path=DEMO_TEXT_EVIDENCE,
        output_dir=output_dir,
    )


def pubchem_preflight(
    *,
    client: object | None = None,
    timeout: float = PUBCHEM_PREFLIGHT_TIMEOUT,
) -> OnlineLookupPreflight:
    """Test PubChem's aspirin CID endpoint and retain visible diagnostics."""
    active_client = client or UrllibIdentityClient()
    try:
        response = active_client.get_text(PUBCHEM_PREFLIGHT_URL, timeout)
        if response.strip() != "2244":
            raise ValueError(f"Unexpected PubChem response: {response.strip()!r}")
        return OnlineLookupPreflight(
            available=True,
            python_executable=sys.executable,
            url=PUBCHEM_PREFLIGHT_URL,
        )
    except Exception as exc:
        return OnlineLookupPreflight(
            available=False,
            python_executable=sys.executable,
            url=PUBCHEM_PREFLIGHT_URL,
            exception_type=type(exc).__name__,
            exception_message=" ".join(str(exc).split())[:300],
        )


def environment_model_check_row(check_type: str, selection) -> dict[str, str]:
    """Return one compact model availability check row."""
    name = selection.model_id or selection.label
    try:
        load_optional_model(selection)
    except DomainModelUnavailableError as exc:
        return {
            "Check type": check_type,
            "Name or endpoint": name,
            "Status": "Not available",
            "Notes": str(exc),
        }
    return {
        "Check type": check_type,
        "Name or endpoint": name,
        "Status": "Available",
        "Notes": "Model loaded from local cache/environment.",
    }


def run_environment_checks() -> list[dict[str, str]]:
    """Run online lookup and embedding model availability checks."""
    online = pubchem_preflight()
    rows = [
        {
            "Check type": "Online lookup",
            "Name or endpoint": online.url,
            "Status": "Available" if online.available else "Not available",
            "Notes": (
                "PubChem aspirin CID endpoint returned 2244."
                if online.available
                else pubchem_preflight_failure_message(online)
            ),
        }
    ]
    biomedical = preferred_model_selection("biomedical")
    patent = preferred_model_selection("patent")
    fallback_biomedical = resolve_model_selection(
        model_type="biomedical",
        option=CLOUD_SAFE_FALLBACK_LABEL,
        fallback_model_id=FALLBACK_MODEL_ID,
    )
    rows.append(environment_model_check_row("Biomedical model", biomedical))
    rows.append(environment_model_check_row("Patent model", patent))
    rows.append(environment_model_check_row("Lightweight fallback", fallback_biomedical))
    st.session_state["environment_check_result"] = rows
    return rows

def render_environment_check_rows(rows: list[dict[str, str]]) -> None:
    """Render compact environment check table while preserving preflight messages."""
    if not rows:
        return
    available = sum(1 for row in rows if row.get("Status") == "Available")
    render_status_badge(
        f"{available} of {len(rows)} checks available",
        status="success" if available == len(rows) else "warning",
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    for row in rows:
        if row.get("Check type") == "Online lookup":
            if row.get("Status") == "Available":
                st.success(f"Online database lookup is available. {row.get('Notes')}")
            else:
                st.error(str(row.get("Notes", "")))
                st.info(ONLINE_LOOKUP_RESTART_MESSAGE)
            continue
        if row.get("Status") != "Available":
            st.warning(f"{row.get('Check type')}: {row.get('Notes')}")

def pubchem_preflight_failure_message(result: OnlineLookupPreflight) -> str:
    """Return browser-visible process and exception diagnostics."""
    return (
        f"{ONLINE_LOOKUP_UNAVAILABLE_MESSAGE}\n\n"
        f"Python executable: {result.python_executable}\n\n"
        f"PubChem URL: {result.url}\n\n"
        f"Exception type: {result.exception_type or 'Unknown'}\n\n"
        f"Exception message: {result.exception_message or 'No message returned.'}"
    )


def render_pubchem_preflight_result(result: OnlineLookupPreflight) -> None:
    """Show a connection-test result without creating workflow outputs."""
    if result.available:
        st.success(
            "Online database lookup is available from this process. "
            f"PubChem aspirin CID endpoint returned 2244.\n\n"
            f"Python executable: {result.python_executable}\n\n"
            f"PubChem URL: {result.url}"
        )
        return
    st.error(pubchem_preflight_failure_message(result))
    st.info(ONLINE_LOOKUP_RESTART_MESSAGE)


def demo_paths_from_output(output_dir: Path) -> PipelinePaths:
    """Rebuild public-demo pipeline paths for a tutorial workspace."""
    return build_paths(
        input_path=DEMO_INPUT,
        references_path=DEMO_REFERENCES,
        text_evidence_path=DEMO_TEXT_EVIDENCE,
        output_dir=output_dir,
    )


def step3_outputs_exist(paths: PipelinePaths) -> bool:
    """Return whether both guided Step 3 output files already exist."""
    return paths.public_lookup.is_file() and paths.surechembl_lookup.is_file()


def step3_summary(
    public_lookup: pd.DataFrame,
    surechembl: pd.DataFrame,
    standardized: pd.DataFrame,
) -> dict[str, int]:
    """Summarize completed Step 3 database statuses."""
    def source_counts(source: str) -> dict[str, int]:
        if public_lookup.empty or "source_database" not in public_lookup.columns:
            return {}
        rows = public_lookup[
            public_lookup["source_database"].fillna("").astype(str).eq(source)
        ]
        return (
            rows.get("lookup_status", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .value_counts()
            .to_dict()
        )

    pubchem = source_counts("PubChem")
    chembl = source_counts("ChEMBL")
    sure_status = (
        surechembl.get("lookup_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
    )
    sure_queried = surechembl[
        sure_status.isin({"match_found", "no_match", "lookup_error"})
    ]
    valid = (
        standardized.get("valid_smiles", pd.Series(dtype=object))
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    return {
        "PubChem matches": int(pubchem.get("match_found", 0)),
        "PubChem no matches": int(pubchem.get("no_match", 0)),
        "PubChem errors": int(pubchem.get("lookup_error", 0)),
        "ChEMBL matches": int(chembl.get("match_found", 0)),
        "ChEMBL no matches": int(chembl.get("no_match", 0)),
        "ChEMBL errors": int(chembl.get("lookup_error", 0)),
        "SureChEMBL queried": int(
            sure_queried.get("molecule_id", pd.Series(dtype=str)).nunique()
        ),
        "SureChEMBL errors": int(
            surechembl[sure_status.eq("lookup_error")]
            .get("molecule_id", pd.Series(dtype=str))
            .nunique()
        ),
        "Invalid molecules skipped": int((~valid).sum()),
    }


def run_public_demo_step3(
    paths: PipelinePaths,
    *,
    progress_callback: object | None = None,
    public_client: object | None = None,
    surechembl_client: object | None = None,
) -> dict[str, int]:
    """Run guided Step 3 with molecule-level progress and atomic outputs."""
    standardized = pd.read_csv(paths.standardized).fillna("")
    required = {"molecule_id", "canonical_smiles", "inchi_key", "valid_smiles"}
    missing = required - set(standardized.columns)
    if missing:
        raise ValueError(
            "Standardized CSV is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    total = len(standardized)
    started = time.monotonic()
    output_dir = paths.public_lookup.parent
    public_temp = paths.public_lookup.with_suffix(".tmp.csv")
    surechembl_temp = paths.surechembl_lookup.with_suffix(".tmp.csv")
    active_public_client = public_client or UrllibJsonClient()
    active_surechembl_client = surechembl_client or UrllibSurechemblClient()
    public_results = []
    surechembl_results = []

    def update(database: str, completed: int) -> None:
        if progress_callback is not None:
            progress_callback(
                Step3Progress(
                    database=database,
                    completed=completed,
                    total=total,
                    elapsed_seconds=time.monotonic() - started,
                    output_dir=output_dir,
                )
            )

    try:
        records = [
            {key: str(value).strip() for key, value in row.items()}
            for _, row in standardized.iterrows()
        ]
        valid_rows = [
            row
            for row in records
            if row["valid_smiles"].lower() in {"true", "1", "yes", "y"}
            and row["canonical_smiles"]
        ]

        for index, row in enumerate(records, start=1):
            valid = row["valid_smiles"].lower() in {"true", "1", "yes", "y"}
            if not valid or not row["canonical_smiles"]:
                public_results.append(
                    placeholder_result(
                        row["molecule_id"],
                        row["canonical_smiles"],
                        row["inchi_key"],
                        False,
                        source_database="not_available",
                        match_type="no_match",
                        lookup_status="invalid_molecule",
                        evidence_note=(
                            "Public lookup was skipped for an invalid molecule."
                        ),
                        error_message=row.get("error_message", ""),
                    )
                )
                update("PubChem", index)
                continue
            public_results.append(
                lookup_pubchem(
                    row["molecule_id"],
                    row["canonical_smiles"],
                    row["inchi_key"],
                    active_public_client,
                    15.0,
                )
            )
            update("PubChem", index)
        pubchem_rows = [
            item for item in public_results if item.source_database == "PubChem"
        ]

        for index, row in enumerate(records, start=1):
            valid = row["valid_smiles"].lower() in {"true", "1", "yes", "y"}
            if not valid or not row["canonical_smiles"]:
                update("ChEMBL", index)
                continue
            public_results.append(
                lookup_chembl(
                    row["molecule_id"],
                    row["canonical_smiles"],
                    row["inchi_key"],
                    active_public_client,
                    15.0,
                )
            )
            update("ChEMBL", index)
        chembl_rows = [
            item for item in public_results if item.source_database == "ChEMBL"
        ]

        for index, row in enumerate(records, start=1):
            hits = lookup_online_rows(
                [row],
                top_k=5,
                max_molecules=None,
                client=active_surechembl_client,
            )
            surechembl_results.extend(hits)
            update("SureChEMBL", index)
        surechembl_valid_results = [
            item for item in surechembl_results if item.valid_smiles
        ]
        if not public_results and not surechembl_results:
            raise RuntimeError("No Step 3 lookup rows could be produced.")

        degraded_sources = step3_degraded_sources(
            pubchem_rows,
            chembl_rows,
            surechembl_valid_results,
        )

        write_public_lookup_csv(public_temp, public_results)
        write_surechembl_output_csv(surechembl_temp, surechembl_results)
        public_temp.replace(paths.public_lookup)
        surechembl_temp.replace(paths.surechembl_lookup)
    except Exception:
        public_temp.unlink(missing_ok=True)
        surechembl_temp.unlink(missing_ok=True)
        raise

    return step3_summary_with_completion(
        pd.read_csv(paths.public_lookup),
        pd.read_csv(paths.surechembl_lookup),
        standardized,
    )


def step3_degraded_sources(
    pubchem_rows: list[object],
    chembl_rows: list[object],
    surechembl_valid_results: list[object],
) -> tuple[str, ...]:
    """Return source names whose valid Step 3 lookups all ended in errors."""
    degraded = []
    for source_name, rows in (
        ("PubChem", pubchem_rows),
        ("ChEMBL", chembl_rows),
        ("SureChEMBL", surechembl_valid_results),
    ):
        if rows and all(
            getattr(item, "lookup_status", "") == "lookup_error" for item in rows
        ):
            degraded.append(source_name)
    return tuple(degraded)


def add_step3_completion_metadata(
    summary: dict[str, int],
    degraded_sources: tuple[str, ...],
) -> dict[str, object]:
    """Attach non-CSV UI metadata describing Step 3 evidence completeness."""
    result: dict[str, object] = dict(summary)
    if degraded_sources:
        completion_status = "degraded_lookup"
    elif summary.get("PubChem errors", 0) or summary.get("ChEMBL errors", 0) or summary.get("SureChEMBL errors", 0):
        completion_status = "partial_evidence"
    else:
        completion_status = "full_evidence"
    result["__completion_status"] = completion_status
    result["__degraded_sources"] = ", ".join(degraded_sources)
    result["__chembl_unavailable"] = "ChEMBL" in degraded_sources
    return result


def step3_dataframe_degraded_sources(
    public_lookup: pd.DataFrame,
    surechembl: pd.DataFrame,
) -> tuple[str, ...]:
    """Infer degraded Step 3 sources from activated output CSV statuses."""
    degraded = []
    if {"source_database", "lookup_status"}.issubset(public_lookup.columns):
        for source_name in ("PubChem", "ChEMBL"):
            rows = public_lookup[public_lookup["source_database"].astype(str).eq(source_name)]
            statuses = rows["lookup_status"].astype(str) if not rows.empty else pd.Series(dtype=str)
            if not statuses.empty and statuses.eq("lookup_error").all():
                degraded.append(source_name)
    if "lookup_status" in surechembl.columns:
        sure_rows = surechembl
        if "valid_smiles" in sure_rows.columns:
            valid = sure_rows["valid_smiles"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
            sure_rows = sure_rows[valid]
        statuses = sure_rows["lookup_status"].astype(str) if not sure_rows.empty else pd.Series(dtype=str)
        if not statuses.empty and statuses.eq("lookup_error").all():
            degraded.append("SureChEMBL")
    return tuple(degraded)


def step3_summary_with_completion(
    public_lookup: pd.DataFrame,
    surechembl: pd.DataFrame,
    standardized: pd.DataFrame,
) -> dict[str, object]:
    """Build Step 3 counts plus full/partial/degraded completion metadata."""
    return add_step3_completion_metadata(
        step3_summary(public_lookup, surechembl, standardized),
        step3_dataframe_degraded_sources(public_lookup, surechembl),
    )


def step3_completion_label(summary: dict[str, object]) -> str:
    """Return a user-facing Step 3 completion label."""
    status = str(summary.get("__completion_status", "full_evidence"))
    labels = {
        "full_evidence": "full evidence",
        "partial_evidence": "partial evidence",
        "degraded_lookup": "degraded lookup",
    }
    return labels.get(status, "completed")


def render_step3_summary(summary: dict[str, object]) -> None:
    """Show the requested Step 3 completion counts and degraded lookup warnings."""
    st.success(f"Step 3 public database lookup completed with {step3_completion_label(summary)}.")
    if summary.get("__chembl_unavailable"):
        st.warning(STEP3_CHEMBL_UNAVAILABLE_WARNING)
        render_warning_card("ChEMBL unavailable", STEP3_CHEMBL_WARNING_CARD)
    elif summary.get("__completion_status") == "degraded_lookup":
        degraded = summary.get("__degraded_sources")
        detail = (
            f" Affected service(s): {degraded}." if degraded else ""
        )
        st.warning(STEP3_DEGRADED_LOOKUP_WARNING + detail)

    display_summary = {
        key: value
        for key, value in summary.items()
        if key not in STEP3_SUMMARY_METADATA_KEYS
    }
    st.dataframe(
        pd.DataFrame(
            {
                "Result": list(display_summary),
                "Molecules": list(display_summary.values()),
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_step3_progress(progress: Step3Progress, bar: object, status: object) -> None:
    """Update the visible Step 3 database, molecule, time, and output details."""
    fraction = progress.completed / progress.total if progress.total else 1.0
    bar.progress(
        fraction,
        text=(
            f"{progress.database}: {progress.completed} / "
            f"{progress.total} molecules"
        ),
    )
    status.markdown(
        f"**Current database:** {progress.database}  \n"
        f"**Completed molecules:** {progress.completed} / {progress.total}  \n"
        f"**Elapsed time:** {progress.elapsed_seconds:.1f} seconds  \n"
        f"**Active output folder:** `{progress.output_dir}`"
    )


def optional_domain_model_state(model_type: str) -> dict[str, str]:
    """Return preferred local model settings from session state."""
    if model_type == "biomedical":
        return {
            "option": str(st.session_state.get("biomedical_domain_model_option", CLOUD_SAFE_FALLBACK_LABEL)),
            "custom_model_id": str(st.session_state.get("biomedical_custom_model_id", "")),
        }
    return {
        "option": str(st.session_state.get("patent_domain_model_option", CLOUD_SAFE_FALLBACK_LABEL)),
        "custom_model_id": str(st.session_state.get("patent_custom_model_id", "")),
    }


def preferred_model_selection(model_type: str):
    """Resolve preferred model selection for Step 6 or Step 7."""
    state = optional_domain_model_state(model_type)
    return resolve_model_selection(
        model_type=model_type,
        option=state.get("option", CLOUD_SAFE_FALLBACK_LABEL),
        custom_model_id=state.get("custom_model_id", ""),
        fallback_model_id=FALLBACK_MODEL_ID,
    )


def model_metadata_for_status(
    *,
    preferred_model: str,
    fallback_model: str = FALLBACK_MODEL_ID,
    actual_model: str = "",
    embedding_backend: str = "",
    pooling_method: str = "",
) -> dict[str, str]:
    """Return common model provenance columns for Step 6 and Step 7 outputs."""
    return {
        "embedding_backend": embedding_backend,
        "pooling_method": pooling_method,
        "model_source": actual_model,
        "preferred_model_name": preferred_model,
        "fallback_model_name": fallback_model,
        "actual_model_used": actual_model,
    }


def load_selected_domain_model(
    *,
    model_type: str,
    fallback_model_id: str = FALLBACK_MODEL_ID,
) -> tuple[object | None, str, str, dict[str, str], str]:
    """Load preferred model, then fallback model, returning safe output metadata."""
    preferred = preferred_model_selection(model_type)
    preferred_name = preferred.model_id or preferred.label or fallback_model_id
    fallback_selection = resolve_model_selection(
        model_type=model_type,
        option=CLOUD_SAFE_FALLBACK_LABEL,
        fallback_model_id=fallback_model_id,
    )
    fallback_name = fallback_selection.model_id
    if preferred.label == CLOUD_SAFE_FALLBACK_LABEL:
        try:
            model = load_optional_model(fallback_selection)
        except DomainModelUnavailableError as exc:
            metadata = model_metadata_for_status(
                preferred_model=fallback_name,
                fallback_model=fallback_name,
                actual_model="",
                embedding_backend=fallback_selection.embedding_backend,
                pooling_method=fallback_selection.pooling_method,
            )
            return None, fallback_name, "model_unavailable", metadata, (
                "Lightweight general-purpose fallback model was unavailable; "
                "embedding evidence was skipped."
            )
        metadata = encoder_metadata(model, model_source=fallback_name)
        metadata.update(
            model_metadata_for_status(
                preferred_model=fallback_name,
                fallback_model=fallback_name,
                actual_model=fallback_name,
                embedding_backend=metadata.get("embedding_backend", ""),
                pooling_method=metadata.get("pooling_method", ""),
            )
        )
        return model, fallback_name, "fallback_model_used", metadata, (
            "Lightweight general-purpose fallback model was used; this is not an error."
        )

    try:
        model = load_optional_model(preferred)
    except DomainModelUnavailableError:
        try:
            fallback_model = load_optional_model(fallback_selection)
        except DomainModelUnavailableError:
            metadata = model_metadata_for_status(
                preferred_model=preferred_name,
                fallback_model=fallback_name,
                actual_model="",
                embedding_backend=preferred.embedding_backend,
                pooling_method=preferred.pooling_method,
            )
            return None, preferred_name, "model_unavailable", metadata, (
                "Preferred model and lightweight general-purpose fallback were unavailable; "
                "embedding evidence was skipped."
            )
        metadata = encoder_metadata(fallback_model, model_source=fallback_name)
        metadata.update(
            model_metadata_for_status(
                preferred_model=preferred_name,
                fallback_model=fallback_name,
                actual_model=fallback_name,
                embedding_backend=metadata.get("embedding_backend", ""),
                pooling_method=metadata.get("pooling_method", ""),
            )
        )
        return fallback_model, fallback_name, "fallback_model_used", metadata, (
            "Preferred model was unavailable; lightweight general-purpose fallback was used."
        )

    metadata = encoder_metadata(model, model_source=preferred.model_id)
    metadata.update(
        model_metadata_for_status(
            preferred_model=preferred_name,
            fallback_model=fallback_name,
            actual_model=preferred.model_id,
            embedding_backend=metadata.get("embedding_backend", ""),
            pooling_method=metadata.get("pooling_method", ""),
        )
    )
    return model, preferred.model_id, "preferred_model_used", metadata, (
        "Preferred embedding model was used for local evidence ranking."
    )

def run_public_demo_step(
    step_number: int,
    paths: PipelinePaths,
) -> None:
    """Run exactly one public-demo workflow stage."""
    identity_path = paths.chemical_identity or paths.prioritized.parent / "chemical_identity.csv"
    context_path = paths.compound_context or paths.prioritized.parent / "compound_context.csv"

    if step_number == 1:
        standardize_csv(paths.generated_smiles, paths.standardized)
    elif step_number == 2:
        chemical_identity_csv(
            paths.standardized,
            identity_path,
            online=True,
            max_molecules=GUIDED_EXAMPLE_MAX_MOLECULES,
        )
    elif step_number == 3:
        run_public_demo_step3(paths)
    elif step_number == 4:
        descriptor_csv(paths.standardized, paths.descriptors)
        similarity_csv(paths.descriptors, paths.references, paths.similarity)
        top_hits_csv(
            paths.descriptors,
            paths.references,
            paths.similarity_top_hits,
            5,
        )
    elif step_number == 5:
        chemberta_embeddings_csv(
            paths.standardized,
            paths.chemberta_embeddings,
        )
        visualization_coordinates_csv(
            paths.chemberta_embeddings,
            None,
            paths.visualization_coordinates,
            reference_path=paths.references,
        )
    elif step_number == 6:
        compound_context_csv(
            paths.descriptors,
            paths.public_lookup,
            paths.similarity_top_hits,
            paths.references,
            context_path,
            identity_path=identity_path,
        )
        text_nlp_csv(
            paths.text_evidence,
            paths.text_nlp,
            context_path=context_path,
            molecule_path=paths.generated_smiles,
            descriptor_path=paths.descriptors,
            identity_path=identity_path,
        )
        biomedical_model, biomedical_model_name, biomedical_model_status, biomedical_metadata, biomedical_note = load_selected_domain_model(
            model_type="biomedical",
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        biomedical_evidence_csv(
            context_path,
            paths.text_evidence,
            paths.biomedical_evidence,
            model=biomedical_model,
            model_name=biomedical_model_name,
            identity_path=identity_path,
            descriptor_path=paths.descriptors,
            unavailable_status=biomedical_model_status,
            unavailable_metadata=biomedical_metadata if biomedical_model is None else None,
            unavailable_note=biomedical_note,
            model_status=biomedical_model_status,
            available_note=biomedical_note,
            model_metadata=biomedical_metadata,
        )
    elif step_number == 7:
        patent_model, patent_model_name, patent_model_status, patent_metadata, patent_note = load_selected_domain_model(
            model_type="patent",
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        patent_evidence_embeddings_csv(
            paths.surechembl_lookup,
            paths.patent_evidence_embeddings,
            public_lookup_path=paths.public_lookup,
            identity_path=identity_path,
            context_path=context_path,
            model=patent_model,
            model_name=patent_model_name,
            unavailable_status=patent_model_status,
            unavailable_metadata=patent_metadata if patent_model is None else None,
            unavailable_note=patent_note,
            model_status=patent_model_status,
            available_note=patent_note,
            model_metadata=patent_metadata,
        )
    elif step_number == 8:
        scoring_csv(
            paths.descriptors,
            paths.similarity,
            paths.prioritized,
            nlp_path=paths.text_nlp,
            public_lookup_path=paths.public_lookup,
            surechembl_path=paths.surechembl_lookup,
            identity_path=identity_path,
            context_path=context_path,
            chemberta_path=paths.chemberta_embeddings,
            nlp_was_run=csv_has_data_rows(paths.text_evidence),
        )
        merge_chemberta_into_prioritized(
            paths.prioritized,
            paths.chemberta_embeddings,
        )
        visualization_coordinates_csv(
            paths.chemberta_embeddings,
            paths.prioritized,
            paths.visualization_coordinates,
            reference_path=paths.references,
        )
    elif step_number == 9:
        loaded = load_output_directory(paths.prioritized.parent)
        prioritization = loaded.tables["prioritization"]
        score = score_column(prioritization)
        ranked = prioritization.sort_values(score, ascending=False).head(5)
        for molecule_id in ranked["molecule_id"].astype(str):
            generate_report(loaded, molecule_id)
    else:
        raise ValueError(f"Unsupported workflow step: {step_number}")


def render_public_demo_choice() -> Path | None:
    """Start a fresh tutorial without running any pipeline stage."""
    st.header("Guided example workflow")
    st.write(
        "Learn the evaluation process one stage at a time. Each step explains "
        "what is calculated, why it matters, and what evidence it produces."
    )
    render_section_header(
        "Public example files",
        "Use these files as templates for preparing your own generated SMILES input.",
        level=3,
    )
    for path in public_demo_input_paths():
        render_example_file_preview(path)
    environment_rows = []
    if st.button("Run environment checks"):
        environment_rows = run_environment_checks()
    else:
        environment_rows = st.session_state.get("environment_check_result", [])
    render_section_header(
        "Run environment checks",
        "Check local package, online lookup, and optional model availability without starting a workflow.",
        level=3,
    )
    render_environment_check_rows(environment_rows)
    if not st.button("Run guided example workflow", type="primary"):
        return None
    preflight = pubchem_preflight()
    if not preflight.available:
        render_pubchem_preflight_result(preflight)
    try:
        paths = create_public_demo_workflow()
    except Exception as exc:
        st.error(f"Could not start the guided example: {exc}")
        return None
    output_dir = paths.prioritized.parent
    st.session_state["active_output_dir"] = str(output_dir)
    st.session_state["workflow_step"] = 1
    st.session_state["completed_workflow_steps"] = []
    st.session_state["workflow_mode"] = "public_demo"
    st.success("Guided example ready. Step 1 has not run yet.")
    return output_dir


def render_load_existing_choice() -> Path | None:
    """Activate an existing result folder only after the user clicks load."""
    st.header("Load existing results")
    st.info(LOAD_EXISTING_NOTE)
    output_dir_text = st.text_input(
        "Output directory",
        value="",
        placeholder="outputs/my-analysis",
    )
    if not st.button("Load results", type="primary"):
        return None
    if not output_dir_text.strip():
        st.error("Enter an output directory before loading results.")
        return None
    output_dir = Path(output_dir_text.strip())
    if not (output_dir / OUTPUT_FILES["prioritization"]).is_file():
        st.error(
            "No prioritization_results.csv was found in the selected output directory."
        )
        return None
    st.session_state["active_output_dir"] = str(output_dir)
    st.session_state["workflow_step"] = 1
    st.session_state["completed_workflow_steps"] = list(
        range(1, len(WORKFLOW_STEP_NAMES) + 1)
    )
    st.session_state["workflow_mode"] = "existing_results"
    return output_dir


def active_output_directory() -> Path | None:
    """Return the explicitly activated result folder, if any."""
    value = str(st.session_state.get("active_output_dir", "")).strip()
    return Path(value) if value else None


def available_columns(
    frame: pd.DataFrame, columns: Iterable[str]
) -> list[str]:
    """Return requested columns that exist in a result table."""
    return [column for column in columns if column in frame.columns]


def columns_with_available_values(
    frame: pd.DataFrame, columns: Iterable[str]
) -> list[str]:
    """Return columns that exist and contain at least one useful value."""
    available = []
    unavailable = {"", "not_available", "nan", "none"}
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip().str.lower()
        if values.map(lambda value: value not in unavailable).any():
            available.append(column)
    return available


def show_step_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    rows: int = 8,
) -> None:
    """Show a compact, readable result table for a workflow stage."""
    selected = available_columns(frame, columns)
    if frame.empty or not selected:
        st.info("This output is not available for the selected run.")
        return
    st.dataframe(
        display_dataframe(frame[selected].head(rows)),
        width="stretch",
        hide_index=True,
    )


def status_bar_chart(
    frame: pd.DataFrame, column: str, title: str
) -> None:
    """Render a count chart for a categorical result column."""
    if frame.empty or column not in frame.columns:
        return
    counts = (
        frame[column]
        .fillna("not available")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    figure = px.bar(
        counts,
        x=column,
        y="count",
        title=title,
        labels={column: display_label(column), "count": "Molecules"},
    )
    st.plotly_chart(figure, width="stretch")


def surechembl_detail_columns(sure_rows: pd.DataFrame) -> list[str]:
    """Return safe SureChEMBL detail columns, hiding empty patent metadata."""
    structure_columns = available_columns(
        sure_rows,
        (
            "molecule_id",
            "compound_name",
            "tanimoto_similarity",
            "similarity_category",
        ),
    )
    patent_columns = columns_with_available_values(
        sure_rows,
        (
            "patent_metadata_status",
            "patent_section",
            "patent_id",
            "patent_number",
            "patent_title",
            "patent_date",
            "patent_metadata_source",
        ),
    )
    trailing = available_columns(sure_rows, ("evidence_note",))
    if (
        "error_message" in sure_rows.columns
        and sure_rows["error_message"].fillna("").astype(str).str.strip().ne("").any()
    ):
        trailing.append("error_message")
    return structure_columns + patent_columns + trailing


def surechembl_summary_counts(sure_rows: pd.DataFrame) -> dict[str, int]:
    """Return headline SureChEMBL structure and metadata counts."""
    lookup = (
        sure_rows.get("lookup_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    metadata = (
        sure_rows.get("patent_metadata_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return {
        "Structure matches found": int(lookup.eq("match_found").sum()),
        "Document metadata found": int(metadata.eq("found").sum()),
        "Structure-only matches": int(metadata.eq("structure_match_only").sum()),
        "Lookup errors": int(lookup.eq("lookup_error").sum()),
    }


def render_surechembl_summary_cards(sure_rows: pd.DataFrame) -> None:
    """Show compact SureChEMBL evidence counts."""
    counts = surechembl_summary_counts(sure_rows)
    columns = st.columns(len(counts))
    for column, (label, value) in zip(columns, counts.items()):
        column.metric(label, value)


def readable_surechembl_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return SureChEMBL rows with safe, readable display values."""
    result = readable_ui_dataframe(frame)
    for column in ("similarity_category", "patent_metadata_status"):
        if column in result.columns:
            result[column] = result[column].map(readable_status)
    for column in ("patent_section", "patent_metadata_source"):
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: (
                    "SureChEMBL structure lookup"
                    if str(value or "").strip() == "SureChEMBL API"
                    else readable_status(value)
                )
            )
    return result


def render_surechembl_public_structure_evidence(sure_rows: pd.DataFrame) -> None:
    """Render SureChEMBL structure matches and optional patent metadata."""
    st.caption("SureChEMBL public structure evidence")
    render_surechembl_summary_cards(sure_rows)
    statuses = (
        sure_rows.get("patent_metadata_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    if statuses.eq("found").any():
        st.success("Patent document metadata found")
    elif (
        statuses.eq("structure_match_only").any()
        or sure_rows.get("lookup_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("match_found")
        .any()
    ):
        st.info(
            "Structure match found, but patent document metadata was not retrieved"
        )
        st.info(
            "SureChEMBL returned public structure matches. Patent document metadata "
            "was not returned by the current lookup, so this is structure-level "
            "public evidence only."
        )
    columns = surechembl_detail_columns(sure_rows)
    if not columns:
        st.info("SureChEMBL public structure evidence is unavailable.")
        return
    st.dataframe(
        display_dataframe(readable_surechembl_dataframe(sure_rows[columns])),
        width="stretch",
        hide_index=True,
    )


def artifact_display_name(item: Path | str) -> str:
    """Return a public-safe workflow artifact label without local directories."""
    if isinstance(item, Path):
        return item.name
    text = str(item)
    if "/" in text or "\\" in text:
        return Path(text).name
    return text


def render_artifact_name_list(items: Iterable[Path | str]) -> None:
    """Show artifact names without code-style copy boxes."""
    for item in items:
        st.markdown(f"- {artifact_display_name(item)}")


def render_csv_output_artifact(path: Path) -> None:
    """Show one workflow CSV output as a compact artifact card."""
    if not path.exists():
        st.markdown(f"#### {path.name}")
        st.info("Run this step to create the output file.")
        return

    data = path.read_bytes()
    raw_text = data.decode("utf-8", errors="replace")
    try:
        frame = pd.read_csv(path).fillna("")
    except Exception as exc:
        st.markdown(f"#### {path.name}")
        st.error(f"Could not preview {path.name}: {exc}")
        st.download_button(
            f"Download {path.name}",
            data=data,
            file_name=path.name,
            mime="text/csv",
            key=f"download_output_{path.name}",
        )
        with st.expander(f"Show raw CSV text: {path.name}"):
            st.text_area(
                "Raw CSV text",
                value=raw_text,
                height=240,
                key=f"raw_output_{path.name}",
            )
        return

    render_output_artifact_card(
        path.name,
        frame,
        path.name,
        "Generated workflow CSV output.",
        data=data,
        preview_rows=10,
        key_prefix=f"output_{path.stem}",
        file_size=format_file_size(path),
        raw_text=raw_text,
    )
    return
    row_word = "row" if len(frame) == 1 else "rows"
    st.write(f"{len(frame)} {row_word} · {format_file_size(path)}")
    st.dataframe(frame.head(10), width="stretch", hide_index=True)
    st.download_button(
        f"Download {path.name}",
        data=data,
        file_name=path.name,
        mime="text/csv",
        key=f"download_output_{path.name}",
    )
    with st.expander(f"Show raw CSV text: {path.name}"):
        st.text_area(
            "Raw CSV text",
            value=raw_text,
            height=240,
            key=f"raw_output_{path.name}",
        )


def render_reports_output_artifact(reports_dir: Path) -> None:
    """Show generated report artifacts for Step 9, when present."""
    st.markdown(f"#### {reports_dir.name}")
    render_reports_browser(
        reports_dir=reports_dir,
        key_prefix=f"output_artifact_reports_{reports_dir.name}",
    )


def report_molecule_id(path: Path) -> str:
    """Extract a molecule ID from a generated report filename."""
    prefix = "compound_intelligence_report_"
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def first_existing_report_image(images_dir: Path, molecule_id: str) -> Path | None:
    """Return a linked 2D structure image for a report, if one exists."""
    for suffix in (".png", ".jpg", ".jpeg", ".svg"):
        path = images_dir / f"{molecule_id}{suffix}"
        if path.exists():
            return path
    return None


def report_metadata_lookup(prioritization: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return optional report table metadata keyed by molecule ID."""
    if prioritization.empty or "molecule_id" not in prioritization.columns:
        return {}
    rows = {}
    for _, row in prioritization.fillna("").iterrows():
        molecule_id = str(row.get("molecule_id", "")).strip()
        if not molecule_id:
            continue
        rows[molecule_id] = {
            "Priority/design category": str(
                row.get("prioritization_category_with_nlp")
                or row.get("prioritization_category")
                or row.get("design_category")
                or ""
            ),
            "Exact public identity": str(
                row.get("exact_public_name")
                or row.get("preferred_name")
                or row.get("known_public_match")
                or ""
            ),
            "Drug-likeness category": str(row.get("druglikeness_category") or ""),
            "Text-evidence status": str(row.get("nlp_status") or ""),
        }
    return rows


def report_summary_values(
    reports: Iterable[Path], reports_dir: Path
) -> dict[str, object]:
    """Return Step 9 report summary card values."""
    report_list = list(reports)
    molecule_ids = {report_molecule_id(path) for path in report_list}
    return {
        "Reports generated": len(report_list),
        "Molecules with reports": len(molecule_ids),
        "Report folder": reports_dir,
        "Downloadable files available": len(report_list),
    }


def render_report_summary_cards(reports: list[Path], reports_dir: Path) -> None:
    """Show Step 9 report summary cards."""
    values = report_summary_values(reports, reports_dir)
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values.items()):
        if label == "Report folder":
            column.markdown(f"**{label}**")
            column.caption(str(value))
        else:
            column.metric(label, value)


def build_reports_zip(reports: Iterable[Path]) -> bytes:
    """Create an in-memory ZIP containing report Markdown files."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for report in reports:
            archive.writestr(report.name, report.read_bytes())
    return buffer.getvalue()


def report_table_dataframe(
    reports: list[Path], prioritization: pd.DataFrame
) -> pd.DataFrame:
    """Return the readable Step 9 report table without button widgets."""
    metadata = report_metadata_lookup(prioritization)
    rows = []
    for report in reports:
        molecule_id = report_molecule_id(report)
        row = {
            "Molecule ID": molecule_id,
            "Report file": report.name,
            "Download report button": f"Download {report.name}",
        }
        row.update(metadata.get(molecule_id, {}))
        rows.append(row)
    columns = [
        "Molecule ID",
        "Report file",
        "Priority/design category",
        "Exact public identity",
        "Drug-likeness category",
        "Text-evidence status",
        "Download report button",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    result = frame[columns].fillna("")
    for column in (
        "Priority/design category",
        "Drug-likeness category",
        "Text-evidence status",
    ):
        result[column] = result[column].map(readable_status)
    return display_dataframe(result)


def render_reports_browser(
    *,
    reports_dir: Path,
    prioritization: pd.DataFrame | None = None,
    images_dir: Path | None = None,
    key_prefix: str = "reports_browser",
) -> None:
    """Render Step 9 reports without an artifact-count chart."""
    reports = sorted(reports_dir.glob("compound_intelligence_report_*.md"))
    if not reports:
        st.info("Run Step 9 to generate molecule reports.")
        return

    prioritization = prioritization if prioritization is not None else pd.DataFrame()
    images_dir = images_dir or reports_dir.parent / "report_images"
    render_report_summary_cards(reports, reports_dir)
    with st.expander("Preview report table", expanded=False):
        st.dataframe(
            report_table_dataframe(reports, prioritization).head(10),
            width="stretch",
            hide_index=True,
            height=260,
        )
    st.download_button(
        "Download all reports as ZIP",
        data=build_reports_zip(reports),
        file_name="molecule_reports.zip",
        mime="application/zip",
        key=f"{key_prefix}_download_all_reports_zip",
    )
    for report in reports:
        data = report.read_bytes()
        raw_text = data.decode("utf-8", errors="replace")
        molecule_id = report_molecule_id(report)
        image_path = first_existing_report_image(images_dir, molecule_id)
        st.download_button(
            f"Download {report.name}",
            data=data,
            file_name=report.name,
            mime="text/markdown",
            key=f"{key_prefix}_download_report_{report.name}",
        )
        with st.expander(f"Preview report: {report.name}"):
            if image_path is not None:
                st.image(
                    str(image_path),
                    caption=f"2D structure: {molecule_id}",
                    width=240,
                )
            st.text_area(
                "Markdown preview",
                value=raw_text[:4000],
                height=240,
                key=f"{key_prefix}_preview_report_{report.name}",
            )


def render_output_artifacts(outputs: Iterable[Path]) -> None:
    """Show the preview/download/copy panel for workflow outputs."""
    st.markdown("#### Output files")
    for path in outputs:
        if path.suffix.lower() == ".csv":
            render_csv_output_artifact(path)
        else:
            render_reports_output_artifact(path)


def render_step_references(step_number: int) -> None:
    """Render compact documentation links for one workflow step."""
    with st.expander("References and documentation"):
        for label, url in WORKFLOW_STEP_REFERENCES[step_number - 1]:
            st.markdown(f"- [{label}]({url})")
def render_step_header(
    step_number: int,
    description: str,
    inputs: Iterable[Path | str],
    outputs: Iterable[Path],
) -> None:
    """Render shared explanation and artifact details for one workflow step."""
    step_name = WORKFLOW_STEP_NAMES[step_number - 1]
    st.header(step_name)
    st.markdown(WORKFLOW_STEP_PARAGRAPHS[step_number - 1])
    render_modern_card(
        f"Step {step_number}: {step_name}",
        WORKFLOW_STEP_PARAGRAPHS[step_number - 1],
        description=description,
        badges=(("Step ready", "secondary"),),
        key=f"workflow_step_{step_number}_header_card",
    )
    render_modern_badge(
        "Completed output available" if outputs else "No outputs listed",
        variant="secondary",
        status="success" if outputs else "warning",
        key=f"workflow_step_{step_number}_status_badge",
    )
    render_step_references(step_number)
    left, right = st.columns(2)
    with left:
        st.markdown("**Input used**")
        render_artifact_name_list(inputs)
        render_modern_step_card(
            "Prerequisites",
            inputs,
            key=f"workflow_step_{step_number}_prerequisites_card",
        )
    with right:
        st.markdown("**Output file created**")
        render_artifact_name_list(outputs)
        render_modern_step_card(
            "Outputs produced",
            outputs,
            key=f"workflow_step_{step_number}_outputs_card",
        )
    render_output_artifacts(outputs)
def render_workflow_step(
    loaded: LoadedOutputs,
    step_number: int,
    *,
    results_available: bool = True,
) -> None:
    """Render one educational workflow stage from completed local outputs."""
    output_dir = loaded.output_dir
    prioritization = loaded.tables["prioritization"]

    if step_number == 1:
        frame = loaded.tables["standardized"]
        render_step_header(
            1,
            "Checks whether each SMILES can be parsed, creates a canonical "
            "representation, and calculates an InChIKey for valid structures.",
            [DEMO_INPUT if "public_demo_" in str(output_dir) else "uploaded SMILES CSV"],
            get_step_artifacts(1, output_dir),
        )
        if not results_available:
            return
        selected = render_validation_view(frame, key="workflow_step_1")
        render_detail_panel(loaded, selected)
    elif step_number == 2:
        frame = loaded.tables["chemical_identity"]
        render_step_header(
            2,
            "Uses standardized structures to find exact public chemical names "
            "when supported and clearly labels unmatched structures.",
            [loaded.paths["standardized"]],
            get_step_artifacts(2, output_dir),
        )
        if not results_available:
            return
        selected = render_identity_view(
            frame,
            key="workflow_step_2",
            output_dir=loaded.output_dir,
            csv_path=loaded.paths["chemical_identity"],
        )
        render_detail_panel(loaded, selected)
    elif step_number == 3:
        frame = loaded.tables["public_lookup"]
        render_step_header(
            3,
            "Checks public compound databases for exact or structurally related "
            "records and records match, no-match, and query status separately.",
            [loaded.paths["standardized"], loaded.paths["chemical_identity"]],
            get_step_artifacts(3, output_dir),
        )
        if not results_available:
            return
        molecule_ids = (
            loaded.tables["standardized"]["molecule_id"].astype(str).tolist()
            if not loaded.tables["standardized"].empty
            and "molecule_id" in loaded.tables["standardized"].columns
            else ()
        )
        selected = render_public_evidence_view(
            frame,
            loaded.tables["surechembl"],
            molecule_ids,
            key="workflow_step_3",
        )
        render_step3_summary(
            step3_summary_with_completion(
                frame,
                loaded.tables["surechembl"],
                loaded.tables["standardized"],
            )
        )
        render_detail_panel(loaded, selected)
    elif step_number == 4:
        frame = loaded.tables["descriptors"]
        render_step_header(
            4,
            "Calculates interpretable RDKit properties and classifies each "
            "molecule as favorable, borderline, unfavorable, or invalid.",
            [loaded.paths["standardized"]],
            get_step_artifacts(4, output_dir),
        )
        if not results_available:
            return
        selected = render_druglikeness_views(frame, key="workflow_step_4")
        render_detail_panel(loaded, selected)
    elif step_number == 5:
        frame = loaded.tables["visualization"]
        render_step_header(
            5,
            "Places molecules in a learned chemical-space representation so "
            "clusters, close neighbors, and outliers can be explored visually.",
            [loaded.paths["standardized"]],
            get_step_artifacts(5, output_dir),
        )
        if not results_available:
            return
        selected = render_chemical_space(
            loaded, prioritization, key="workflow_step_5"
        )
        render_detail_panel(loaded, selected)
    elif step_number == 6:
        context = loaded.tables["compound_context"]
        biomedical = loaded.tables["biomedical_evidence"]
        render_step_header(
            6,
            "Connects biomedical text evidence with grounded molecule identity "
            "and biological context. The default model is a lightweight general "
            "sentence-transformer baseline; BioBERT/PubMedBERT-style biomedical "
            "models are optional advanced local/cached models. On Streamlit Cloud, "
            "biomedical embeddings may be skipped when the model is unavailable, "
            "and that skipped state is not an error.",
            [
                DEMO_TEXT_EVIDENCE
                if "public_demo_" in str(output_dir)
                else "uploaded text evidence CSV",
                loaded.paths["chemical_identity"],
                loaded.paths["public_lookup"],
            ],
            get_step_artifacts(6, output_dir),
        )
        if not results_available:
            return
        molecule_ids = (
            context["molecule_id"].astype(str).tolist()
            if not context.empty and "molecule_id" in context.columns
            else ()
        )
        text_nlp = loaded.tables["text_nlp"]
        if not text_nlp.empty:
            render_modern_evidence_card(
                "Text evidence summary",
                {
                    "Rows": len(text_nlp),
                    "Molecules": int(text_nlp["molecule_id"].nunique()) if "molecule_id" in text_nlp.columns else len(molecule_ids),
                    "Available": int(text_nlp["nlp_status"].astype(str).eq("available").sum()) if "nlp_status" in text_nlp.columns else 0,
                },
                key="workflow_step_6_text_evidence_summary_card",
            )
        selected = render_biomedical_evidence_view(
            biomedical, molecule_ids, key="workflow_step_6"
        )
        render_detail_panel(loaded, selected)
    elif step_number == 7:
        render_step_header(
            7,
            "Matches molecule IP-context summaries against public patent text "
            "signals when an optional advanced local/cached PaECTER or patent-BERT-style "
            "model is available. SureChEMBL structure evidence and patent-text "
            "embedding evidence stay separate, and patent/IP-context evidence is a "
            "research triage signal rather than a legal conclusion.",
            [
                loaded.paths["surechembl"],
                loaded.paths["public_lookup"],
                loaded.paths["chemical_identity"],
                loaded.paths["compound_context"],
            ],
            get_step_artifacts(7, output_dir),
        )
        if not results_available:
            return
        patent = loaded.tables["patent_evidence_embeddings"]
        molecule_ids = (
            loaded.tables["compound_context"]["molecule_id"].astype(str).tolist()
            if not loaded.tables["compound_context"].empty
            and "molecule_id" in loaded.tables["compound_context"].columns
            else ()
        )
        selected = render_patent_evidence_view(
            patent, molecule_ids, key="workflow_step_7"
        )
        render_detail_panel(loaded, selected)
    elif step_number == 8:
        render_step_header(
            8,
            FINAL_RANKING_EXPLANATION,
            [
                loaded.paths["chemical_identity"],
                loaded.paths["public_lookup"],
                loaded.paths["descriptors"],
                loaded.paths["visualization"],
                loaded.paths["text_nlp"],
                loaded.paths["compound_context"],
            ],
            get_step_artifacts(8, output_dir),
        )
        if not results_available:
            return
        st.info(FINAL_RANKING_EXPLANATION)
        selected = render_score_similarity(
            prioritization, key="workflow_step_8"
        )
        render_detail_panel(loaded, selected)
    else:
        render_step_header(
            9,
            "Creates molecule-level Markdown reports that bring the available "
            "identity, properties, public evidence, context, and ranking into one view.",
            [loaded.paths["prioritization"], loaded.paths["compound_context"]],
            get_step_artifacts(9, output_dir),
        )
        if not results_available:
            return
        render_reports_browser(
            reports_dir=loaded.reports_dir,
            prioritization=prioritization,
            images_dir=loaded.images_dir,
            key_prefix="workflow_step_9_reports",
        )


def render_step_workflow(output_dir: Path) -> None:
    """Explain, run, and reveal one workflow stage at a time."""
    loaded = load_output_directory(output_dir)
    current = int(st.session_state.get("workflow_step", 1))
    current = max(1, min(current, len(WORKFLOW_STEP_NAMES)))
    completed = {
        int(value)
        for value in st.session_state.get("completed_workflow_steps", [])
    }
    results_available = current in completed
    workflow_mode = st.session_state.get("workflow_mode", "")

    st.caption(f"Active output folder: {output_dir}")
    st.progress(current / len(WORKFLOW_STEP_NAMES))
    st.caption(f"Workflow progress: {current} of {len(WORKFLOW_STEP_NAMES)}")
    render_workflow_step(
        loaded,
        current,
        results_available=results_available,
    )

    if not results_available:
        missing_previous = [
            step for step in range(1, current) if step not in completed
        ]
        if missing_previous:
            required = ", ".join(f"Step {step}" for step in missing_previous)
            st.warning(f"Run required previous step first: {required}.")
            return
        if workflow_mode != "public_demo":
            st.warning("This workflow step has not been run for the selected output.")
            return
        paths = demo_paths_from_output(output_dir)
        if current == 3 and step3_outputs_exist(paths):
            if st.button(
                "Use existing Step 3 results",
                type="primary",
                key="use_existing_step_3",
            ):
                completed.add(3)
                st.session_state["completed_workflow_steps"] = sorted(completed)
                st.rerun()
            return

        run_label = f"Run Step {current} on public example"
        button_slot = st.empty()
        run_clicked = button_slot.button(
            run_label,
            type="primary",
            key=f"run_demo_step_{current}",
        )
        if run_clicked:
            st.info(
                "This step is currently running. Output will appear after completion."
            )
            button_slot.button(
                run_label,
                type="primary",
                key=f"running_demo_step_{current}",
                disabled=True,
            )
            try:
                if current == 3:
                    preflight = pubchem_preflight()
                    if not preflight.available:
                        render_pubchem_preflight_result(preflight)
                        return
                    progress_bar = st.progress(
                        0.0, text="Preparing public database lookup..."
                    )
                    progress_status = st.empty()
                    summary = run_public_demo_step3(
                        paths,
                        progress_callback=lambda progress: render_step3_progress(
                            progress,
                            progress_bar,
                            progress_status,
                        ),
                    )
                    render_step3_summary(summary)
                else:
                    with st.spinner(
                        f"Running {WORKFLOW_STEP_NAMES[current - 1]}..."
                    ):
                        run_public_demo_step(current, paths)
            except Exception as exc:
                if current == 3:
                    st.error(
                        "Step 3 online lookup failed. No new Step 3 results "
                        f"were activated. {type(exc).__name__}: {exc}"
                    )
                else:
                    st.error(f"Step {current} failed: {exc}")
                return
            completed.add(current)
            st.session_state["completed_workflow_steps"] = sorted(completed)
            st.rerun()
        else:
            st.info(
                "No calculation has run for this step yet. Review the explanation "
                "above, then run the public example when you are ready."
            )
        return

    if current < len(WORKFLOW_STEP_NAMES):
        if st.button(
            f"Continue to {WORKFLOW_STEP_NAMES[current]}",
            type="primary",
            key=f"continue_step_{current}",
        ):
            st.session_state["workflow_step"] = current + 1
            next_page = WORKFLOW_STEP_TO_NAVIGATION_LABEL[current + 1]
            st.session_state[PENDING_ACTIVE_RUN_PAGE_KEY] = next_page
            st.rerun()
    else:
        if st.button("Start over", key="start_over"):
            clear_active_run_state()
            st.rerun()


def render_results_dashboard(output_dir: Path) -> None:
    """Render plots, tables, molecule detail, and report controls."""
    with st.sidebar:
        st.caption(f"Loaded results: {output_dir}")
        if st.button("Refresh / reload"):
            st.cache_data.clear()
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
        known_filter = st.selectbox("Known public match", ["All", "True", "False"])
        top_n = st.number_input("Top-N report count", min_value=1, max_value=100, value=5)

    loaded = load_output_directory(output_dir)
    prioritization = loaded.tables["prioritization"]
    if prioritization.empty:
        st.error(f"No prioritization_results.csv found in {loaded.output_dir}")
        return

    status_filters: dict[str, list[str]] = {}
    with st.sidebar:
        for column in STATUS_COLUMNS:
            if column not in prioritization.columns:
                continue
            options = sorted(prioritization[column].dropna().astype(str).unique())
            status_filters[column] = st.multiselect(
                display_label(column),
                options=options,
                default=options,
            )

    filtered = apply_filters(
        prioritization,
        min_score=min_score,
        known_public_match=known_filter,
        status_filters=status_filters,
    )

    st.header("Run Summary")
    render_summary_cards(loaded)

    st.header("Chemical Space")
    render_chemical_space(loaded, prioritization)

    st.header("Score vs Reference Similarity")
    render_score_similarity(prioritization)

    st.header("Property Distributions")
    render_descriptor_histograms(loaded.tables["descriptors"])

    st.header("Results Table")
    results_table = display_dataframe(filtered[ordered_columns(filtered)])
    with st.expander("Preview table", expanded=False):
        st.dataframe(results_table.head(10), width="stretch", height=260)

    molecule_ids = filtered["molecule_id"].astype(str).tolist() if "molecule_id" in filtered.columns else []
    default_molecule = molecule_ids[0] if molecule_ids else ""
    typed_id = st.text_input("Select or type Molecule ID", value=default_molecule)
    if molecule_ids:
        selected_id = st.selectbox(
            "Choose from filtered molecules",
            molecule_ids,
            index=molecule_ids.index(typed_id) if typed_id in molecule_ids else 0,
        )
    else:
        selected_id = typed_id
    molecule_id = typed_id.strip() or selected_id

    render_detail_panel(loaded, molecule_id)

    st.header("Report Generation")
    col1, col2 = st.columns(2)
    if col1.button("Generate report for selected molecule", disabled=not bool(molecule_id)):
        report_path = generate_report(loaded, molecule_id)
        st.success(f"Wrote report: {report_path}")
    if col2.button("Generate reports for top N filtered molecules"):
        active_score = score_column(filtered)
        top_rows = filtered.sort_values(active_score, ascending=False).head(int(top_n))
        written = [generate_report(loaded, str(row["molecule_id"])) for _, row in top_rows.iterrows()]
        st.success(f"Wrote {len(written)} report(s) to {loaded.reports_dir}")


def render_workflow_mode_sidebar(output_dir: Path | None) -> str:
    """Render the top-level sidebar workflow-mode scaffold."""
    with st.sidebar:
        st.markdown("### Workflow mode")
        mode = st.selectbox(
            "Workflow mode",
            WORKFLOW_MODE_OPTIONS,
            index=0,
            key="sidebar_workflow_mode",
        )
        if output_dir is not None:
            render_active_run_sidebar_status(output_dir)
    return str(mode)


def active_run_folder_name(output_dir: Path) -> str:
    """Return a compact run-folder name for sidebar display."""
    return output_dir.parent.name if output_dir.name == "outputs" else output_dir.name


def render_active_run_sidebar_status(output_dir: Path) -> None:
    """Show compact active-run location details in the sidebar."""
    st.caption(f"Active run: {active_run_folder_name(output_dir)}")
    with st.expander("Full run path", expanded=False):
        st.caption(str(output_dir))
def render_start_workflow_mode() -> None:
    """Explain the available workflow modes without starting calculations."""
    st.markdown("### Choose a workflow mode")
    st.markdown(
        '<div class="ui-hero-card">'
        f'<div class="ui-hero-card__title">{_ui_text(APP_TITLE)}</div>'
        '<div class="ui-card__body">Evaluate generated molecules with structure validation, public evidence, biomedical context, patent/IP triage, and transparent report artifacts.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    badge_columns = st.columns(3)
    with badge_columns[0]:
        render_modern_badge("Cheminformatics", variant="default", status="success", key="start_badge_cheminformatics")
    with badge_columns[1]:
        render_modern_badge("Biomedical evidence", variant="secondary", status="neutral", key="start_badge_biomedical")
    with badge_columns[2]:
        render_modern_badge("Patent/IP-context triage", variant="outline", status="warning", key="start_badge_patent")

    st.write(
        "Use the sidebar Workflow mode selector to choose how you want to begin."
    )
    st.markdown("Guided demo | Analyze my molecules | Load previous run")
    workflow_columns = st.columns(3)
    with workflow_columns[0]:
        render_modern_card(
            "Guided demo",
            "Start the bundled public-safe example workflow and run each step only when you click its button.",
            description="Best for learning the evidence flow.",
            key="start_card_guided_demo",
        )
    with workflow_columns[1]:
        render_modern_card(
            "Analyze my molecules",
            "Upload generated SMILES with optional reference and text-evidence files for a new local run.",
            description="Best for your own public-safe datasets.",
            key="start_card_analyze",
        )
    with workflow_columns[2]:
        render_modern_card(
            "Load previous run",
            "Open an existing output folder without rerunning the pipeline or changing outputs.",
            description="Best for reviewing completed artifacts.",
            key="start_card_load_previous",
        )
    render_timeline_chips(
        (
            "1 Standardize",
            "2 Identity",
            "3 Public evidence",
            "4 Descriptors",
            "5 Chemical space",
            "6 Biomedical",
            "7 Patent/IP",
            "8 Prioritize",
            "9 Reports",
        )
    )
    st.info(
        "No pipeline step runs until you click an explicit run or load button."
    )
def completed_workflow_steps() -> set[int]:
    """Return completed workflow steps from session state."""
    completed = set()
    for value in st.session_state.get("completed_workflow_steps", []):
        text = str(value).strip()
        if text.isdigit():
            completed.add(int(text))
    return completed


def step_output_exists(output_dir: Path, step_number: int) -> bool:
    """Return whether a known output artifact exists for a workflow step."""
    keys = STEP_OUTPUT_KEYS.get(step_number, ())
    if step_number == 9:
        reports_dir = output_dir / "reports"
        return reports_dir.exists() and any(
            reports_dir.glob("compound_intelligence_report_*.md")
        )
    for key in keys:
        filename = OUTPUT_FILES.get(key, "")
        if filename and (output_dir / filename).exists():
            return True
    return False


def step_has_missing_prerequisite(
    output_dir: Path,
    step_number: int,
    completed: set[int],
) -> bool:
    """Return whether earlier required step outputs are absent."""
    for previous_step in range(1, step_number):
        if previous_step in completed or step_output_exists(output_dir, previous_step):
            continue
        return True
    return False


def output_csv_has_status(path: Path, status_columns: Iterable[str]) -> bool:
    """Return whether a CSV contains skipped or unavailable model statuses."""
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path).fillna("")
    except Exception:
        return False
    skipped_values = {"model_unavailable", "skipped"}
    for column in status_columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.strip().str.lower()
        if values.isin(skipped_values).any():
            return True
    return False


def optional_step_skipped(output_dir: Path, step_number: int) -> bool:
    """Return whether an optional embedding step produced a skipped fallback."""
    if step_number == 6:
        return output_csv_has_status(
            output_dir / OUTPUT_FILES["biomedical_evidence"],
            ("biomedical_model_status", "biomedical_evidence_status"),
        )
    if step_number == 7:
        return output_csv_has_status(
            output_dir / OUTPUT_FILES["patent_evidence_embeddings"],
            ("patent_model_status", "patent_evidence_status"),
        )
    return False


def step_navigation_status(output_dir: Path, step_number: int) -> str:
    """Return the sidebar status key for one workflow step."""
    completed = completed_workflow_steps()
    if optional_step_skipped(output_dir, step_number):
        return "skipped"
    if step_number in completed or step_output_exists(output_dir, step_number):
        return "completed"
    if step_has_missing_prerequisite(output_dir, step_number, completed):
        return "missing_prerequisite"
    return "not_run"


def step_navigation_display_labels(output_dir: Path) -> dict[str, str]:
    """Return display labels keyed by internal sidebar page label."""
    labels: dict[str, str] = {}
    for label in STEP_NAVIGATION_LABELS:
        step_number = STEP_NAVIGATION_TO_WORKFLOW_STEP.get(label)
        if step_number is None:
            labels[label] = label
            continue
        status = step_navigation_status(output_dir, step_number)
        labels[label] = f"{STEP_STATUS_ICONS[status]} {label}"
    return labels


def normalize_step_navigation_label(label: str, display_labels: dict[str, str]) -> str:
    """Map a visible sidebar label back to the internal page label."""
    for plain_label, display_label in display_labels.items():
        if label == display_label or label == plain_label:
            return plain_label
    for icon in STEP_STATUS_ICONS.values():
        prefix = f"{icon} "
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def ordered_step_labels(labels: Iterable[str]) -> list[str]:
    """Return labels in custom analysis step order."""
    selected = set(labels)
    return [label for label in CUSTOM_ANALYSIS_STEPS if label in selected]


def required_prerequisite_steps(selected_steps: Iterable[str]) -> list[str]:
    """Return transitive prerequisites for selected analysis steps."""
    required: set[str] = set()

    def visit(step: str) -> None:
        for dependency in CUSTOM_ANALYSIS_DEPENDENCIES.get(step, ()):
            if dependency in required:
                continue
            required.add(dependency)
            visit(dependency)

    for step in selected_steps:
        visit(step)
    return ordered_step_labels(required)


def analysis_step_satisfied(output_dir: Path, step_label: str) -> bool:
    """Return whether a step appears complete in session state or outputs."""
    step_number = STEP_NAVIGATION_TO_WORKFLOW_STEP[step_label]
    return step_number in completed_workflow_steps() or step_output_exists(
        output_dir, step_number
    )


def missing_planner_prerequisites(
    output_dir: Path,
    selected_steps: Iterable[str],
) -> list[str]:
    """Return required steps not selected and not already present in the run."""
    selected = set(selected_steps)
    missing = []
    for dependency in required_prerequisite_steps(selected):
        if dependency in selected or analysis_step_satisfied(output_dir, dependency):
            continue
        missing.append(dependency)
    return missing


def custom_analysis_plan(
    output_dir: Path,
    selected_steps: Iterable[str],
    *,
    include_prerequisites: bool,
) -> dict[str, list[str]]:
    """Build a passive selected-step plan without executing anything."""
    selected = ordered_step_labels(selected_steps)
    required = required_prerequisite_steps(selected)
    missing = missing_planner_prerequisites(output_dir, selected)
    if include_prerequisites:
        planned = ordered_step_labels([*required, *selected])
    else:
        planned = selected
    return {
        "selected": selected,
        "required": required,
        "missing": missing,
        "planned": planned,
    }


def required_input_labels_for_plan(planned_steps: Iterable[str]) -> tuple[str, ...]:
    """Return active-run input labels needed for a planned step list."""
    required = {"generated_smiles"}
    steps = set(planned_steps)
    if steps.intersection(
        {
            "Descriptors",
            "Chemical-space map",
            "Biomedical evidence",
            "Prioritization",
            "Reports",
        }
    ):
        required.add("references")
    if "Biomedical evidence" in steps:
        required.add("text_evidence")
    return tuple(
        label
        for label in ("generated_smiles", "references", "text_evidence")
        if label in required
    )


def missing_required_inputs_for_plan(
    status: ActiveRunPathStatus,
    planned_steps: Iterable[str],
) -> tuple[str, ...]:
    """Return required input labels missing for the selected plan."""
    missing = set(status.missing_inputs)
    return tuple(
        label
        for label in required_input_labels_for_plan(planned_steps)
        if label in missing
    )


def mark_workflow_step_completed(step_number: int) -> None:
    """Mark one workflow step complete in session state."""
    completed = completed_workflow_steps()
    completed.add(step_number)
    st.session_state["completed_workflow_steps"] = sorted(completed)


def active_run_step_output_paths(
    paths: PipelinePaths,
    step_number: int,
) -> list[Path]:
    """Return expected output artifacts for an active-run step."""
    identity_path = (
        paths.chemical_identity
        or paths.prioritized.parent / "chemical_identity.csv"
    )
    if step_number == 1:
        return [paths.standardized]
    if step_number == 2:
        return [identity_path]
    if step_number == 3:
        return [paths.public_lookup, paths.surechembl_lookup]
    if step_number == 4:
        return [paths.descriptors, paths.similarity, paths.similarity_top_hits]
    if step_number == 5:
        return [paths.chemberta_embeddings, paths.visualization_coordinates]
    if step_number == 6:
        return [paths.biomedical_evidence]
    if step_number == 7:
        return [paths.patent_evidence_embeddings]
    if step_number == 8:
        return [paths.prioritized]
    if step_number == 9:
        reports_dir = paths.prioritized.parent / "reports"
        return sorted(reports_dir.glob("compound_intelligence_report_*.md"))
    return []


def active_run_step_outputs_exist(
    paths: PipelinePaths,
    step_number: int,
) -> bool:
    """Return whether expected active-run step outputs already exist."""
    if step_number == 9:
        return bool(active_run_step_output_paths(paths, step_number))
    expected = active_run_step_output_paths(paths, step_number)
    return bool(expected) and all(path.exists() for path in expected)


def public_demo_step_output_paths(
    paths: PipelinePaths,
    step_number: int,
) -> list[Path]:
    """Return expected output artifacts for a public-demo step."""
    return active_run_step_output_paths(paths, step_number)


def public_demo_step_outputs_exist(
    paths: PipelinePaths,
    step_number: int,
) -> bool:
    """Return whether expected public-demo step outputs already exist."""
    return active_run_step_outputs_exist(paths, step_number)


def run_active_run_step(
    step_number: int,
    paths: PipelinePaths,
    *,
    step3_progress_callback: object | None = None,
) -> None:
    """Run one active-run workflow step using existing step execution logic."""
    if step_number == 3 and step3_progress_callback is not None:
        run_public_demo_step3(paths, progress_callback=step3_progress_callback)
        return
    run_public_demo_step(step_number, paths)


def run_selected_active_run_steps(
    paths: PipelinePaths,
    selected_steps: Iterable[str],
    workflow_mode: str,
    include_existing: bool = True,
    *,
    step3_progress_callback: object | None = None,
) -> dict[str, list[str]]:
    """Run selected active-run steps in dependency order."""
    if workflow_mode == "existing_results":
        return {
            "executed": [],
            "skipped_existing": [],
            "blocked": ["Loaded previous runs are view-only."],
            "failed": [],
        }
    selected = ordered_step_labels(selected_steps)
    plan = ordered_step_labels([*required_prerequisite_steps(selected), *selected])
    summary: dict[str, list[str]] = {
        "executed": [],
        "skipped_existing": [],
        "blocked": [],
        "failed": [],
    }
    available = completed_workflow_steps()

    for step_label in plan:
        step_number = STEP_NAVIGATION_TO_WORKFLOW_STEP[step_label]
        missing = [
            dependency
            for dependency in CUSTOM_ANALYSIS_DEPENDENCIES.get(step_label, ())
            if STEP_NAVIGATION_TO_WORKFLOW_STEP[dependency] not in available
            and not active_run_step_outputs_exist(
                paths,
                STEP_NAVIGATION_TO_WORKFLOW_STEP[dependency],
            )
        ]
        if missing:
            summary["blocked"].append(
                f"{step_label}: missing prerequisite "
                + ", ".join(missing)
            )
            continue

        if step_number in available:
            mark_workflow_step_completed(step_number)
            summary["skipped_existing"].append(step_label)
            continue

        if include_existing and active_run_step_outputs_exist(paths, step_number):
            mark_workflow_step_completed(step_number)
            available.add(step_number)
            summary["skipped_existing"].append(step_label)
            continue

        try:
            run_active_run_step(
                step_number,
                paths,
                step3_progress_callback=step3_progress_callback,
            )
        except Exception as exc:
            summary["failed"].append(
                f"{step_label}: {type(exc).__name__}: {exc}"
            )
            break
        mark_workflow_step_completed(step_number)
        available.add(step_number)
        summary["executed"].append(step_label)

    return summary


def run_selected_public_demo_steps(
    paths: PipelinePaths,
    selected_steps: Iterable[str],
    include_existing: bool = True,
) -> dict[str, list[str]]:
    """Run selected public-demo steps in dependency order."""
    return run_selected_active_run_steps(
        paths,
        selected_steps,
        "public_demo",
        include_existing=include_existing,
    )


def render_step_list(label: str, steps: Iterable[str]) -> None:
    """Render a compact step list for the custom planner."""
    step_list = list(steps)
    st.markdown(f"**{label}**")
    if not step_list:
        st.write("None")
        return
    st.markdown("\n".join(f"- {step}" for step in step_list))

def render_selected_step_run_summary(summary: dict[str, list[str]]) -> None:
    """Render the result of a public-demo selected-step run."""
    st.success("Selected planned steps finished.")
    left, middle, right = st.columns(3)
    with left:
        render_step_summary_card("Executed steps", summary.get("executed", []))
    with middle:
        render_step_summary_card(
            "Skipped existing outputs", summary.get("skipped_existing", [])
        )
    with right:
        render_step_summary_card(
            "Failed or blocked steps",
            [
                *summary.get("blocked", []),
                *summary.get("failed", []),
            ],
        )

def render_custom_analysis_planner(output_dir: Path) -> None:
    """Render a passive step-selection planner for future custom execution."""
    st.markdown("### Custom analysis planner")
    st.write(
        "Choose analysis steps to preview dependencies and a recommended execution "
        "plan. This planner does not run pipeline steps yet."
    )
    include_prerequisites = st.checkbox(
        "Include required prerequisite steps automatically",
        value=True,
        key="custom_planner_include_prerequisites",
    )
    selected_steps = []
    for step in CUSTOM_ANALYSIS_STEPS:
        if st.checkbox(step, key=f"custom_planner_step_{slugify_key(step)}"):
            selected_steps.append(step)

    plan = custom_analysis_plan(
        output_dir,
        selected_steps,
        include_prerequisites=include_prerequisites,
    )
    has_missing = bool(plan["missing"])
    if has_missing:
        st.warning(
            "This step requires earlier outputs. Include prerequisite steps or "
            "load a previous run containing those outputs."
        )
    render_step_list("Selected steps", plan["selected"])
    render_step_list("Required prerequisite steps", plan["required"])
    render_step_list("Missing prerequisites", plan["missing"])
    render_step_list("Recommended full execution plan", plan["planned"])
    workflow_mode = str(st.session_state.get("workflow_mode", ""))
    path_status = resolve_active_run_paths(output_dir, workflow_mode)
    missing_inputs = missing_required_inputs_for_plan(path_status, plan["planned"])
    if workflow_mode == "existing_results":
        st.info(
            "Loaded previous runs are view-only for selected-step execution. "
            "Start a new analysis to run selected steps."
        )
        return
    can_run_selected = workflow_mode == "public_demo" or (
        path_status.run_type == "uploaded" and path_status.paths_resolved
    )
    if missing_inputs:
        st.warning(
            "Selected-step execution needs these stored input files: "
            + ", ".join(missing_inputs)
            + "."
        )
        can_run_selected = False
    if not can_run_selected:
        st.info(
            "Run selected steps for uploaded analyses will be added in a later "
            "update. For now, use the guided workflow or full run."
        )
        return
    if st.button(
        "Run selected planned steps",
        type="primary",
        disabled=not bool(plan["planned"]) or bool(missing_inputs),
    ):
        if has_missing:
            st.error(
                "Selected steps were not run because required prerequisite outputs "
                "are missing."
            )
            return
        if "Public evidence" in plan["planned"]:
            preflight = pubchem_preflight()
            if not preflight.available:
                render_pubchem_preflight_result(preflight)
                return
        progress_bar = None
        progress_status = None
        if "Public evidence" in plan["planned"]:
            progress_bar = st.progress(
                0.0, text="Preparing public database lookup..."
            )
            progress_status = st.empty()
        summary = run_selected_active_run_steps(
            path_status.pipeline_paths,
            plan["planned"],
            workflow_mode,
            step3_progress_callback=(
                (
                    lambda progress: render_step3_progress(
                        progress,
                        progress_bar,
                        progress_status,
                    )
                )
                if progress_bar is not None and progress_status is not None
                else None
            ),
        )
        render_selected_step_run_summary(summary)


def render_step_navigation_sidebar(output_dir: Path) -> str:
    """Render active-run step navigation without running workflow steps."""
    current = int(st.session_state.get("workflow_step", 1))
    current = max(1, min(current, len(WORKFLOW_STEP_NAMES)))
    display_labels = step_navigation_display_labels(output_dir)
    pending_page = str(st.session_state.pop(PENDING_ACTIVE_RUN_PAGE_KEY, "")).strip()
    if pending_page:
        pending_page = normalize_step_navigation_label(pending_page, display_labels)
        if pending_page in STEP_NAVIGATION_LABELS:
            st.session_state["active_run_page"] = pending_page
            st.session_state[SIDEBAR_STEP_NAVIGATION_WIDGET_KEY] = display_labels[
                pending_page
            ]
    page_value = str(st.session_state.get("active_run_page", "")).strip()
    default_label = page_value or WORKFLOW_STEP_TO_NAVIGATION_LABEL.get(
        current, "Overview"
    )
    if default_label not in STEP_NAVIGATION_LABELS:
        default_label = WORKFLOW_STEP_TO_NAVIGATION_LABEL.get(current, "Overview")
    options = [display_labels[label] for label in STEP_NAVIGATION_LABELS]
    default_index = STEP_NAVIGATION_LABELS.index(default_label)
    with st.sidebar:
        st.markdown("### Step navigation")
        selected_display = st.radio(
            "Step navigation",
            options,
            index=default_index,
            key=SIDEBAR_STEP_NAVIGATION_WIDGET_KEY,
            label_visibility="collapsed",
        )
        st.caption("✓ completed")
        st.caption("○ not run")
        st.caption("⚠ skipped/unavailable")
        st.caption("✗ missing prerequisite")
    selected = normalize_step_navigation_label(str(selected_display), display_labels)
    st.session_state["active_run_page"] = str(selected)
    mapped_step = STEP_NAVIGATION_TO_WORKFLOW_STEP.get(str(selected))
    if mapped_step is not None:
        st.session_state["workflow_step"] = mapped_step
    return str(selected)


def render_active_run_page(
    selected_page: str,
    output_dir: Path,
) -> None:
    """Render the selected active-run sidebar page."""
    if selected_page in STEP_NAVIGATION_TO_WORKFLOW_STEP:
        render_step_workflow(output_dir)
        return
    render_step_navigation_context(selected_page, output_dir)
    render_active_run_reset_button(key=f"start_over_{slugify_key(selected_page)}")


def render_step_navigation_context(
    selected_page: str,
    output_dir: Path,
) -> None:
    """Render passive context panels for active-run sidebar pages."""
    if selected_page in STEP_NAVIGATION_TO_WORKFLOW_STEP:
        return
    if selected_page == "Overview":
        render_active_run_overview(output_dir)
    elif selected_page == "Input data":
        render_active_run_input_data(output_dir)
    elif selected_page == "Downloads":
        render_active_run_downloads(output_dir)
    elif selected_page == "Settings":
        render_active_run_settings(output_dir)


def slugify_key(value: str) -> str:
    """Return a stable lowercase key fragment for Streamlit widget keys."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def clear_active_run_state() -> None:
    """Clear active-run state while leaving unrelated session state intact."""
    for key in (
        "active_output_dir",
        "workflow_step",
        "completed_workflow_steps",
        "workflow_mode",
        "active_run_page",
        PENDING_ACTIVE_RUN_PAGE_KEY,
        "sidebar_step_navigation",
        SIDEBAR_STEP_NAVIGATION_WIDGET_KEY,
    ):
        st.session_state.pop(key, None)


def render_active_run_reset_button(*, key: str) -> None:
    """Show the active-run reset action on routed sidebar pages."""
    if st.button("Start over", key=key):
        clear_active_run_state()
        st.rerun()
def render_active_run_overview(output_dir: Path) -> None:
    """Show passive current-run status before the guided workflow body."""
    completed = {
        int(value)
        for value in st.session_state.get("completed_workflow_steps", [])
        if str(value).strip().isdigit()
    }
    current = int(st.session_state.get("workflow_step", 1))
    current = max(1, min(current, len(WORKFLOW_STEP_NAMES)))
    workflow_mode = str(st.session_state.get("workflow_mode", "active run")) or "active run"
    artifacts = count_existing_outputs(output_dir)
    render_section_header(
        "Overview",
        "Passive run status. Selecting pages here does not execute pipeline steps.",
    )
    st.caption(f"Current run folder: {output_dir}")
    columns = st.columns(4)
    render_modern_metric_card("Workflow mode", workflow_mode, container=columns[0])
    render_modern_metric_card(
        "Completed steps",
        f"{len(completed)} of {len(WORKFLOW_STEP_NAMES)}",
        container=columns[1],
    )
    render_modern_metric_card("Current guided step", current, container=columns[2])
    render_modern_metric_card("Available output files", artifacts, container=columns[3])
    if completed:
        st.write(
            "Completed workflow steps: "
            + ", ".join(str(step) for step in sorted(completed))
        )
        next_action = "Continue from the sidebar step navigation or open Downloads to review artifacts."
    else:
        st.info("No workflow steps have been marked complete for this active run yet.")
        next_action = "Open the first workflow step and run it only when you are ready."
    render_modern_card(
        "Next recommended action",
        next_action,
        key="active_run_next_action_card",
    )
    render_modern_card(
        "Run status",
        f"{artifacts} known artifact(s) are available. Current guided step is {current}.",
        description="This panel is read-only and does not execute workflow logic.",
        key="active_run_status_card",
    )
def count_existing_outputs(output_dir: Path) -> int:
    """Count known output artifacts that already exist for a run."""
    count = sum(
        1 for filename in OUTPUT_FILES.values() if (output_dir / filename).exists()
    )
    reports_dir = output_dir / "reports"
    if reports_dir.exists() and any(
        reports_dir.glob("compound_intelligence_report_*.md")
    ):
        count += 1
    return count


def render_active_run_input_data(output_dir: Path) -> None:
    """Show passive input artifact context for the active run."""
    st.subheader("Input data")
    workflow_mode = str(st.session_state.get("workflow_mode", ""))
    if workflow_mode == "public_demo":
        st.write("This active run uses the bundled public-safe demo inputs.")
        render_artifact_name_list(public_demo_input_paths())
        return
    run_dir = output_dir.parent
    input_dir = run_dir / "inputs"
    input_files = sorted(input_dir.glob("*.csv")) if input_dir.exists() else []
    if input_files:
        st.write("Input artifacts stored with this run:")
        render_artifact_name_list(input_files)
        return
    st.info(
        "Input files are not stored with this output folder. Step 1 and the output "
        "artifact cards below show the available processed inputs."
    )


def render_active_run_downloads(output_dir: Path) -> None:
    """Show known output artifacts for download without changing workflow state."""
    st.subheader("Downloads")
    loaded = load_output_directory(output_dir)
    artifacts = [path for path in loaded.paths.values() if path.exists()]
    if loaded.reports_dir.exists():
        artifacts.append(loaded.reports_dir)
    if not artifacts:
        st.info("No downloadable workflow outputs were found yet.")
        return
    render_output_artifacts(artifacts)

def render_run_path_status(output_dir: Path) -> None:
    """Show read-only active-run path resolution details."""
    status = resolve_active_run_paths(
        output_dir,
        str(st.session_state.get("workflow_mode", "")),
    )
    render_section_header("Run path status", level=3)
    columns = st.columns(3)
    render_metric_card("Run type", status.run_type, container=columns[0])
    render_metric_card(
        "Paths resolved", "yes" if status.paths_resolved else "no", container=columns[1]
    )
    render_metric_card(
        "Existing outputs", len(status.existing_outputs), container=columns[2]
    )
    left, middle, right = st.columns(3)
    with left:
        render_step_summary_card("Missing inputs", status.missing_inputs)
    with middle:
        render_step_summary_card("Existing outputs", status.existing_outputs)
    with right:
        render_step_summary_card("Unresolved paths", status.unresolved_paths)

def render_optional_domain_model_settings(output_dir: Path) -> None:
    """Render local-only optional domain-model testing controls."""
    st.markdown("### Optional local domain-model testing")
    st.warning(
        "Large domain-specific models are intended for local testing. Streamlit "
        "Cloud may skip these models or fall back to the cloud-safe model if they "
        "are unavailable."
    )
    st.caption(
        "Selected domain models are attempted from local/cache-safe sources first; "
        "the lightweight general-purpose fallback is used when a preferred model "
        "is unavailable."
    )
    biomedical_option = st.selectbox(
        "Biomedical evidence model",
        BIOMEDICAL_MODEL_OPTIONS,
        key="biomedical_domain_model_option",
    )
    if biomedical_option == CUSTOM_MODEL_LABEL:
        st.text_input(
            "Biomedical custom Hugging Face model ID",
            key="biomedical_custom_model_id",
        )
    patent_option = st.selectbox(
        "Patent/IP-context evidence model",
        PATENT_MODEL_OPTIONS,
        key="patent_domain_model_option",
    )
    if patent_option == "PaECTER":
        st.info(
            "PaECTER was described as available on Hugging Face, but this app does "
            "not hardcode an unverified model ID. Select Custom Hugging Face model "
            "ID to test a local PaECTER checkpoint."
        )
    if patent_option == CUSTOM_MODEL_LABEL:
        st.text_input(
            "Patent/IP custom Hugging Face model ID",
            key="patent_custom_model_id",
        )
    gate_enabled = bool(os.environ.get(ALLOW_LOCAL_MODEL_DOWNLOADS_ENV) == "1")
    st.caption(
        f"{ALLOW_LOCAL_MODEL_DOWNLOADS_ENV}={'1' if gate_enabled else 'not set'}"
    )
    latest_checks = st.session_state.get("environment_check_result", [])
    if latest_checks:
        st.markdown("**Latest environment-check result**")
        render_environment_check_rows(latest_checks)
def render_active_run_settings(output_dir: Path) -> None:
    """Show passive configuration notes for the active run."""
    render_section_header(
        "Settings",
        "Read-only configuration notes and local model controls for this active run.",
    )
    if not shadcn_ui_available():
        st.caption("Advanced UI components unavailable; using native Streamlit rendering.")
    render_modern_card(
        "Environment checks",
        "Run package, online lookup, and optional model availability checks without starting a workflow.",
        key="settings_environment_checks_card",
    )
    render_modern_card(
        "Model configuration/status",
        "Optional local biomedical and patent/IP evidence models can be selected here; fallback behavior is unchanged.",
        key="settings_model_configuration_card",
    )
    render_optional_domain_model_settings(output_dir)
    render_modern_card(
        "Run path status",
        "Review resolved inputs, missing inputs, unresolved paths, and existing outputs for this active run.",
        key="settings_run_path_card",
    )
    render_run_path_status(output_dir)
    render_modern_card(
        "Custom analysis planner",
        "Preview selected-step dependencies and recommended execution plans without running pipeline steps automatically.",
        key="settings_custom_planner_card",
    )
    render_custom_analysis_planner(output_dir)
def run_app() -> None:
    """Run the guided Streamlit workflow without loading old results."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_design_foundation_css()
    st.title(APP_TITLE)
    st.markdown(WELCOME_TEXT)
    st.info(START_GUIDANCE)
    with st.expander("About the workflow"):
        for heading, explanation in ABOUT_WORKFLOW_SECTIONS:
            st.markdown(f"**{heading}**")
            st.markdown(explanation)

    output_dir = active_output_directory()
    workflow_mode = render_workflow_mode_sidebar(output_dir)
    if output_dir is not None:
        selected_step_page = render_step_navigation_sidebar(output_dir)
        render_active_run_page(selected_step_page, output_dir)
        return

    demo_output = None
    upload_output = None
    existing_output = None
    sidebar_existing_output = None
    if workflow_mode == "Start":
        with st.sidebar:
            with st.expander("Load existing results"):
                sidebar_existing_output = render_load_existing_choice()
        render_start_workflow_mode()
    elif workflow_mode == "Guided demo":
        demo_output = render_public_demo_choice()
    elif workflow_mode == "Analyze my molecules":
        upload_output = render_new_analysis_form()
    elif workflow_mode == "Load previous run":
        existing_output = render_load_existing_choice()

    output_dir = demo_output or upload_output or existing_output or sidebar_existing_output
    if output_dir is not None:
        selected_step_page = render_step_navigation_sidebar(output_dir)
        render_active_run_page(selected_step_page, output_dir)


if __name__ == "__main__":
    run_app()
