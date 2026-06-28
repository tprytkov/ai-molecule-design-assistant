"""Streamlit dashboard for local molecule-intelligence outputs."""

from __future__ import annotations

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
import streamlit as st
from rdkit import Chem

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
WORKFLOW_STEP_WHY = (
    "Invalid or inconsistent structures can make every downstream result "
    "misleading, so structure quality is checked first.",
    "Exact names and identifiers make later public evidence easier to interpret "
    "without guessing what a structure represents.",
    "Public records show whether exact or related compounds are already present "
    "in external chemistry resources.",
    "These interpretable properties reveal size, polarity, lipophilicity, "
    "drug-likeness, and rule-based property concerns.",
    "Learned embeddings provide a complementary view of structural relationships "
    "that is not limited to a single hand-designed fingerprint.",
    "Biomedical evidence uses a lightweight general embedding baseline by default; "
    "BioBERT/PubMedBERT-style local cached models are optional, and a skipped cloud "
    "model is not an error.",
    "Patent/IP-context evidence keeps SureChEMBL structure evidence separate from "
    "optional PaECTER/patent-BERT-style patent-text embeddings and is not a legal "
    "conclusion.",
    "The prior evidence is combined into one transparent research-prioritization "
    "view while preserving each stage's availability status.",
    "A report collects the evidence for one molecule into a readable artifact "
    "that can be reviewed or shared locally.",
)
WORKFLOW_STEP_GET = (
    "A standardized table with valid/invalid status, canonical SMILES, and InChIKeys.",
    "Exact public names and identifiers when supported, plus explicit no-match status.",
    "PubChem/ChEMBL lookup evidence and SureChEMBL structure-evidence status.",
    "RDKit descriptors, drug-likeness categories, flags, and property visualizations.",
    "ChemBERTa embeddings and two-dimensional chemical-space coordinates.",
    "Grounded compound context and biomedical evidence relevance results.",
    "A patent/IP-context evidence file that is safe when the patent model is unavailable.",
    "A ranked molecule table with component scores and evidence-stage statuses.",
    "Molecule-level Markdown reports with structures and evidence summaries.",
)
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
        metrics[index].metric(label, value)

    st.markdown("#### External Public Evidence")
    st.caption(
        "Not queried means the molecule was not checked because of the molecule limit or workflow settings."
    )
    st.table(
        build_external_public_evidence_table(
            df,
            public_lookup_exists=loaded.paths["public_lookup"].exists(),
            surechembl_exists=loaded.paths["surechembl"].exists(),
        )
    )
    st.markdown("#### Computed Analysis Status")
    st.info(nlp_output_note(loaded.paths["text_nlp"], loaded.tables["text_nlp"]))
    st.caption(
        "ChemBERTa availability means molecular embeddings were generated; "
        "it is not a public database match. ChemBERTa is unavailable for invalid SMILES."
    )
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
        ["x", "y", "prioritization_score_with_nlp", "prioritization_score", "tanimoto_similarity"],
    )


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
    color_column = next(
        (
            column
            for column in ("druglikeness_category", "cluster_id", "chemberta_status")
            if column in plot_df.columns
        ),
        "",
    )
    if color_column == "cluster_id":
        plot_df[color_column] = plot_df[color_column].fillna("not assigned").astype(str)
    hover = [
        column
        for column in (
            "molecule_id",
            "best_reference_name",
            "tanimoto_similarity",
            "cluster_id",
            "druglikeness_category",
            "novelty_flag",
        )
        if column in plot_df.columns
    ]
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_column or None,
        hover_name="molecule_id",
        hover_data=hover,
        custom_data=["molecule_id"],
        labels=display_labels(plot_df.columns),
        title="ChemBERTa / UMAP Chemical Space",
        color_discrete_map=(
            None
            if color_column == "cluster_id"
            else category_color_map(plot_df, color_column)
            if color_column
            else None
        ),
    )
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}_plot",
        on_select="rerun",
        selection_mode="points",
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
    size_text = f" - {file_size}" if file_size else ""
    st.markdown(f"#### {label}")
    st.write(description)
    st.write(
        f"Artifact: {filename} - {len(dataframe)} {row_word} - "
        f"{len(dataframe.columns)} columns{size_text}"
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
        if pubchem_rows and all(
            item.lookup_status == "lookup_error" for item in pubchem_rows
        ):
            raise RuntimeError(
                f"PubChem lookup failed for all {len(pubchem_rows)} valid molecules."
            )

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
        if chembl_rows and all(
            item.lookup_status == "lookup_error" for item in chembl_rows
        ):
            raise RuntimeError(
                f"ChEMBL lookup failed for all {len(chembl_rows)} valid molecules."
            )

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
        if surechembl_valid_results and all(
            item.lookup_status == "lookup_error"
            for item in surechembl_valid_results
        ):
            raise RuntimeError(
                "SureChEMBL lookup failed for all "
                f"{len(valid_rows)} valid molecules."
            )

        write_public_lookup_csv(public_temp, public_results)
        write_surechembl_output_csv(surechembl_temp, surechembl_results)
        public_temp.replace(paths.public_lookup)
        surechembl_temp.replace(paths.surechembl_lookup)
    except Exception:
        public_temp.unlink(missing_ok=True)
        surechembl_temp.unlink(missing_ok=True)
        raise

    return step3_summary(
        pd.read_csv(paths.public_lookup),
        pd.read_csv(paths.surechembl_lookup),
        standardized,
    )


def render_step3_summary(summary: dict[str, int]) -> None:
    """Show the requested Step 3 completion counts."""
    st.success("Step 3 public database lookup completed.")
    st.dataframe(
        pd.DataFrame(
            {
                "Result": list(summary),
                "Molecules": list(summary.values()),
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
        biomedical_evidence_csv(
            context_path,
            paths.text_evidence,
            paths.biomedical_evidence,
            identity_path=identity_path,
            descriptor_path=paths.descriptors,
        )
    elif step_number == 7:
        patent_evidence_embeddings_csv(
            paths.surechembl_lookup,
            paths.patent_evidence_embeddings,
            public_lookup_path=paths.public_lookup,
            identity_path=identity_path,
            context_path=context_path,
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
    st.markdown("**Public example files**")
    st.markdown(
        "Use these files as templates for preparing your own generated SMILES input."
    )
    for path in public_demo_input_paths():
        render_example_file_preview(path)
    if st.button("Test online lookup connection"):
        render_pubchem_preflight_result(pubchem_preflight())
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


def render_step_header(
    step_number: int,
    description: str,
    inputs: Iterable[Path | str],
    outputs: Iterable[Path],
) -> None:
    """Render shared explanation and artifact details for one workflow step."""
    st.header(WORKFLOW_STEP_NAMES[step_number - 1])
    st.markdown("#### What this step calculates")
    st.write(description)
    st.markdown("#### Why we run it")
    st.write(WORKFLOW_STEP_WHY[step_number - 1])
    st.markdown("#### What you will get")
    st.write(WORKFLOW_STEP_GET[step_number - 1])
    left, right = st.columns(2)
    with left:
        st.markdown("**Input used**")
        render_artifact_name_list(inputs)
    with right:
        st.markdown("**Output file created**")
        render_artifact_name_list(outputs)
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
            [loaded.paths["standardized"]],
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
            [loaded.paths["chemical_identity"]],
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
            [loaded.paths["public_lookup"], loaded.paths["surechembl"]],
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
            step3_summary(
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
            [loaded.paths["descriptors"]],
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
            [
                loaded.paths["chemberta_embeddings"],
                loaded.paths["visualization"],
            ],
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
            [
                loaded.paths["biomedical_evidence"],
                loaded.paths["text_nlp"],
                loaded.paths["compound_context"],
            ],
        )
        if not results_available:
            return
        molecule_ids = (
            context["molecule_id"].astype(str).tolist()
            if not context.empty and "molecule_id" in context.columns
            else ()
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
            [loaded.paths["patent_evidence_embeddings"]],
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
            [loaded.paths["prioritization"]],
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
            [loaded.reports_dir],
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
        if workflow_mode != "public_demo":
            st.warning("This workflow step has not been run for the selected output.")
            return
        st.info(
            "No calculation has run for this step yet. Review the explanation "
            "above, then run the public example when you are ready."
        )
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
        return

    if current < len(WORKFLOW_STEP_NAMES):
        if st.button(
            f"Continue to {WORKFLOW_STEP_NAMES[current]}",
            type="primary",
            key=f"continue_step_{current}",
        ):
            st.session_state["workflow_step"] = current + 1
            st.rerun()
    else:
        if st.button("Start over", key="start_over"):
            st.session_state.pop("active_output_dir", None)
            st.session_state.pop("workflow_step", None)
            st.session_state.pop("completed_workflow_steps", None)
            st.session_state.pop("workflow_mode", None)
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


def run_app() -> None:
    """Run the guided Streamlit workflow without loading old results."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.markdown(WELCOME_TEXT)
    st.info(START_GUIDANCE)
    with st.expander("About the workflow"):
        for heading, explanation in ABOUT_WORKFLOW_SECTIONS:
            st.markdown(f"**{heading}**")
            st.markdown(explanation)

    output_dir = active_output_directory()
    if output_dir is not None:
        render_step_workflow(output_dir)
        return

    demo_tab, upload_tab = st.tabs(
        ["Guided example workflow", "Upload my own SMILES"]
    )
    with demo_tab:
        demo_output = render_public_demo_choice()
    with upload_tab:
        upload_output = render_new_analysis_form()

    existing_output = None
    with st.sidebar:
        with st.expander("Load existing results"):
            existing_output = render_load_existing_choice()

    output_dir = demo_output or upload_output or existing_output
    if output_dir is not None:
        render_step_workflow(output_dir)


if __name__ == "__main__":
    run_app()
