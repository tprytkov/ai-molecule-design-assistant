"""Run the complete molecule-intelligence workflow."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from src.biomedical_evidence import (
    DEFAULT_BIOMEDICAL_MODEL,
    BiomedicalEncoder,
    biomedical_evidence_csv,
)
from src.chemberta_embeddings import (
    DEFAULT_CHEMBERTA_MODEL,
    ChembertaEmbedder,
    chemberta_embeddings_csv,
    merge_chemberta_into_prioritized,
    visualization_coordinates_csv,
)
from src.chemical_identity import IdentityClient, chemical_identity_csv
from src.compound_context import compound_context_csv
from src.compound_qa import compound_qa
from src.compound_search import top_hits_csv
from src.descriptors import descriptor_csv
from src.patent_evidence_embeddings import (
    DEFAULT_PATENT_MODEL,
    PatentEncoder,
    patent_evidence_embeddings_csv,
)
from src.public_lookup import JsonClient, public_lookup_csv
from src.scoring import scoring_csv
from src.similarity import similarity_csv
from src.standardize import standardize_csv
from src.surechembl_lookup import (
    OUTPUT_COLUMNS as SURECHEMBL_OUTPUT_COLUMNS,
    SurechemblClient,
    surechembl_lookup_csv,
)
from src.text_nlp import SentenceEncoder, text_nlp_csv

DEFAULT_INPUT = Path("data/demo_generated_smiles.csv")
DEFAULT_REFERENCES = Path("data/reference_molecules.csv")
DEFAULT_TEXT_EVIDENCE = Path("data/demo_text_evidence.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_SURECHEMBL_COMPOUNDS = Path("data/demo_surechembl_compounds.csv")


@dataclass(frozen=True)
class PipelinePaths:
    """Input and output paths for the complete pipeline."""

    generated_smiles: Path = DEFAULT_INPUT
    references: Path = DEFAULT_REFERENCES
    text_evidence: Path = DEFAULT_TEXT_EVIDENCE
    patent_text_evidence: Path | None = None
    surechembl_compounds: Path = DEFAULT_SURECHEMBL_COMPOUNDS
    standardized: Path = Path("outputs/standardized.csv")
    descriptors: Path = Path("outputs/descriptors.csv")
    similarity: Path = Path("outputs/similarity.csv")
    similarity_top_hits: Path = Path("outputs/similarity_top_hits.csv")
    text_nlp: Path = Path("outputs/text_nlp.csv")
    biomedical_evidence: Path = Path("outputs/biomedical_evidence.csv")
    patent_evidence_embeddings: Path = Path("outputs/patent_evidence_embeddings.csv")
    public_lookup: Path = Path("outputs/public_lookup.csv")
    chemical_identity: Path | None = None
    compound_context: Path | None = None
    surechembl_lookup: Path = Path("outputs/surechembl_evidence.csv")
    chemberta_embeddings: Path = Path("outputs/chemberta_embeddings.csv")
    visualization_coordinates: Path = Path(
        "outputs/visualization_coordinates.csv"
    )
    prioritized: Path = Path("outputs/prioritization_results.csv")


def build_paths(
    *,
    input_path: Path = DEFAULT_INPUT,
    references_path: Path = DEFAULT_REFERENCES,
    text_evidence_path: Path = DEFAULT_TEXT_EVIDENCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    surechembl_compounds_path: Path = DEFAULT_SURECHEMBL_COMPOUNDS,
) -> PipelinePaths:
    """Build pipeline paths for a chosen input set and output directory."""
    return PipelinePaths(
        generated_smiles=input_path,
        references=references_path,
        text_evidence=text_evidence_path,
        surechembl_compounds=surechembl_compounds_path,
        standardized=output_dir / "standardized.csv",
        descriptors=output_dir / "descriptors.csv",
        similarity=output_dir / "similarity.csv",
        similarity_top_hits=output_dir / "similarity_top_hits.csv",
        text_nlp=output_dir / "text_nlp.csv",
        biomedical_evidence=output_dir / "biomedical_evidence.csv",
        patent_evidence_embeddings=output_dir / "patent_evidence_embeddings.csv",
        public_lookup=output_dir / "public_lookup.csv",
        chemical_identity=output_dir / "chemical_identity.csv",
        compound_context=output_dir / "compound_context.csv",
        surechembl_lookup=output_dir / "surechembl_evidence.csv",
        chemberta_embeddings=output_dir / "chemberta_embeddings.csv",
        visualization_coordinates=output_dir / "visualization_coordinates.csv",
        prioritized=output_dir / "prioritization_results.csv",
    )


def validate_inputs(paths: PipelinePaths) -> None:
    """Fail before execution if any required demo input is missing."""
    required_inputs = (
        paths.generated_smiles,
        paths.references,
        paths.text_evidence,
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required pipeline input file(s): " + ", ".join(missing)
        )


def validate_surechembl_input(paths: PipelinePaths) -> None:
    """Fail before local SureChEMBL mode if the selected input is missing."""
    if not paths.surechembl_compounds.is_file():
        raise FileNotFoundError(
            "Missing required SureChEMBL structure evidence compound file: "
            f"{paths.surechembl_compounds}"
        )


def write_empty_text_evidence(path: Path) -> None:
    """Create an empty text-evidence CSV for custom workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("evidence_id", "molecule_id", "source_type", "title", "text"),
        )
        writer.writeheader()


def write_not_run_evidence_embeddings_csv(
    descriptor_path: Path,
    output_path: Path,
    *,
    note: str,
) -> int:
    """Write a schema-valid placeholder evidence-embedding CSV."""
    rows: list[dict[str, str]] = []
    if descriptor_path.exists():
        with descriptor_path.open("r", encoding="utf-8-sig", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "molecule_id",
                "evidence_id",
                "embedding_status",
                "max_relevance_score",
                "evidence_relevance_category",
                "evidence_note",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "molecule_id": row.get("molecule_id", "").strip(),
                    "evidence_id": "",
                    "embedding_status": "not_run",
                    "max_relevance_score": "0.000",
                    "evidence_relevance_category": "not_run",
                    "evidence_note": note,
                }
            )
    return len(rows)


def csv_has_data_rows(path: Path) -> bool:
    """Return whether a CSV contains at least one data row."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return next(csv.DictReader(input_file), None) is not None


def write_surechembl_not_run_csv(
    descriptor_path: Path, output_path: Path
) -> int:
    """Write SureChEMBL-compatible placeholder rows when the step is skipped."""
    with descriptor_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        descriptor_rows = list(csv.DictReader(input_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SURECHEMBL_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in descriptor_rows:
            writer.writerow(
                {
                    "molecule_id": row.get("molecule_id", "").strip(),
                    "canonical_smiles": row.get("canonical_smiles", "").strip(),
                    "valid_smiles": row.get("valid_smiles", "").strip(),
                    "lookup_status": "not_run",
                    "evidence_note": (
                        "SureChEMBL structure evidence was not run for this workflow."
                    ),
                }
            )
    return len(descriptor_rows)


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_score(value: object) -> float:
    """Parse prioritization score values for report ranking."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return -1.0


def is_fully_analyzed(
    row: Mapping[str, str],
    *,
    online_surechembl: bool,
    use_chemberta: bool,
) -> bool:
    """Return whether a prioritized molecule has all requested online evidence."""
    if not parse_boolean(row.get("valid_smiles", "")):
        return False
    if row.get("pubchem_status", "").strip() == "not_queried":
        return False
    if row.get("chembl_status", "").strip() == "not_queried":
        return False
    if (
        online_surechembl
        and row.get("surechembl_query_status", "").strip() == "not_queried"
    ):
        return False
    if use_chemberta and row.get("chemberta_status", "").strip() != "available":
        return False
    return True


def select_report_molecule_ids(
    prioritized_path: Path,
    *,
    report_molecule: str | None = None,
    report_top_n: int | None = None,
    report_all: bool = False,
    report_only_fully_analyzed: bool = False,
    online_surechembl: bool = False,
    use_chemberta: bool = False,
) -> list[str]:
    """Select molecule IDs for report generation using requested priority."""
    if report_molecule:
        return [report_molecule]
    with prioritized_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    valid_rows = [
        row
        for row in rows
        if parse_boolean(row.get("valid_smiles", ""))
        and row.get("molecule_id", "").strip()
    ]
    ranked = sorted(
        valid_rows,
        key=lambda row: parse_score(row.get("prioritization_score_with_nlp", "")),
        reverse=True,
    )
    if report_only_fully_analyzed:
        before_count = len(ranked)
        ranked = [
            row
            for row in ranked
            if is_fully_analyzed(
                row,
                online_surechembl=online_surechembl,
                use_chemberta=use_chemberta,
            )
        ]
        if report_top_n is not None and len(ranked) < report_top_n:
            print(
                "Warning: fewer fully analyzed molecules are available "
                f"({len(ranked)}) than requested by --report-top-n "
                f"({report_top_n}); {before_count - len(ranked)} valid "
                "molecule(s) were excluded due to incomplete evidence."
            )
    if report_top_n is not None:
        if report_top_n < 1:
            raise ValueError("report-top-n must be at least 1.")
        ranked = ranked[:report_top_n]
    elif not report_all:
        return []
    return [row["molecule_id"].strip() for row in ranked]


def run_pipeline(
    paths: PipelinePaths | None = None,
    nlp_model: SentenceEncoder | None = None,
    *,
    biomedical_model: BiomedicalEncoder | None = None,
    biomedical_model_name: str = DEFAULT_BIOMEDICAL_MODEL,
    patent_model: PatentEncoder | None = None,
    patent_model_name: str = DEFAULT_PATENT_MODEL,
    online_lookup: bool = False,
    max_molecules: int | None = None,
    top_k: int = 5,
    report_molecule: str | None = None,
    lookup_client: JsonClient | None = None,
    identity_client: IdentityClient | None = None,
    refresh_public_lookup: bool = False,
    online_surechembl: bool = False,
    surechembl_client: SurechemblClient | None = None,
    skip_surechembl: bool = False,
    report_dir: Path | None = None,
    report_top_n: int | None = None,
    report_all: bool = False,
    report_only_fully_analyzed: bool = False,
    clean_report_dir: bool = False,
    use_chemberta: bool = False,
    chemberta_model: str = DEFAULT_CHEMBERTA_MODEL,
    refresh_chemberta: bool = False,
    chemberta_embedder: ChembertaEmbedder | None = None,
) -> Path:
    """Run every demo stage in order and return the final output path."""
    active_paths = paths or PipelinePaths()
    validate_inputs(active_paths)
    active_paths.prioritized.parent.mkdir(parents=True, exist_ok=True)
    context_path = (
        active_paths.compound_context
        or active_paths.prioritized.parent / "compound_context.csv"
    )
    identity_path = (
        active_paths.chemical_identity
        or active_paths.prioritized.parent / "chemical_identity.csv"
    )

    print("Step 1/9: Standardizing and validating generated SMILES")
    standardize_csv(active_paths.generated_smiles, active_paths.standardized)

    print("Step 2/9: Identifying exact public chemical names")
    chemical_identity_csv(
        active_paths.standardized,
        identity_path,
        public_lookup_path=active_paths.public_lookup,
        online=online_lookup and (lookup_client is None or identity_client is not None),
        max_molecules=max_molecules,
        client=identity_client,
    )

    print("Step 3/9: Running public database and SureChEMBL lookups")
    if active_paths.public_lookup.exists() and not refresh_public_lookup:
        print("Using existing public lookup file.")
    else:
        lookup_mode = "online" if online_lookup else "offline"
        if active_paths.public_lookup.exists() and refresh_public_lookup:
            print(f"Refreshing public lookup ({lookup_mode})")
        elif online_lookup:
            print("No public lookup file found; running online public lookup.")
        else:
            print(
                "No public lookup file found; generating offline "
                "placeholder lookup."
            )
        public_lookup_csv(
            active_paths.standardized,
            active_paths.standardized,
            active_paths.public_lookup,
            offline=not online_lookup,
            max_molecules=max_molecules,
            client=lookup_client,
        )

    if skip_surechembl:
        print("SureChEMBL structure evidence was not run for this workflow.")
        write_surechembl_not_run_csv(
            active_paths.standardized,
            active_paths.surechembl_lookup,
        )
    else:
        if not online_surechembl:
            validate_surechembl_input(active_paths)
        surechembl_lookup_csv(
            active_paths.standardized,
            None if online_surechembl else active_paths.surechembl_compounds,
            active_paths.surechembl_lookup,
            top_k,
            online_surechembl=online_surechembl,
            max_molecules=max_molecules,
            client=surechembl_client,
        )

    print("Step 4/9: Calculating RDKit properties and reference similarity")
    descriptor_csv(active_paths.standardized, active_paths.descriptors)
    similarity_csv(
        active_paths.descriptors,
        active_paths.references,
        active_paths.similarity,
    )
    top_hits_csv(
        active_paths.descriptors,
        active_paths.references,
        active_paths.similarity_top_hits,
        top_k,
    )

    print("Step 5/9: Generating ChemBERTa chemical-space outputs")
    if use_chemberta:
        if active_paths.chemberta_embeddings.exists() and not refresh_chemberta:
            print("Using existing ChemBERTa embeddings file.")
        else:
            chemberta_embeddings_csv(
                active_paths.standardized,
                active_paths.chemberta_embeddings,
                model_name=chemberta_model,
                embedder=chemberta_embedder,
            )
        visualization_coordinates_csv(
            active_paths.chemberta_embeddings,
            None,
            active_paths.visualization_coordinates,
        )
    else:
        print("ChemBERTa was not run for this workflow.")

    print("Step 6/9: Building biomedical context and scoring biomedical evidence")
    compound_context_csv(
        active_paths.descriptors,
        active_paths.public_lookup,
        active_paths.similarity_top_hits,
        active_paths.references,
        context_path,
        identity_path=identity_path,
    )

    nlp_was_run = csv_has_data_rows(active_paths.text_evidence)
    if nlp_was_run:
        text_nlp_csv(
            active_paths.text_evidence,
            active_paths.text_nlp,
            model=nlp_model,
            context_path=context_path,
            molecule_path=active_paths.generated_smiles,
            descriptor_path=active_paths.descriptors,
            identity_path=identity_path,
        )
    else:
        text_nlp_csv(
            active_paths.text_evidence,
            active_paths.text_nlp,
            model=nlp_model,
        )
    biomedical_evidence_csv(
        context_path,
        active_paths.text_evidence,
        active_paths.biomedical_evidence,
        model=biomedical_model,
        model_name=biomedical_model_name,
        identity_path=identity_path,
        descriptor_path=active_paths.descriptors,
    )

    print("Step 7/9: Scoring patent/IP-context evidence")
    patent_evidence_embeddings_csv(
        active_paths.surechembl_lookup,
        active_paths.patent_evidence_embeddings,
        public_lookup_path=active_paths.public_lookup,
        identity_path=identity_path,
        context_path=context_path,
        patent_text_path=active_paths.patent_text_evidence,
        model=patent_model,
        model_name=patent_model_name,
    )

    print("Step 8/9: Calculating final prioritization")
    scoring_csv(
        active_paths.descriptors,
        active_paths.similarity,
        active_paths.prioritized,
        nlp_path=active_paths.text_nlp,
        public_lookup_path=active_paths.public_lookup,
        surechembl_path=active_paths.surechembl_lookup,
        identity_path=identity_path,
        context_path=context_path,
        chemberta_path=(
            active_paths.chemberta_embeddings if use_chemberta else None
        ),
        nlp_was_run=nlp_was_run,
    )

    if use_chemberta:
        merge_chemberta_into_prioritized(
            active_paths.prioritized,
            active_paths.chemberta_embeddings,
        )
        visualization_coordinates_csv(
            active_paths.chemberta_embeddings,
            active_paths.prioritized,
            active_paths.visualization_coordinates,
        )

    report_ids = select_report_molecule_ids(
        active_paths.prioritized,
        report_molecule=report_molecule,
        report_top_n=report_top_n,
        report_all=report_all,
        report_only_fully_analyzed=report_only_fully_analyzed,
        online_surechembl=online_surechembl,
        use_chemberta=use_chemberta,
    )
    if report_ids:
        active_report_dir = report_dir or (active_paths.prioritized.parent / "reports")
        active_report_dir.mkdir(parents=True, exist_ok=True)
        if clean_report_dir:
            for old_report in active_report_dir.glob(
                "compound_intelligence_report_*.md"
            ):
                old_report.unlink()
        print(f"Step 9/9: Generating {len(report_ids)} compound report(s)")
        for molecule_id in report_ids:
            report_path = active_report_dir / (
                f"compound_intelligence_report_{molecule_id}.md"
            )
            print(f"Generating compound report for {molecule_id}")
            compound_qa(
                molecule_id,
                "full_report",
                report_path,
                prioritized_path=active_paths.prioritized,
                similarity_path=active_paths.similarity_top_hits,
                public_lookup_path=active_paths.public_lookup,
                compound_context_path=context_path,
                chemical_identity_path=identity_path,
                nlp_path=active_paths.text_nlp,
                descriptor_path=active_paths.descriptors,
                surechembl_path=active_paths.surechembl_lookup,
                visualization_path=active_paths.visualization_coordinates,
                image_dir=active_paths.prioritized.parent / "report_images",
            )
            print(f"Compound report complete: {report_path}")
    else:
        print("Step 9/9: Compound report generation skipped")

    print(f"Pipeline complete: {active_paths.prioritized}")
    return active_paths.prioritized

def final_score_range(output_path: Path) -> tuple[float, float]:
    """Read the minimum and maximum NLP-enhanced final scores."""
    with output_path.open("r", encoding="utf-8-sig", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    scores = [
        float(row["prioritization_score_with_nlp"])
        for row in rows
        if row.get("prioritization_score_with_nlp", "").strip()
    ]
    if not scores:
        raise ValueError("Final pipeline output contains no prioritization scores.")
    return min(scores), max(scores)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the default demo workflow."""
    parser = argparse.ArgumentParser(
        description="Run the complete molecule-intelligence workflow."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Generated SMILES CSV with molecule_id and smiles columns.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=DEFAULT_REFERENCES,
        help="Reference molecule CSV for local similarity analysis.",
    )
    parser.add_argument(
        "--text-evidence",
        type=Path,
        default=None,
        help="Text evidence CSV for local NLP scoring.",
    )
    parser.add_argument(
        "--patent-evidence",
        type=Path,
        default=None,
        help="Optional patent/IP-context text evidence CSV for Step 7 embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all intermediate and final pipeline outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory for generated Markdown reports. Default is <output-dir>/reports.",
    )
    parser.add_argument(
        "--online-lookup",
        action="store_true",
        help="Enable PubChem and ChEMBL requests. Default is offline.",
    )
    parser.add_argument(
        "--refresh-public-lookup",
        action="store_true",
        help="Overwrite public lookup output instead of preserving it.",
    )
    parser.add_argument(
        "--online-patent-search",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--online-surechembl",
        action="store_true",
        help="Enable online SureChEMBL chemical search. Default uses local demo data.",
    )
    parser.add_argument(
        "--skip-surechembl",
        action="store_true",
        help="Disable SureChEMBL lookup for this workflow.",
    )
    parser.add_argument(
        "--surechembl-compounds",
        type=Path,
        help="Custom local SureChEMBL structure evidence compound CSV.",
    )
    parser.add_argument(
        "--use-demo-surechembl",
        action="store_true",
        help="Use the bundled demo SureChEMBL structure evidence compounds for a custom run.",
    )
    parser.add_argument(
        "--refresh-patent-search",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        help="Limit the number of molecules sent to public lookup.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of local reference hits retained per valid molecule.",
    )
    parser.add_argument(
        "--report-molecule",
        help="Generate a full local report for this molecule ID.",
    )
    parser.add_argument(
        "--report-top-n",
        type=int,
        help="Generate reports for the top N valid molecules by final score.",
    )
    parser.add_argument(
        "--report-all",
        action="store_true",
        help="Generate reports for all valid molecules.",
    )
    parser.add_argument(
        "--report-only-fully-analyzed",
        action="store_true",
        help=(
            "With --report-top-n, only generate reports for valid molecules "
            "whose requested evidence stages were queried or available."
        ),
    )
    parser.add_argument(
        "--clean-report-dir",
        action="store_true",
        help="Delete old compound_intelligence_report_*.md files before reporting.",
    )
    parser.add_argument(
        "--use-chemberta",
        action="store_true",
        help="Generate optional ChemBERTa embeddings and visualization coordinates.",
    )
    parser.add_argument(
        "--chemberta-model",
        default=DEFAULT_CHEMBERTA_MODEL,
        help="Hugging Face ChemBERTa model name.",
    )
    parser.add_argument(
        "--biomedical-model",
        default=DEFAULT_BIOMEDICAL_MODEL,
        help=(
            "Cached sentence-transformer compatible biomedical evidence model "
            "name. BioBERT/PubMedBERT-style sentence embedding models may be "
            "used when available locally."
        ),
    )
    parser.add_argument(
        "--patent-model",
        default=DEFAULT_PATENT_MODEL,
        help=(
            "Cached sentence-transformer compatible patent/IP-context model "
            "name. PaECTER or patent-BERT-style sentence embedding models may "
            "be used when available locally."
        ),
    )
    parser.add_argument(
        "--refresh-chemberta",
        action="store_true",
        help="Regenerate ChemBERTa embeddings even if the output file exists.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    nlp_model: SentenceEncoder | None = None,
) -> int:
    """Run the pipeline command-line interface."""
    args = build_parser().parse_args(argv)
    custom_workflow = (
        args.input != DEFAULT_INPUT
        or args.references != DEFAULT_REFERENCES
        or args.output_dir != DEFAULT_OUTPUT_DIR
    )
    text_evidence = args.text_evidence
    if text_evidence is None:
        if custom_workflow:
            text_evidence = args.output_dir / "_empty_text_evidence.csv"
            write_empty_text_evidence(text_evidence)
        else:
            text_evidence = DEFAULT_TEXT_EVIDENCE

    skip_surechembl = args.skip_surechembl
    surechembl_compounds = args.surechembl_compounds or DEFAULT_SURECHEMBL_COMPOUNDS
    if custom_workflow and not (
        args.use_demo_surechembl
        or args.surechembl_compounds
        or args.online_surechembl
    ):
        skip_surechembl = True

    paths = build_paths(
        input_path=args.input,
        references_path=args.references,
        text_evidence_path=text_evidence,
        output_dir=args.output_dir,
        surechembl_compounds_path=surechembl_compounds,
    )
    paths = replace(paths, patent_text_evidence=args.patent_evidence)
    report_dir = args.report_dir or (args.output_dir / "reports")
    try:
        output_path = run_pipeline(
            paths=paths,
            nlp_model=nlp_model,
            online_lookup=args.online_lookup,
            max_molecules=args.max_molecules,
            top_k=args.top_k,
            report_molecule=args.report_molecule,
            report_dir=report_dir,
            report_top_n=args.report_top_n,
            report_all=args.report_all,
            report_only_fully_analyzed=args.report_only_fully_analyzed,
            refresh_public_lookup=args.refresh_public_lookup,
            online_surechembl=args.online_surechembl,
            skip_surechembl=skip_surechembl,
            clean_report_dir=args.clean_report_dir,
            use_chemberta=args.use_chemberta,
            chemberta_model=args.chemberta_model,
            biomedical_model_name=args.biomedical_model,
            patent_model_name=args.patent_model,
            refresh_chemberta=args.refresh_chemberta,
        )
        minimum, maximum = final_score_range(output_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Pipeline error: {exc}") from exc

    print(f"Final score range: {minimum:.3f}-{maximum:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
