"""Compare generated molecules with public references using RDKit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


GENERATED_INPUT_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")
REFERENCE_INPUT_COLUMNS: tuple[str, ...] = ()
OUTPUT_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "best_reference_id",
    "best_reference_name",
    "best_reference_smiles",
    "tanimoto_similarity",
    "similarity_category",
    "similarity_error",
)

MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_FP_SIZE,
)


@dataclass(frozen=True)
class ReferenceFingerprint:
    """A valid public reference molecule and its Morgan fingerprint."""

    reference_id: str
    name: str
    smiles: str
    fingerprint: ExplicitBitVect


@dataclass(frozen=True)
class SimilarityResult:
    """Best-reference similarity result for one generated molecule."""

    molecule_id: str
    canonical_smiles: str
    valid_smiles: bool
    best_reference_id: str = ""
    best_reference_name: str = ""
    best_reference_smiles: str = ""
    tanimoto_similarity: str = ""
    similarity_category: str = "not_available"
    similarity_error: str = ""


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def categorize_similarity(similarity: float | None) -> str:
    """Assign a similarity category from a Tanimoto score."""
    if similarity is None:
        return "not_available"
    if similarity >= 0.85:
        return "very_close_analog"
    if similarity >= 0.70:
        return "related_chemotype"
    if similarity >= 0.50:
        return "moderate_similarity"
    return "structurally_distinct"


def create_morgan_fingerprint(smiles: str) -> ExplicitBitVect:
    """Parse a SMILES and create a radius-2, 2048-bit Morgan fingerprint."""
    cleaned_smiles = smiles.strip()
    if not cleaned_smiles:
        raise ValueError("SMILES is missing.")

    molecule = Chem.MolFromSmiles(cleaned_smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse the SMILES.")
    return MORGAN_GENERATOR.GetFingerprint(molecule)


def prepare_references(
    rows: Iterable[Mapping[str, str]],
) -> list[ReferenceFingerprint]:
    """Create fingerprints for valid references and skip invalid records."""
    references: list[ReferenceFingerprint] = []
    for index, row in enumerate(rows, start=1):
        smiles = (
            row.get("canonical_smiles", "").strip()
            or row.get("smiles", "").strip()
        )
        try:
            fingerprint = create_morgan_fingerprint(smiles)
        except (ValueError, RuntimeError):
            continue

        references.append(
            ReferenceFingerprint(
                reference_id=(
                    row.get("reference_id", "").strip()
                    or f"ref_{index:04d}"
                ),
                name=(
                    row.get("reference_name", "").strip()
                    or row.get("name", "").strip()
                ),
                smiles=smiles,
                fingerprint=fingerprint,
            )
        )
    return references


def calculate_best_similarity(
    molecule_id: str,
    canonical_smiles: str,
    valid_smiles: bool,
    references: Sequence[ReferenceFingerprint],
    upstream_error: str = "",
) -> SimilarityResult:
    """Find the most similar valid reference for one generated molecule."""
    cleaned_smiles = canonical_smiles.strip()
    if not valid_smiles:
        error = upstream_error.strip() or "Input row is marked as invalid."
        return SimilarityResult(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            similarity_error=error,
        )

    if not cleaned_smiles:
        return SimilarityResult(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            similarity_error="Canonical SMILES is missing.",
        )

    if not references:
        return SimilarityResult(
            molecule_id=molecule_id,
            canonical_smiles=cleaned_smiles,
            valid_smiles=True,
            similarity_error="No valid reference molecules are available.",
        )

    try:
        fingerprint = create_morgan_fingerprint(cleaned_smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            [reference.fingerprint for reference in references],
        )
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        best_reference = references[best_index]
        best_similarity = similarities[best_index]
    except Exception as exc:
        return SimilarityResult(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            similarity_error=str(exc),
        )

    return SimilarityResult(
        molecule_id=molecule_id,
        canonical_smiles=cleaned_smiles,
        valid_smiles=True,
        best_reference_id=best_reference.reference_id,
        best_reference_name=best_reference.name,
        best_reference_smiles=best_reference.smiles,
        tanimoto_similarity=f"{best_similarity:.3f}",
        similarity_category=categorize_similarity(best_similarity),
    )


def calculate_rows(
    generated_rows: Iterable[Mapping[str, str]],
    references: Sequence[ReferenceFingerprint],
) -> list[SimilarityResult]:
    """Calculate best-reference similarity for generated CSV rows."""
    return [
        calculate_best_similarity(
            molecule_id=row.get("molecule_id", "").strip(),
            canonical_smiles=row.get("canonical_smiles", ""),
            valid_smiles=parse_boolean(row.get("valid_smiles", "")),
            references=references,
            upstream_error=row.get("descriptor_error", ""),
        )
        for row in generated_rows
    ]


def read_generated_csv(input_path: Path) -> list[dict[str, str]]:
    """Read generated molecules and validate required columns."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(GENERATED_INPUT_COLUMNS) - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Generated input CSV is missing required columns: {missing}"
            )
        return [dict(row) for row in reader]


def read_reference_csv(reference_path: Path) -> list[dict[str, str]]:
    """Read references and accept either smiles or canonical_smiles."""
    with reference_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as reference_file:
        reader = csv.DictReader(reference_file)
        fieldnames = set(reader.fieldnames or [])
        if not {"name", "reference_name"} & fieldnames:
            raise ValueError(
                "Reference CSV must contain name or reference_name."
            )
        if not {"smiles", "canonical_smiles"} & fieldnames:
            raise ValueError(
                "Reference CSV must contain smiles or canonical_smiles."
            )
        return [dict(row) for row in reader]


def write_output_csv(
    output_path: Path, records: Iterable[SimilarityResult]
) -> None:
    """Write molecular similarity results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def similarity_csv(
    input_path: Path, reference_path: Path, output_path: Path
) -> int:
    """Calculate similarities from CSV files and return the output row count."""
    generated_rows = read_generated_csv(input_path)
    reference_rows = read_reference_csv(reference_path)
    references = prepare_references(reference_rows)
    results = calculate_rows(generated_rows, references)
    write_output_csv(output_path, results)
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare generated molecules with public references."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Descriptor CSV containing generated canonical SMILES.",
    )
    parser.add_argument(
        "--references",
        required=True,
        type=Path,
        help="CSV containing public reference molecules.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV for best-reference similarities.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the molecular similarity command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        row_count = similarity_csv(args.input, args.references, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {row_count} similarity records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
