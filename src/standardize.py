"""Validate and standardize molecular SMILES with RDKit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.io.molecule_table_input import normalize_molecule_rows


INPUT_COLUMNS = ("molecule_id", "smiles")
OUTPUT_COLUMNS = (
    "molecule_id",
    "smiles",
    "valid_smiles",
    "canonical_smiles",
    "inchi_key",
    "molecular_formula",
    "molecular_weight",
    "error_message",
)


@dataclass(frozen=True)
class StandardizedMolecule:
    """Result of validating and standardizing one input record."""

    molecule_id: str
    smiles: str
    valid_smiles: bool
    canonical_smiles: str = ""
    inchi_key: str = ""
    molecular_formula: str = ""
    molecular_weight: str = ""
    error_message: str = ""


def standardize_record(molecule_id: str, smiles: str) -> StandardizedMolecule:
    """Validate one SMILES and calculate identifiers and basic properties."""
    cleaned_smiles = smiles.strip()
    if not cleaned_smiles:
        return StandardizedMolecule(
            molecule_id=molecule_id,
            smiles=smiles,
            valid_smiles=False,
            error_message="SMILES is empty.",
        )

    try:
        molecule = Chem.MolFromSmiles(cleaned_smiles)
        if molecule is None:
            raise ValueError("RDKit could not parse the SMILES.")

        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        inchi_key = Chem.MolToInchiKey(molecule)
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        molecular_weight = f"{Descriptors.MolWt(molecule):.3f}"
    except Exception as exc:
        return StandardizedMolecule(
            molecule_id=molecule_id,
            smiles=smiles,
            valid_smiles=False,
            error_message=str(exc),
        )

    return StandardizedMolecule(
        molecule_id=molecule_id,
        smiles=smiles,
        valid_smiles=True,
        canonical_smiles=canonical_smiles,
        inchi_key=inchi_key,
        molecular_formula=formula,
        molecular_weight=molecular_weight,
    )


def standardize_rows(
    rows: Iterable[Mapping[str, str]],
) -> list[StandardizedMolecule]:
    """Standardize rows and retain only the first valid canonical duplicate."""
    results: list[StandardizedMolecule] = []
    seen_canonical_smiles: set[str] = set()

    for row in rows:
        result = standardize_record(
            molecule_id=row.get("molecule_id", "").strip(),
            smiles=row.get("smiles", ""),
        )
        if result.valid_smiles:
            if result.canonical_smiles in seen_canonical_smiles:
                continue
            seen_canonical_smiles.add(result.canonical_smiles)
        results.append(result)

    return results


def read_input_csv(input_path: Path) -> list[dict[str, str]]:
    """Read and validate the required columns from an input CSV."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = [dict(row) for row in reader]
    return [
        {"molecule_id": row["molecule_id"], "smiles": row["smiles"]}
        for row in normalize_molecule_rows(rows)
    ]


def write_output_csv(
    output_path: Path, records: Iterable[StandardizedMolecule]
) -> None:
    """Write standardized records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def standardize_csv(input_path: Path, output_path: Path) -> int:
    """Standardize an input CSV, write the output, and return its row count."""
    records = standardize_rows(read_input_csv(input_path))
    write_output_csv(output_path, records)
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate and standardize generated molecular SMILES."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV containing molecule_id and smiles columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV for standardized records.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standardization command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        row_count = standardize_csv(args.input, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {row_count} standardized records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
