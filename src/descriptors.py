"""Calculate molecular descriptors and drug-like properties with RDKit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


INPUT_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")
OUTPUT_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "qed",
    "lipinski_violations",
    "lipinski_pass",
    "druglikeness_category",
    "druglikeness_score",
    "druglikeness_flags",
    "mw_status",
    "logp_status",
    "tpsa_status",
    "qed_status",
    "lipinski_status",
    "descriptor_error",
)


@dataclass(frozen=True)
class MolecularDescriptors:
    """Descriptor result for one standardized molecule."""

    molecule_id: str
    canonical_smiles: str
    valid_smiles: bool
    molecular_weight: str = ""
    logp: str = ""
    tpsa: str = ""
    hbd: str = ""
    hba: str = ""
    rotatable_bonds: str = ""
    aromatic_rings: str = ""
    qed: str = ""
    lipinski_violations: str = ""
    lipinski_pass: str = ""
    druglikeness_category: str = "invalid"
    druglikeness_score: str = ""
    druglikeness_flags: str = ""
    mw_status: str = "invalid"
    logp_status: str = "invalid"
    tpsa_status: str = "invalid"
    qed_status: str = "invalid"
    lipinski_status: str = "invalid"
    descriptor_error: str = ""


def range_status(
    value: float,
    *,
    favorable_min: float | None = None,
    favorable_max: float,
    borderline_max: float | None = None,
) -> str:
    """Classify one continuous descriptor using requested boundaries."""
    if favorable_min is not None and value < favorable_min:
        return "borderline"
    if value <= favorable_max:
        return "favorable"
    if borderline_max is not None and value <= borderline_max:
        return "borderline"
    return "unfavorable"


def threshold_status(value: float, favorable_max: float) -> str:
    """Classify a count threshold as favorable or unfavorable."""
    return "favorable" if value <= favorable_max else "unfavorable"


def qed_status(value: float) -> str:
    """Classify QED using the requested favorable/borderline thresholds."""
    if value >= 0.60:
        return "favorable"
    if value >= 0.40:
        return "borderline"
    return "unfavorable"


def summarize_druglikeness(
    *,
    molecular_weight: float,
    logp: float,
    tpsa: float,
    hbd: int,
    hba: int,
    rotatable_bonds: int,
    qed: float,
    lipinski_pass: bool,
) -> tuple[str, float, str, dict[str, str]]:
    """Return overall category, score, flags, and property statuses."""
    statuses = {
        "MW": range_status(
            molecular_weight,
            favorable_min=150,
            favorable_max=500,
            borderline_max=650,
        ),
        "LogP": range_status(
            logp,
            favorable_min=-1,
            favorable_max=5,
            borderline_max=6,
        ),
        "TPSA": range_status(
            tpsa,
            favorable_max=140,
            borderline_max=180,
        ),
        "HBD": threshold_status(hbd, 5),
        "HBA": threshold_status(hba, 10),
        "Rotatable bonds": threshold_status(rotatable_bonds, 10),
        "QED": qed_status(qed),
        "Lipinski": "favorable" if lipinski_pass else "unfavorable",
    }
    points = {"favorable": 1.0, "borderline": 0.5, "unfavorable": 0.0}
    score = sum(points[status] for status in statuses.values()) / len(statuses)
    if "unfavorable" in statuses.values():
        category = "unfavorable"
    elif "borderline" in statuses.values():
        category = "borderline"
    else:
        category = "favorable"
    flags = "; ".join(
        f"{name}: {status}"
        for name, status in statuses.items()
        if status != "favorable"
    )
    return category, score, flags, statuses


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def count_lipinski_violations(
    molecular_weight: float, logp: float, hbd: int, hba: int
) -> int:
    """Count violations of the Lipinski rule-of-five thresholds."""
    return sum(
        (
            molecular_weight > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        )
    )


def passes_lipinski(violations: int) -> bool:
    """Return whether a molecule satisfies all Lipinski thresholds."""
    return violations == 0


def calculate_descriptors(
    molecule_id: str,
    canonical_smiles: str,
    valid_smiles: bool = True,
    upstream_error: str = "",
) -> MolecularDescriptors:
    """Calculate descriptors for one standardized molecule."""
    cleaned_smiles = canonical_smiles.strip()
    if not valid_smiles:
        error = upstream_error.strip() or "Input row is marked as invalid."
        return MolecularDescriptors(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            descriptor_error=error,
        )

    if not cleaned_smiles:
        return MolecularDescriptors(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            descriptor_error="Canonical SMILES is missing.",
        )

    try:
        molecule = Chem.MolFromSmiles(cleaned_smiles)
        if molecule is None:
            raise ValueError("RDKit could not parse the canonical SMILES.")

        molecular_weight = Descriptors.MolWt(molecule)
        logp = Crippen.MolLogP(molecule)
        tpsa = rdMolDescriptors.CalcTPSA(molecule)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(molecule)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(molecule)
        rotatable_bonds = Lipinski.NumRotatableBonds(molecule)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(molecule)
        qed = QED.qed(molecule)
        violations = count_lipinski_violations(
            molecular_weight=molecular_weight,
            logp=logp,
            hbd=hbd,
            hba=hba,
        )
        lipinski_pass = passes_lipinski(violations)
        (
            druglikeness_category,
            druglikeness_score,
            druglikeness_flags,
            statuses,
        ) = summarize_druglikeness(
            molecular_weight=molecular_weight,
            logp=logp,
            tpsa=tpsa,
            hbd=hbd,
            hba=hba,
            rotatable_bonds=rotatable_bonds,
            qed=qed,
            lipinski_pass=lipinski_pass,
        )
    except Exception as exc:
        return MolecularDescriptors(
            molecule_id=molecule_id,
            canonical_smiles=canonical_smiles,
            valid_smiles=False,
            descriptor_error=str(exc),
        )

    return MolecularDescriptors(
        molecule_id=molecule_id,
        canonical_smiles=cleaned_smiles,
        valid_smiles=True,
        molecular_weight=f"{molecular_weight:.3f}",
        logp=f"{logp:.3f}",
        tpsa=f"{tpsa:.3f}",
        hbd=str(hbd),
        hba=str(hba),
        rotatable_bonds=str(rotatable_bonds),
        aromatic_rings=str(aromatic_rings),
        qed=f"{qed:.3f}",
        lipinski_violations=str(violations),
        lipinski_pass=str(lipinski_pass),
        druglikeness_category=druglikeness_category,
        druglikeness_score=f"{druglikeness_score:.3f}",
        druglikeness_flags=druglikeness_flags,
        mw_status=statuses["MW"],
        logp_status=statuses["LogP"],
        tpsa_status=statuses["TPSA"],
        qed_status=statuses["QED"],
        lipinski_status=statuses["Lipinski"],
    )


def calculate_rows(
    rows: Iterable[Mapping[str, str]],
) -> list[MolecularDescriptors]:
    """Calculate descriptors for standardized CSV rows."""
    return [
        calculate_descriptors(
            molecule_id=row.get("molecule_id", "").strip(),
            canonical_smiles=row.get("canonical_smiles", ""),
            valid_smiles=parse_boolean(row.get("valid_smiles", "")),
            upstream_error=row.get("error_message", ""),
        )
        for row in rows
    ]


def read_input_csv(input_path: Path) -> list[dict[str, str]]:
    """Read a standardized CSV and validate its required columns."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(INPUT_COLUMNS) - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        return [dict(row) for row in reader]


def write_output_csv(
    output_path: Path, records: Iterable[MolecularDescriptors]
) -> None:
    """Write molecular descriptor records to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            writer.writerow(row)


def descriptor_csv(input_path: Path, output_path: Path) -> int:
    """Calculate descriptors from a standardized CSV and return row count."""
    records = calculate_rows(read_input_csv(input_path))
    write_output_csv(output_path, records)
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate RDKit descriptors for standardized molecules."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Standardized input CSV containing canonical_smiles.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV for molecular descriptors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the descriptor calculation command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        row_count = descriptor_csv(args.input, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {row_count} descriptor records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
