"""Load and write target profile metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.target.target_schema import (
    TARGET_DISCLAIMER,
    TARGET_PROFILE_COLUMNS,
    TARGET_SOURCE_DEMO,
    TARGET_SOURCE_GENERAL_DEMO,
    TARGET_SOURCE_MISSING,
    TARGET_SOURCE_TARGET_SPECIFIC_DEMO,
    TARGET_SOURCE_USER,
)

DEMO_TARGET_PROFILE_PATH = Path("data/demo_target/target_profile.csv")
TARGET_SPECIFIC_DEMO_DIR = Path("data/demo_target_specific")
TARGET_SPECIFIC_DEMO_PROFILE_PATH = TARGET_SPECIFIC_DEMO_DIR / "target_profile.csv"
TARGET_SPECIFIC_DEMO_MOLECULES_PATH = TARGET_SPECIFIC_DEMO_DIR / "demo_molecules.csv"
TARGET_SPECIFIC_DEMO_REFERENCES_PATH = TARGET_SPECIFIC_DEMO_DIR / "reference_ligands.csv"
TARGET_SPECIFIC_DEMO_DOCKING_PATH = TARGET_SPECIFIC_DEMO_DIR / "demo_docking_results.csv"


@dataclass(frozen=True)
class TargetProfile:
    """One project target profile."""

    target_id: str
    target_name: str
    gene_symbol: str = ""
    organism: str = ""
    uniprot_id: str = ""
    pdb_id: str = ""
    protein_structure_source: str = ""
    binding_site_description: str = ""
    disease_context: str = ""
    mechanism_context: str = ""
    reference_ligands: str = ""
    docking_protocol_note: str = ""
    target_relevance_note: str = ""
    disclaimer: str = TARGET_DISCLAIMER
    target_source: str = TARGET_SOURCE_MISSING

    def as_csv_row(self) -> dict[str, str]:
        """Return the public target-profile CSV row."""
        values = self.__dict__.copy()
        values.pop("target_source", None)
        return {column: str(values.get(column, "") or "") for column in TARGET_PROFILE_COLUMNS}


def _clean_row(row: Mapping[str, object]) -> dict[str, str]:
    return {column: str(row.get(column, "") or "").strip() for column in TARGET_PROFILE_COLUMNS}


def missing_target_profile() -> TargetProfile:
    """Return an explicit missing-target placeholder."""
    return TargetProfile(
        target_id="target_missing",
        target_name="Target profile missing",
        docking_protocol_note="No target profile CSV was provided.",
        target_relevance_note="Target-aware interpretation is unavailable until a target profile is provided.",
        target_source=TARGET_SOURCE_MISSING,
    )


def load_target_profile(path: Path | None = None) -> TargetProfile:
    """Load the first target profile row, falling back to an explicit missing profile."""
    source_path = path or DEMO_TARGET_PROFILE_PATH
    if source_path is None or not Path(source_path).is_file():
        return missing_target_profile()
    with Path(source_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(TARGET_PROFILE_COLUMNS) - fieldnames
        if missing:
            raise ValueError(
                "Target profile CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        row = next(reader, None)
    if not row:
        return missing_target_profile()
    values = _clean_row(row)
    source = classify_target_source(Path(source_path), values)
    return TargetProfile(**values, target_source=source)


def classify_target_source(path: Path, values: Mapping[str, str] | None = None) -> str:
    """Classify target metadata as general demo, target demo, user, or missing."""
    normalized_path = Path(path)
    try:
        normalized_path = normalized_path.resolve()
    except OSError:
        pass
    try:
        placeholder_path = DEMO_TARGET_PROFILE_PATH.resolve()
        target_demo_path = TARGET_SPECIFIC_DEMO_PROFILE_PATH.resolve()
    except OSError:
        placeholder_path = DEMO_TARGET_PROFILE_PATH
        target_demo_path = TARGET_SPECIFIC_DEMO_PROFILE_PATH
    target_id = str((values or {}).get("target_id", "")).strip()
    if normalized_path == placeholder_path or target_id == "demo_target_placeholder":
        return TARGET_SOURCE_GENERAL_DEMO
    if normalized_path == target_demo_path or target_id == "adora2a_xanthine_demo":
        return TARGET_SOURCE_TARGET_SPECIFIC_DEMO
    if "data/demo_target" in normalized_path.as_posix().replace("\\", "/"):
        return TARGET_SOURCE_DEMO
    return TARGET_SOURCE_USER


def target_profile_csv(
    output_path: Path,
    *,
    source_path: Path | None = None,
) -> TargetProfile:
    """Write the selected target profile to an output CSV."""
    profile = load_target_profile(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerow(profile.as_csv_row())
    return profile
