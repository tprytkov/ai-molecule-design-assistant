"""Small ADMET benchmark loading helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

BENCHMARK_COLUMNS = ("molecule_id", "smiles", "label", "split")


def read_local_benchmark_csv(path: Path) -> list[dict[str, str]]:
    """Read a local public-safe benchmark CSV with SMILES, label, and split."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(BENCHMARK_COLUMNS) - set(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if missing:
        raise ValueError("Benchmark CSV is missing required columns: " + ", ".join(sorted(missing)))
    return rows


def split_rows(rows: Iterable[Mapping[str, str]], split: str = "test") -> list[dict[str, str]]:
    """Return rows matching a benchmark split."""
    requested = str(split or "").strip().lower()
    return [dict(row) for row in rows if str(row.get("split", "")).strip().lower() == requested]
