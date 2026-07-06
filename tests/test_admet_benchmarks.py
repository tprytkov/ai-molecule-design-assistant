from pathlib import Path

import pytest

from src.admet.admet_benchmarks import read_local_benchmark_csv, split_rows


def test_read_local_benchmark_csv_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.csv"
    path.write_text("molecule_id,smiles,label,split\nm1,CCO,1,test\n", encoding="utf-8")

    rows = read_local_benchmark_csv(path)

    assert rows == [{"molecule_id": "m1", "smiles": "CCO", "label": "1", "split": "test"}]


def test_read_local_benchmark_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.csv"
    path.write_text("molecule_id,smiles,label\nm1,CCO,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        read_local_benchmark_csv(path)


def test_split_rows_filters_requested_split() -> None:
    rows = [
        {"molecule_id": "m1", "split": "train"},
        {"molecule_id": "m2", "split": "test"},
    ]

    assert split_rows(rows, "test") == [{"molecule_id": "m2", "split": "test"}]
