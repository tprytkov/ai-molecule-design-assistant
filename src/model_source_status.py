"""App-managed model and public-data cache status helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = PROJECT_ROOT / "app_data"
MODEL_CACHE_DIR = APP_DATA_DIR / "model_cache"
HUGGINGFACE_CACHE_DIR = MODEL_CACHE_DIR / "huggingface"
PUBLIC_LOOKUP_CACHE_DIR = APP_DATA_DIR / "public_lookup_cache"
MANIFEST_DIR = APP_DATA_DIR / "manifests"
MODEL_MANIFEST_PATH = MANIFEST_DIR / "model_manifest.json"
PUBLIC_DATA_MANIFEST_PATH = MANIFEST_DIR / "public_data_manifest.json"
RUN_MANIFEST_PATH = MANIFEST_DIR / "run_manifest.json"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for manifest rows."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_app_data_dirs() -> None:
    """Create app-managed cache and manifest folders."""
    for path in (
        APP_DATA_DIR,
        MODEL_CACHE_DIR,
        HUGGINGFACE_CACHE_DIR,
        PUBLIC_LOOKUP_CACHE_DIR,
        MANIFEST_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_huggingface_cache_env() -> None:
    """Default Hugging Face cache env vars to app-managed folders."""
    ensure_app_data_dirs()
    os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HUGGINGFACE_CACHE_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HUGGINGFACE_CACHE_DIR))


def read_manifest(path: Path) -> dict[str, object]:
    """Read a JSON manifest if it exists and is valid."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON manifest atomically enough for local Streamlit usage."""
    ensure_app_data_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def huggingface_model_cache_candidates(model_id: str) -> list[Path]:
    """Return likely app-managed Hugging Face cache paths for a model ID."""
    if not model_id:
        return []
    safe_id = model_id.replace("/", "--")
    flat_id = model_id.replace("/", "_")
    return [
        HUGGINGFACE_CACHE_DIR / f"models--{safe_id}",
        HUGGINGFACE_CACHE_DIR / model_id,
        HUGGINGFACE_CACHE_DIR / flat_id,
        HUGGINGFACE_CACHE_DIR / "sentence_transformers" / flat_id,
    ]


def model_is_cached(model_id: str) -> bool:
    """Return whether a model appears in the app-managed Hugging Face cache."""
    return any(path.exists() for path in huggingface_model_cache_candidates(model_id))


def model_cache_path(model_id: str) -> str:
    """Return the most specific app-managed cache path for a model ID."""
    for path in huggingface_model_cache_candidates(model_id):
        if path.exists():
            return str(path)
    candidates = huggingface_model_cache_candidates(model_id)
    return str(candidates[0]) if candidates else str(HUGGINGFACE_CACHE_DIR)


@dataclass(frozen=True)
class ModelSourceRecord:
    """Manifest row for one model role."""

    role: str
    configured_model: str
    fallback_model: str
    actual_model_used: str
    cache_path: str
    cached: bool
    downloadable: bool
    fallback_used: bool
    status: str
    error_message: str
    last_checked: str


def build_model_record(
    *,
    role: str,
    configured_model: str,
    fallback_model: str,
    actual_model_used: str = "",
    status: str = "not_checked",
    error_message: str = "",
    downloadable: bool = True,
) -> ModelSourceRecord:
    """Build a model manifest record from app state."""
    cache_model = actual_model_used or configured_model or fallback_model
    return ModelSourceRecord(
        role=role,
        configured_model=configured_model,
        fallback_model=fallback_model,
        actual_model_used=actual_model_used,
        cache_path=model_cache_path(cache_model),
        cached=model_is_cached(cache_model),
        downloadable=downloadable,
        fallback_used=bool(actual_model_used and actual_model_used == fallback_model),
        status=status,
        error_message=error_message,
        last_checked=utc_timestamp(),
    )


def update_model_manifest(records: Iterable[ModelSourceRecord]) -> dict[str, object]:
    """Merge model records into the model manifest and return the payload."""
    existing = read_manifest(MODEL_MANIFEST_PATH)
    model_records = {
        str(key): value
        for key, value in (existing.get("models") or {}).items()
        if isinstance(value, dict)
    }
    for record in records:
        model_records[record.role] = asdict(record)
    payload = {
        "last_checked": utc_timestamp(),
        "cache_root": str(HUGGINGFACE_CACHE_DIR),
        "downloads_enabled": os.environ.get("ALLOW_LOCAL_MODEL_DOWNLOADS") == "1",
        "models": model_records,
    }
    write_manifest(MODEL_MANIFEST_PATH, payload)
    return payload


def source_status_from_frame(path: Path, status_columns: Iterable[str]) -> dict[str, object]:
    """Return a compact public-source status without importing pandas globally."""
    exists = path.exists()
    result: dict[str, object] = {
        "path": str(path),
        "exists": exists,
        "rows": 0,
        "statuses": {},
    }
    if not exists:
        return result
    try:
        import pandas as pd

        frame = pd.read_csv(path).fillna("")
    except Exception as exc:
        result["error_message"] = f"{type(exc).__name__}: {exc}"
        return result
    result["rows"] = int(len(frame))
    statuses: dict[str, dict[str, int]] = {}
    for column in status_columns:
        if column not in frame.columns:
            continue
        counts = frame[column].astype(str).str.strip().value_counts().to_dict()
        statuses[column] = {str(key): int(value) for key, value in counts.items()}
    result["statuses"] = statuses
    return result


def update_public_data_manifest(output_dir: Path | None = None) -> dict[str, object]:
    """Write a public data/source status manifest for the active output folder."""
    output_dir = Path(output_dir) if output_dir else None
    sources: dict[str, object] = {}
    if output_dir is not None:
        sources["PubChem"] = source_status_from_frame(
            output_dir / "public_lookup.csv",
            ("pubchem_status",),
        )
        sources["ChEMBL"] = source_status_from_frame(
            output_dir / "public_lookup.csv",
            ("chembl_status",),
        )
        sources["SureChEMBL"] = source_status_from_frame(
            output_dir / "surechembl_evidence.csv",
            ("lookup_status", "surechembl_query_status"),
        )
    payload = {
        "last_checked": utc_timestamp(),
        "cache_root": str(PUBLIC_LOOKUP_CACHE_DIR),
        "output_dir": str(output_dir) if output_dir else "",
        "sources": sources,
    }
    write_manifest(PUBLIC_DATA_MANIFEST_PATH, payload)
    return payload


def update_run_manifest(output_dir: Path | None = None) -> dict[str, object]:
    """Write a latest-run manifest pointer."""
    payload = {
        "last_checked": utc_timestamp(),
        "output_dir": str(output_dir) if output_dir else "",
    }
    write_manifest(RUN_MANIFEST_PATH, payload)
    return payload


def initialize_manifests() -> None:
    """Ensure all app-managed manifests exist."""
    ensure_app_data_dirs()
    for path, payload in (
        (
            MODEL_MANIFEST_PATH,
            {
                "last_checked": "",
                "cache_root": str(HUGGINGFACE_CACHE_DIR),
                "downloads_enabled": os.environ.get("ALLOW_LOCAL_MODEL_DOWNLOADS") == "1",
                "models": {},
            },
        ),
        (
            PUBLIC_DATA_MANIFEST_PATH,
            {
                "last_checked": "",
                "cache_root": str(PUBLIC_LOOKUP_CACHE_DIR),
                "output_dir": "",
                "sources": {},
            },
        ),
        (RUN_MANIFEST_PATH, {"last_checked": "", "output_dir": ""}),
    ):
        if not path.exists():
            write_manifest(path, payload)
