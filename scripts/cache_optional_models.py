"""Explicitly cache optional Hugging Face domain models for local testing."""

from __future__ import annotations

import argparse
from typing import Sequence

from src.model_source_status import HUGGINGFACE_CACHE_DIR, ensure_app_data_dirs


def cache_model(model_id: str) -> tuple[str, bool, str]:
    """Download/cache tokenizer and model for one Hugging Face model ID."""
    ensure_app_data_dirs()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        return model_id, False, f"transformers is not installed: {exc}"
    try:
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(HUGGINGFACE_CACHE_DIR))
        AutoModel.from_pretrained(model_id, cache_dir=str(HUGGINGFACE_CACHE_DIR))
    except Exception as exc:
        return model_id, False, f"{type(exc).__name__}: {exc}"
    return model_id, True, "cached"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Cache optional local domain models for offline app testing."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Hugging Face model IDs to download/cache explicitly.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Cache requested models and print per-model status."""
    args = build_parser().parse_args(argv)
    exit_code = 0
    for model_id in args.models:
        model, ok, message = cache_model(model_id)
        status = "SUCCESS" if ok else "FAILED"
        print(f"{status}\t{model}\t{message}")
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
