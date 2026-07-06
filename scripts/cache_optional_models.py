"""Explicitly cache optional Hugging Face domain models for local testing."""

from __future__ import annotations

import argparse
from typing import Sequence

from src.optional_domain_models import cache_huggingface_model


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
    parser.add_argument(
        "--backend",
        default="transformers",
        help="Model backend hint: transformers, sentence-transformers, or transformers-sequence-classification.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Cache requested models and print per-model status."""
    args = build_parser().parse_args(argv)
    exit_code = 0
    for model_id in args.models:
        model, ok, message = cache_huggingface_model(model_id, backend=args.backend)
        status = "SUCCESS" if ok else "FAILED"
        print(f"{status}\t{model}\t{message}")
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
