"""Lightweight lexical similarity helpers for deployment-safe fallbacks."""

from __future__ import annotations

import math
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def token_set(text: str) -> set[str]:
    """Return normalized alphanumeric tokens from free text."""
    return set(TOKEN_PATTERN.findall(str(text).lower()))


def lexical_similarity(left: str, right: str) -> float:
    """Return cosine-style token-overlap similarity for two text strings."""
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens.intersection(right_tokens))
    return overlap / math.sqrt(len(left_tokens) * len(right_tokens))
