"""Stable hashing utilities for reproducible seeds/signatures."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _normalize_seed_input(value: object) -> str:
    """Normalize a value into a deterministic string representation."""
    if isinstance(value, tuple):
        return "|".join(_normalize_seed_input(item) for item in value)
    if isinstance(value, list):
        return "|".join(_normalize_seed_input(item) for item in value)
    if isinstance(value, dict):
        return "|".join(
            f"{_normalize_seed_input(k)}:{_normalize_seed_input(v)}"
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(value, set):
        return "|".join(sorted(_normalize_seed_input(item) for item in value))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return "|".join(_normalize_seed_input(item) for item in value)
    return str(value)


def stable_int_seed(string_or_tuple: object) -> int:
    """Return a stable unsigned int32 seed from arbitrary input."""
    normalized = _normalize_seed_input(string_or_tuple)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
