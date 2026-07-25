from __future__ import annotations


def normalize_keys(row: dict) -> dict[str, object]:
    """Lowercase and strip a row's keys so column lookups are case-insensitive."""
    return {str(k).strip().lower(): v for k, v in row.items() if k is not None}
