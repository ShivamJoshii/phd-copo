from __future__ import annotations

from pathlib import Path

def normalize_keys(row: dict) -> dict[str, object]:
    """Lowercase and strip a row's keys so column lookups are case-insensitive."""
    return {str(k).strip().lower(): v for k, v in row.items() if k is not None}


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def decode_text_bytes(data: bytes, *, source: str = "input") -> str:
    """Decode bytes using a small set of common encodings used by CSV/JSON exports."""
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"{source}: could not decode text file. "
        "Please upload UTF-8/UTF-8-BOM/CP1252/Latin-1 encoded files."
    )


def read_text_file(path: Path) -> str:
    """Read text from path with encoding fallback for common spreadsheet exports."""
    return decode_text_bytes(path.read_bytes(), source=path.name)
