"""Pure helpers behind the Streamlit UI's raw faculty-export ingestion controls.

The Streamlit app receives uploads as bytes while the :mod:`copo_mapper.ingest`
parsers take paths, so these helpers bridge the two (temp-file round trip) and
package the parse -> canonicalize -> CSV steps as small functions that can be
unit tested without importing streamlit.
"""

from __future__ import annotations

import csv
import tempfile
from io import StringIO
from pathlib import Path
from typing import Sequence

from .ingest import (
    CoRecord,
    PoRecord,
    list_courses,
    parse_real_co_file,
    parse_real_po_file,
    to_canonical_co_rows,
    to_canonical_po_rows,
)

ALL_COURSES_OPTION = "All courses (prefixed ids)"
PO_KIND_ORDER = ("PO", "PSO", "PEO")
DEFAULT_PO_KINDS = ("PO", "PSO")

_COURSE_LABEL_SEPARATOR = " — "


def _parse_upload_bytes(data: bytes, parser):
    """Write upload bytes to a temp file and run a path-based ingest parser."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tf.write(data)
        tmp_name = tf.name
    try:
        return parser(tmp_name)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def parse_raw_co_bytes(data: bytes) -> list[CoRecord]:
    """Parse raw CO export bytes into records; raise ValueError if none found."""
    records = _parse_upload_bytes(data, parse_real_co_file)
    if not records:
        raise ValueError(
            "No CO statements detected in the uploaded file. "
            "Expected lines like 'CO1: <statement>' (optionally under course header rows)."
        )
    return records


def parse_raw_po_bytes(data: bytes) -> list[PoRecord]:
    """Parse raw PO export bytes into records; raise ValueError if none found."""
    records = _parse_upload_bytes(data, parse_real_po_file)
    if not records:
        raise ValueError(
            "No PO/PSO/PEO statements detected in the uploaded file. "
            "Expected lines like 'PO1: <statement>', 'PSO1: ...' or 'PEO1: ...'."
        )
    return records


def course_options(records: Sequence[CoRecord]) -> list[str]:
    """Selectbox labels: the all-courses option followed by each detected course."""
    options = [ALL_COURSES_OPTION]
    for code, name in list_courses(records):
        options.append(f"{code}{_COURSE_LABEL_SEPARATOR}{name}" if name else code)
    return options


def course_from_option(option: str) -> str | None:
    """Map a selectbox label back to a course code (None = all courses)."""
    if option == ALL_COURSES_OPTION:
        return None
    return option.split(_COURSE_LABEL_SEPARATOR, 1)[0].strip()


def available_po_kinds(records: Sequence[PoRecord]) -> list[str]:
    """Kinds present in the parsed PO file, in PO -> PSO -> PEO display order."""
    present = {record.kind for record in records}
    return [kind for kind in PO_KIND_ORDER if kind in present]


def canonical_rows_from_raw_co_bytes(
    data: bytes,
    course: str | None = None,
) -> list[dict[str, str]]:
    """Parse raw CO bytes and convert to canonical ``CO,description`` rows."""
    return to_canonical_co_rows(parse_raw_co_bytes(data), course=course)


def canonical_rows_from_raw_po_bytes(
    data: bytes,
    include: Sequence[str] = DEFAULT_PO_KINDS,
) -> list[dict[str, str]]:
    """Parse raw PO bytes and convert to canonical ``PO,description`` rows."""
    records = parse_raw_po_bytes(data)
    rows = to_canonical_po_rows(records, include=include)
    if not rows:
        found = ", ".join(available_po_kinds(records))
        wanted = ", ".join(include) or "none"
        raise ValueError(
            f"No statements matched the selected kinds ({wanted}). "
            f"The file contains: {found}."
        )
    return rows


def _canonical_csv_text(rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def canonical_co_csv_text(rows: Sequence[dict[str, str]]) -> str:
    """Render canonical CO rows as ``CO,description`` CSV text."""
    return _canonical_csv_text(rows, fieldnames=("CO", "description"))


def canonical_po_csv_text(rows: Sequence[dict[str, str]]) -> str:
    """Render canonical PO rows as ``PO,description`` CSV text."""
    return _canonical_csv_text(rows, fieldnames=("PO", "description"))
