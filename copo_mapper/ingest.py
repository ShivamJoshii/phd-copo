"""Parsers that turn messy real-world CO/PO exports into canonical pipeline CSVs.

Real institutional exports (e.g. "CO (1).csv" / "PO (1).csv") are single-column
CSV dumps mixing course header rows, wrapped course names, CO statements with
inconsistent id formats ("CO 1:", "CO2:", "CO1.", "CO4-", even the typo "C0 3"),
CSV-quoted lines, cp1252 curly quotes, non-breaking spaces, and U+FFFD mojibake.

This module parses those files into structured records and converts them to the
canonical ``CO,description`` / ``PO,description`` rows that
``copo_mapper.pipeline.run_pairwise_mapping`` expects.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .io_utils import read_text_file


@dataclass(frozen=True)
class CoRecord:
    course_code: str
    course_name: str
    co_id: str
    description: str


@dataclass(frozen=True)
class PoRecord:
    po_id: str
    description: str
    kind: str  # one of: "PEO", "PO", "PSO"


# Course header lines look like "1 KMBN101MANAGEMENT CONCEPTS ..." with an
# optional leading serial number, a KMB* course code (sometimes containing a
# stray space, e.g. "KMBN HR05"), and the course name glued on with or without
# a separating space.
_COURSE_HEADER_RE = re.compile(r"^(?:\d+\s+)?(KMB[A-Z]*(?:\s[A-Z]+)?\d+)\s*(.*)$")

# CO lines: "CO 1: text", "CO2: text", "CO1. text", "CO4-text", "CO1 text",
# "CO1Understand ..." (no separator at all) and the typo "C0 3: text".
_CO_LINE_RE = re.compile(r"^C[O0]\s*(\d+)\s*[:.\-]*\s*(.*)$")

# PO/PEO/PSO lines: "PEO1: ...", "PO1: ...", "PSO2: ..." (PEO/PSO before PO
# so the longer prefixes win).
_PO_LINE_RE = re.compile(r"^(PEO|PSO|PO)\s*(\d+)\s*[:.\-]*\s*(.*)$")

_CHAR_TRANSLATIONS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}


def clean_text(text: str) -> str:
    """Normalize one raw line: fix smart punctuation, mojibake, quotes, spacing."""
    for source, replacement in _CHAR_TRANSLATIONS.items():
        text = text.replace(source, replacement)
    # U+FFFD between word characters is almost always a lost apostrophe
    # (e.g. "firm�s" -> "firm's"); elsewhere it is dropped as junk.
    text = re.sub(r"(?<=\w)�(?=\w)", "'", text)
    text = text.replace("�", " ")
    # Transliterate remaining non-ascii (accents etc.) and drop what is left.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip()
    return text


# --- Trailing institutional-footer boilerplate ------------------------------
#
# Page footers from the source PDF occasionally get glued onto the last CO of a
# page, e.g. "... communication skills. I.T.S Engineering College Greater Noida
# Department of MBA". The stripper below is deliberately conservative:
#
# * it only ever removes a *trailing* segment that starts after sentence-ending
#   punctuation (". Foo", "! Foo", or glued ".Foo"), never text mid-sentence;
# * the segment must be *dominated* by institutional markers (college /
#   university / institute names, "Department of X", campus/city tails) -- at
#   most two non-marker words may remain once markers are removed;
# * the kept text must still be a plausible outcome (>= 4 words), so a record
#   that is nothing but a footer is left untouched (total-consumption guard).

# Sentence boundary: end punctuation followed by whitespace-then-text, or glued
# directly to a capital letter ("...skills.I.T.S Engineering College").
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?=\s+\S|[A-Z])")

# Each marker consumes one full institutional phrase, including a short run of
# capitalized name words in front ("I.T.S Engineering College", "ABES Institute").
_NAME_WORD = r"[A-Z][\w.&'()-]*"
_FOOTER_MARKER_RES = (
    re.compile(rf"\bDepartment\s+of\s+{_NAME_WORD}(?:\s+(?:and|of|&)\s+{_NAME_WORD})*"),
    re.compile(rf"(?:\b{_NAME_WORD}\s+)*\bCollege\b(?:\s+of\s+{_NAME_WORD})?"),
    re.compile(rf"(?:\b{_NAME_WORD}\s+)*\bUniversity\b(?:\s+of\s+{_NAME_WORD})?"),
    re.compile(rf"\bInstitute\s+of\s+{_NAME_WORD}(?:\s+(?:and|of|&)\s+{_NAME_WORD})*"),
    re.compile(rf"(?:\b{_NAME_WORD}\s+)*\bInstitute\b"),
    re.compile(rf"\bSchool\s+of\s+{_NAME_WORD}"),
    re.compile(r"\bGreater\s+Noida\b"),
)


def _is_institutional_footer(tail: str) -> bool:
    """True if ``tail`` is dominated by institutional-footer markers."""
    residual = tail
    matched = False
    for marker in _FOOTER_MARKER_RES:
        residual, count = marker.subn(" ", residual)
        matched = matched or bool(count)
    if not matched:
        return False
    # Ignore initials/punctuation fragments; count real leftover words.
    leftover = re.findall(r"[A-Za-z]{2,}", residual)
    return len(leftover) <= 2


def strip_trailing_boilerplate(text: str) -> tuple[str, str | None]:
    """Remove a trailing institutional-footer segment from an outcome text.

    Returns ``(kept_text, stripped_tail)``; ``stripped_tail`` is ``None`` when
    nothing was removed. Scanning sentence boundaries left to right and taking
    the first footer-dominated tail keeps abbreviation-internal periods
    ("I.T.S") from splitting the footer itself.
    """
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        kept = text[: boundary.end()].strip()
        tail = text[boundary.end() :].strip()
        if not tail:
            break
        # Keep only if what remains is still a plausible outcome. Count only
        # real words (2+ letters) so acronym initials ("I.T.") do not let a
        # boundary inside the footer itself pass the guard.
        if len(re.findall(r"[A-Za-z]{2,}", kept)) < 4:
            continue
        if _is_institutional_footer(tail):
            return kept, tail
    return text, None


def _normalize_course_code(code: str) -> str:
    """Collapse stray spaces inside course codes ("KMBN HR05" -> "KMBNHR05")."""
    return re.sub(r"\s+", "", code).upper()


def _iter_logical_lines(content: str) -> Iterator[str]:
    """Yield one text line per CSV record, undoing CSV quoting.

    The real exports are single-column CSVs where lines containing commas were
    quoted by the spreadsheet tool. ``csv.reader`` re-assembles quoted records
    (including embedded newlines); joining the fields with "," restores the
    original text for unquoted lines that happened to contain commas.
    """
    for row in csv.reader(StringIO(content)):
        yield ",".join(row)


def parse_real_co_file(
    path: str | Path,
    warnings: list[str] | None = None,
) -> list[CoRecord]:
    """Parse a messy CO export into structured records.

    Course header rows open a new course; unmatched lines directly after a
    header are treated as wrapped course-name continuations; duplicate CO ids
    within one course (repeated blocks in the export) keep the first occurrence.

    Trailing institutional-footer boilerplate glued onto a CO description is
    stripped (see :func:`strip_trailing_boilerplate`). Pass a list as
    ``warnings`` to collect a note for every strip ("<course>-<CO>: ...") so
    nothing is altered silently.
    """
    content = read_text_file(Path(path))
    records: list[CoRecord] = []
    course_code = ""
    course_name_parts: list[str] = []
    seen_ids: set[str] = set()
    awaiting_name = False

    for raw_line in _iter_logical_lines(content):
        line = clean_text(raw_line)
        if not line:
            continue

        header = _COURSE_HEADER_RE.match(line)
        if header:
            course_code = _normalize_course_code(header.group(1))
            course_name_parts = [header.group(2).strip()] if header.group(2).strip() else []
            seen_ids = set()
            awaiting_name = True
            continue

        co_match = _CO_LINE_RE.match(line)
        if co_match:
            awaiting_name = False
            co_id = f"CO{int(co_match.group(1))}"
            description = co_match.group(2).strip()
            if not description or co_id in seen_ids:
                continue
            description, stripped_tail = strip_trailing_boilerplate(description)
            if stripped_tail is not None and warnings is not None:
                label = f"{course_code}-{co_id}" if course_code else co_id
                warnings.append(
                    f"{label}: stripped trailing boilerplate: '{stripped_tail}'"
                )
            seen_ids.add(co_id)
            records.append(
                CoRecord(
                    course_code=course_code,
                    course_name=" ".join(course_name_parts).strip(),
                    co_id=co_id,
                    description=description,
                )
            )
            continue

        if awaiting_name and course_code:
            # Wrapped course name, e.g. "KMBHR02PERFORMANCE AND" / "REWARD MANAGEMENT".
            course_name_parts.append(line)
        # Anything else (e.g. the "Course Name Course Outcomes" banner) is ignored.

    return records


def parse_real_po_file(path: str | Path) -> list[PoRecord]:
    """Parse a messy PO export into records classified as PEO / PO / PSO."""
    content = read_text_file(Path(path))
    parsed: list[tuple[str, str, list[str]]] = []  # (po_id, kind, description parts)

    for raw_line in _iter_logical_lines(content):
        line = clean_text(raw_line)
        if not line:
            continue
        match = _PO_LINE_RE.match(line)
        if match:
            kind = match.group(1)
            po_id = f"{kind}{int(match.group(2))}"
            parsed.append((po_id, kind, [match.group(3).strip()]))
        elif parsed:
            # Continuation of the previous statement (wrapped line).
            parsed[-1][2].append(line)

    return [
        PoRecord(po_id=po_id, description=" ".join(parts).strip(), kind=kind)
        for po_id, kind, parts in parsed
        if " ".join(parts).strip()
    ]


def list_courses(parsed: Sequence[CoRecord]) -> list[tuple[str, str]]:
    """Return unique (course_code, course_name) pairs in file order."""
    courses: list[tuple[str, str]] = []
    seen: set[str] = set()
    for record in parsed:
        if record.course_code and record.course_code not in seen:
            seen.add(record.course_code)
            courses.append((record.course_code, record.course_name))
    return courses


def to_canonical_co_rows(
    parsed: Sequence[CoRecord],
    course: str | None = None,
) -> list[dict[str, str]]:
    """Convert parsed CO records to canonical ``CO,description`` rows.

    With ``course`` given, only that course's COs are kept and ids stay plain
    ("CO1"). Without it, ids are prefixed with the course code
    ("KMBN101-CO1") so they stay unique across the whole program.
    """
    if course is not None:
        target = _normalize_course_code(course)
        selected = [r for r in parsed if r.course_code == target]
        if not selected:
            available = ", ".join(code for code, _ in list_courses(parsed)) or "none"
            raise ValueError(f"No COs found for course '{course}'. Available: {available}.")
        return [{"CO": r.co_id, "description": r.description} for r in selected]
    return [
        {
            "CO": f"{r.course_code}-{r.co_id}" if r.course_code else r.co_id,
            "description": r.description,
        }
        for r in parsed
    ]


def to_canonical_po_rows(
    parsed: Sequence[PoRecord],
    include: Sequence[str] = ("PO", "PSO"),
) -> list[dict[str, str]]:
    """Convert parsed PO records to canonical ``PO,description`` rows.

    PEO statements are excluded by default; pass ``include=("PEO", "PO", "PSO")``
    to keep them.
    """
    kinds = {kind.strip().upper() for kind in include}
    return [{"PO": r.po_id, "description": r.description} for r in parsed if r.kind in kinds]


def _write_rows(rows: Iterable[dict[str, str]], path: Path, fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_canonical_co_csv(rows: Iterable[dict[str, str]], path: str | Path) -> Path:
    """Write canonical CO rows to a ``CO,description`` CSV usable by the pipeline."""
    return _write_rows(rows, Path(path), fieldnames=("CO", "description"))


def write_canonical_po_csv(rows: Iterable[dict[str, str]], path: str | Path) -> Path:
    """Write canonical PO rows to a ``PO,description`` CSV usable by the pipeline."""
    return _write_rows(rows, Path(path), fieldnames=("PO", "description"))
