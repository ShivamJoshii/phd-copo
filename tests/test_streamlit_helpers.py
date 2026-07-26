"""Tests for the streamlit-facing raw-export helpers (no streamlit required)."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from copo_mapper.ui_helpers import (
    ALL_COURSES_OPTION,
    DEFAULT_PO_KINDS,
    available_po_kinds,
    canonical_co_csv_text,
    canonical_po_csv_text,
    canonical_rows_from_raw_co_bytes,
    canonical_rows_from_raw_po_bytes,
    course_from_option,
    course_options,
    parse_raw_co_bytes,
    parse_raw_po_bytes,
)

# The raw institutional exports live next to the repo checkout (phd-work/).
REAL_CO = Path(__file__).resolve().parents[2] / "CO (1).csv"
REAL_PO = Path(__file__).resolve().parents[2] / "PO (1).csv"

CO_SAMPLE = (
    "Course Name Course Outcomes\n"
    "1 KMBN101MANAGEMENT CONCEPTS AND ORGANISATIONAL BEHAVIOUR\n"
    "CO 1: Developing understanding of management concepts.\n"
    "CO2: Apply behavioural theories.\n"
    "2 KMBHR02PERFORMANCE AND\n"
    "REWARD MANAGEMENT\n"
    "CO1: Design reward systems.\n"
).encode("utf-8")

PO_SAMPLE = (
    "PEO1: Prepare graduates for managerial careers.\n"
    "PO1: Apply knowledge of management theories.\n"
    "PO2: Foster analytical and critical thinking\n"
    "for data-based decision making.\n"
    "PSO1: Demonstrate domain-specific skills.\n"
).encode("utf-8")


class TestRawCoBytes:
    def test_parse_and_course_options(self) -> None:
        records = parse_raw_co_bytes(CO_SAMPLE)
        options = course_options(records)
        assert options[0] == ALL_COURSES_OPTION
        assert options[1].startswith("KMBN101")
        assert "PERFORMANCE AND REWARD MANAGEMENT" in options[2]

    def test_course_from_option_round_trip(self) -> None:
        records = parse_raw_co_bytes(CO_SAMPLE)
        options = course_options(records)
        assert course_from_option(options[0]) is None
        assert course_from_option(options[1]) == "KMBN101"
        assert course_from_option(options[2]) == "KMBHR02"

    def test_all_courses_rows_are_prefixed(self) -> None:
        rows = canonical_rows_from_raw_co_bytes(CO_SAMPLE, course=None)
        assert [row["CO"] for row in rows] == [
            "KMBN101-CO1",
            "KMBN101-CO2",
            "KMBHR02-CO1",
        ]
        assert set(rows[0]) == {"CO", "description"}

    def test_single_course_rows_keep_plain_ids(self) -> None:
        rows = canonical_rows_from_raw_co_bytes(CO_SAMPLE, course="KMBN101")
        assert [row["CO"] for row in rows] == ["CO1", "CO2"]
        assert rows[1]["description"] == "Apply behavioural theories."

    def test_unknown_course_raises_with_available_codes(self) -> None:
        with pytest.raises(ValueError, match="Available: KMBN101, KMBHR02"):
            canonical_rows_from_raw_co_bytes(CO_SAMPLE, course="KMB999")

    def test_no_co_statements_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="No CO statements detected"):
            parse_raw_co_bytes(b"just,a,normal,csv\nwith,no,outcomes,here\n")


class TestRawPoBytes:
    def test_parse_and_available_kinds(self) -> None:
        records = parse_raw_po_bytes(PO_SAMPLE)
        assert available_po_kinds(records) == ["PO", "PSO", "PEO"]

    def test_default_kinds_exclude_peo(self) -> None:
        rows = canonical_rows_from_raw_po_bytes(PO_SAMPLE)
        assert [row["PO"] for row in rows] == ["PO1", "PO2", "PSO1"]
        # Wrapped continuation line is merged into PO2.
        assert rows[1]["description"].endswith("data-based decision making.")

    def test_include_peo_only(self) -> None:
        rows = canonical_rows_from_raw_po_bytes(PO_SAMPLE, include=("PEO",))
        assert [row["PO"] for row in rows] == ["PEO1"]

    def test_default_kinds_constant(self) -> None:
        assert DEFAULT_PO_KINDS == ("PO", "PSO")

    def test_no_matching_kinds_raises_clear_error(self) -> None:
        peo_only = b"PEO1: Prepare graduates for managerial careers.\n"
        with pytest.raises(ValueError, match="The file contains: PEO"):
            canonical_rows_from_raw_po_bytes(peo_only, include=("PO", "PSO"))

    def test_no_po_statements_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="No PO/PSO/PEO statements detected"):
            parse_raw_po_bytes(b"nothing to see here\n")


class TestCanonicalCsvText:
    def test_co_csv_text_round_trips(self) -> None:
        rows = canonical_rows_from_raw_co_bytes(CO_SAMPLE, course="KMBN101")
        text = canonical_co_csv_text(rows)
        parsed = list(csv.DictReader(StringIO(text)))
        assert parsed == rows
        assert text.splitlines()[0] == "CO,description"

    def test_po_csv_text_round_trips(self) -> None:
        rows = canonical_rows_from_raw_po_bytes(PO_SAMPLE)
        text = canonical_po_csv_text(rows)
        parsed = list(csv.DictReader(StringIO(text)))
        assert parsed == rows
        assert text.splitlines()[0] == "PO,description"


@pytest.mark.skipif(
    not (REAL_CO.exists() and REAL_PO.exists()),
    reason="real institutional exports not present",
)
class TestRealFileBytes:
    def test_real_co_bytes_convert_cleanly(self) -> None:
        data = REAL_CO.read_bytes()
        records = parse_raw_co_bytes(data)
        options = course_options(records)
        assert options[0] == ALL_COURSES_OPTION
        assert len(options) >= 41  # all-courses option + >=40 courses
        rows = canonical_rows_from_raw_co_bytes(data)
        assert len({row["CO"] for row in rows}) == len(rows)
        # Single-course selection via a real option label round-trips.
        code = course_from_option(options[1])
        single = canonical_rows_from_raw_co_bytes(data, course=code)
        assert single and all("-" not in row["CO"] for row in single)

    def test_real_po_bytes_convert_cleanly(self) -> None:
        data = REAL_PO.read_bytes()
        records = parse_raw_po_bytes(data)
        assert available_po_kinds(records) == ["PO", "PSO", "PEO"]
        rows = canonical_rows_from_raw_po_bytes(data)
        assert all(not row["PO"].startswith("PEO") for row in rows)
        assert [r for r in rows if r["PO"] == "PO1"]
        text = canonical_po_csv_text(rows)
        assert list(csv.DictReader(StringIO(text))) == rows
