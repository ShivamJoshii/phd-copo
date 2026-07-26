from __future__ import annotations

import csv
from pathlib import Path

import pytest

from copo_mapper.ingest import (
    clean_text,
    list_courses,
    parse_real_co_file,
    parse_real_po_file,
    strip_trailing_boilerplate,
    to_canonical_co_rows,
    to_canonical_po_rows,
    write_canonical_co_csv,
    write_canonical_po_csv,
)
from copo_mapper.pipeline import run_pairwise_mapping

FIXTURES = Path(__file__).parent / "fixtures"
CO_FIXTURE = FIXTURES / "co_sample.csv"
PO_FIXTURE = FIXTURES / "po_sample.csv"

# The raw institutional exports live next to the repo checkout (phd-work/).
REAL_CO = Path(__file__).resolve().parents[2] / "CO (1).csv"
REAL_PO = Path(__file__).resolve().parents[2] / "PO (1).csv"


class TestParseRealCoFile:
    def test_detects_courses_and_normalizes_ids(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        codes = [code for code, _ in list_courses(records)]
        assert codes == ["KMBN101", "KMB204", "KMBHR02", "KMBFM01", "KMBNHR05"]
        first = records[0]
        assert first.course_code == "KMBN101"
        assert first.course_name == "MANAGEMENT CONCEPTS AND ORGANISATIONAL BEHAVIOUR"
        # "CO 1:" is normalized to "CO1".
        assert first.co_id == "CO1"
        assert first.description.startswith("Developing understanding")

    def test_merges_wrapped_course_name(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        hr02 = [r for r in records if r.course_code == "KMBHR02"]
        assert len(hr02) == 1
        assert hr02[0].course_name == "PERFORMANCE AND REWARD MANAGEMENT"

    def test_dedupes_repeated_blocks_and_fixes_mojibake(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        kmb204 = [r for r in records if r.course_code == "KMB204"]
        # The fixture repeats the CO3/CO4 block; only one of each survives.
        assert [r.co_id for r in kmb204] == ["CO3", "CO4"]
        # Quoted line: surrounding CSV quotes removed, internal comma kept.
        assert "capital, structure and leverage." in kmb204[0].description
        # cp1252 curly apostrophe normalized to ASCII.
        assert "firm's optimum dividend pay-out." in kmb204[1].description

    def test_handles_c0_typo_missing_separator_and_spaced_course_code(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        fm01 = {r.co_id: r for r in records if r.course_code == "KMBFM01"}
        # "CO1Understand ..." (no separator) still parses.
        assert fm01["CO1"].description == "Understand about various investment avenues."
        # "C0 3:" typo (zero for O) is recognized as CO3.
        assert "CO3" in fm01
        # "KMBN HR05" course code loses its stray space.
        hr05 = [r for r in records if r.course_code == "KMBNHR05"]
        assert [r.co_id for r in hr05] == ["CO1", "CO2"]

    def test_skips_banner_and_blank_lines(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        assert all("Course Name" not in r.description for r in records)
        assert all(r.description for r in records)

    def test_replacement_char_becomes_apostrophe_inside_words(self, tmp_path: Path) -> None:
        raw = (
            "1 KMBN101MANAGEMENT CONCEPTS\n"
            "CO1: Understand the firm�s goals � broadly.\n"
        )
        path = tmp_path / "co.csv"
        path.write_text(raw, encoding="utf-8")
        records = parse_real_co_file(path)
        assert records[0].description == "Understand the firm's goals broadly."


class TestStripTrailingBoilerplate:
    REAL_CONTAMINATED = (
        "Developing effective verbal and non verbal communication skills. "
        "I.T.S Engineering College Greater Noida Department of MBA"
    )

    def test_strips_real_institutional_footer(self) -> None:
        kept, tail = strip_trailing_boilerplate(self.REAL_CONTAMINATED)
        assert kept == "Developing effective verbal and non verbal communication skills."
        assert tail == "I.T.S Engineering College Greater Noida Department of MBA"

    def test_parse_strips_footer_and_reports_warning(self, tmp_path: Path) -> None:
        raw = "7 KMBN107BUSINESS COMMUNICATION\nCO5. " + self.REAL_CONTAMINATED + "\n"
        path = tmp_path / "co.csv"
        path.write_text(raw, encoding="utf-8")
        warnings: list[str] = []
        records = parse_real_co_file(path, warnings=warnings)
        assert records[0].description == (
            "Developing effective verbal and non verbal communication skills."
        )
        assert warnings == [
            "KMBN107-CO5: stripped trailing boilerplate: "
            "'I.T.S Engineering College Greater Noida Department of MBA'"
        ]

    def test_parse_without_warnings_list_still_strips(self, tmp_path: Path) -> None:
        raw = "7 KMBN107BUSINESS COMMUNICATION\nCO5. " + self.REAL_CONTAMINATED + "\n"
        path = tmp_path / "co.csv"
        path.write_text(raw, encoding="utf-8")
        records = parse_real_co_file(path)
        assert records[0].description.endswith("communication skills.")

    def test_mid_sentence_institution_words_not_stripped(self) -> None:
        legit = "Understand the role of university-industry collaboration in innovation"
        assert strip_trailing_boilerplate(legit) == (legit, None)

    def test_legitimate_trailing_sentence_mentioning_college_not_stripped(self) -> None:
        # A real sentence after the boundary is not dominated by footer markers.
        legit = (
            "Analyze admission trends in higher education. "
            "Compare policies adopted by every college and university in the region."
        )
        assert strip_trailing_boilerplate(legit) == (legit, None)

    def test_total_consumption_protected(self) -> None:
        # A record that is nothing but a footer must be left untouched.
        footer_only = "I.T.S Engineering College Greater Noida Department of MBA"
        assert strip_trailing_boilerplate(footer_only) == (footer_only, None)

    def test_short_remainder_protected(self) -> None:
        # Kept text under four words is not a plausible outcome; do not strip.
        short = "Communicate well. I.T.S Engineering College Greater Noida"
        assert strip_trailing_boilerplate(short) == (short, None)

    def test_glued_footer_without_space_stripped(self) -> None:
        glued = (
            "Developing effective verbal and non verbal communication "
            "skills.Department of MBA Greater Noida"
        )
        kept, tail = strip_trailing_boilerplate(glued)
        assert kept == (
            "Developing effective verbal and non verbal communication skills."
        )
        assert tail == "Department of MBA Greater Noida"


class TestParseRealPoFile:
    def test_classifies_kinds_and_strips_artifacts(self) -> None:
        records = parse_real_po_file(PO_FIXTURE)
        kinds = {r.po_id: r.kind for r in records}
        assert kinds == {
            "PEO1": "PEO",
            "PEO2": "PEO",
            "PO1": "PO",
            "PO3": "PO",
            "PO4": "PO",
            "PO5": "PO",
            "PSO1": "PSO",
        }
        by_id = {r.po_id: r for r in records}
        # Trailing non-breaking space (cp1252 0xA0) is stripped.
        assert by_id["PO3"].description.endswith("Leadership ability.")
        # Quoted lines keep internal commas, lose the surrounding quotes.
        assert by_id["PO4"].description.startswith("Ability to understand, analyze")
        assert not by_id["PO5"].description.endswith('"')

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "po.csv"
        path.write_text("\nPO1: First outcome.\n\n\nPSO1: Special outcome.\n", encoding="utf-8")
        records = parse_real_po_file(path)
        assert [r.po_id for r in records] == ["PO1", "PSO1"]


class TestCanonicalConversion:
    def test_co_rows_prefixed_for_program_wide_uniqueness(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        rows = to_canonical_co_rows(records)
        ids = [row["CO"] for row in rows]
        assert "KMBN101-CO1" in ids
        assert "KMBNHR05-CO1" in ids
        assert len(ids) == len(set(ids))
        assert set(rows[0]) == {"CO", "description"}

    def test_co_rows_filtered_to_single_course(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        rows = to_canonical_co_rows(records, course="KMBN101")
        assert [row["CO"] for row in rows] == ["CO1", "CO2"]
        # Filter is tolerant of the raw spaced/lowercase form.
        spaced = to_canonical_co_rows(records, course="kmbn hr05")
        assert [row["CO"] for row in spaced] == ["CO1", "CO2"]

    def test_co_rows_unknown_course_raises(self) -> None:
        records = parse_real_co_file(CO_FIXTURE)
        with pytest.raises(ValueError, match="No COs found"):
            to_canonical_co_rows(records, course="KMB999")

    def test_po_rows_exclude_peo_by_default(self) -> None:
        records = parse_real_po_file(PO_FIXTURE)
        rows = to_canonical_po_rows(records)
        assert [row["PO"] for row in rows] == ["PO1", "PO3", "PO4", "PO5", "PSO1"]
        assert set(rows[0]) == {"PO", "description"}

    def test_po_rows_include_override(self) -> None:
        records = parse_real_po_file(PO_FIXTURE)
        rows = to_canonical_po_rows(records, include=("PEO",))
        assert [row["PO"] for row in rows] == ["PEO1", "PEO2"]

    def test_clean_text_normalizes_punctuation_and_quotes(self) -> None:
        assert clean_text('"today’s  “world” "') == "today's \"world\""


class TestWritersAndPipeline:
    def test_written_csvs_feed_the_pipeline(self, tmp_path: Path) -> None:
        co_records = parse_real_co_file(CO_FIXTURE)
        po_records = parse_real_po_file(PO_FIXTURE)
        co_path = write_canonical_co_csv(
            to_canonical_co_rows(co_records, course="KMBN101"), tmp_path / "co.csv"
        )
        po_path = write_canonical_po_csv(to_canonical_po_rows(po_records), tmp_path / "po.csv")

        with co_path.open(newline="", encoding="utf-8") as f:
            assert csv.DictReader(f).fieldnames == ["CO", "description"]

        pair_path, matrix_path = run_pairwise_mapping(
            str(co_path), str(po_path), str(tmp_path / "out")
        )
        assert pair_path.exists()
        assert matrix_path.exists()
        with pair_path.open(newline="", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))
        assert len(pairs) == 2 * 5  # 2 COs x 5 POs/PSOs


@pytest.mark.skipif(
    not (REAL_CO.exists() and REAL_PO.exists()),
    reason="real institutional exports not present",
)
class TestRealFiles:
    def test_real_footer_contamination_stripped_and_reported(self) -> None:
        warnings: list[str] = []
        co_records = parse_real_co_file(REAL_CO, warnings=warnings)
        target = [
            r for r in co_records if r.course_code == "KMBN107" and r.co_id == "CO5"
        ]
        assert len(target) == 1
        assert target[0].description == (
            "Developing effective verbal and non verbal communication skills."
        )
        # Exactly one row in the whole export is footer-contaminated.
        assert warnings == [
            "KMBN107-CO5: stripped trailing boilerplate: "
            "'I.T.S Engineering College Greater Noida Department of MBA'"
        ]

    def test_parses_real_exports_end_to_end(self) -> None:
        co_records = parse_real_co_file(REAL_CO)
        po_records = parse_real_po_file(REAL_PO)
        assert len(list_courses(co_records)) >= 40
        assert len(co_records) >= 200
        co_rows = to_canonical_co_rows(co_records)
        assert len({row["CO"] for row in co_rows}) == len(co_rows)
        assert [r.po_id for r in po_records if r.kind == "PO"] == [
            f"PO{n}" for n in range(1, 8)
        ]
        assert all(ord(ch) < 128 for row in co_rows for ch in row["description"])
        assert all(ord(ch) < 128 for r in po_records for ch in r.description)
