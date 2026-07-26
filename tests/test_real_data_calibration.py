"""Calibration acceptance tests against REAL institutional CO/PO data.

Converts the acceptance criteria of the C1/C2 adversarial-review fixes into
tests (see scripts/validate_real_data.py for the printable harness):

  - C1: domain overlap must no longer inflate labels across a real MBA
    management grid (pre-fix, 40%+ of the review's 5x5 grid reached label 2
    at semantic similarity 0, and NO pair scored 0 at similarity 0.15).
  - C2: Bloom noun-homographs ("value", "model", "measure", ...) must not
    override the leading action verb of an outcome statement.
  - M2: multi-word domain phrases ("code generation") must match through the
    production score_pair path.

Two layers:
  * Inline-fixture tests always run — they hardcode the real KMBN101 CO and
    program PO/PSO statements (verbatim from the exports).
  * Tests on the raw exports run only when "CO (1).csv" / "PO (1).csv" exist
    one directory above the repo root (skipped otherwise).
"""

from __future__ import annotations

import collections
import csv
import tempfile
import unittest
from pathlib import Path

from copo_mapper.features import detect_bloom, detect_domains, token_set
from copo_mapper.pipeline import run_pairwise_mapping
from copo_mapper.scoring import score_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CO_FILE = REPO_ROOT.parent / "CO (1).csv"
REAL_PO_FILE = REPO_ROOT.parent / "PO (1).csv"

# Real KMBN101 (Management Concepts and Organisational Behaviour) COs,
# verbatim from the institutional export.
KMBN101_COS = [
    "Developing understanding of managerial practices and their perspectives.",
    "Understanding and Applying the concepts of organizational behavior.",
    "Applying the concepts of management and analyze organizational behaviors "
    "in real world situations",
    "Comprehend and practice contemporary issues in management.",
    "Applying managerial and leadership skills among students",
]

# Real program PO/PSO statements, verbatim from the institutional export.
PROGRAM_POS = {
    "PO1": "Apply knowledge of management theories and practices to solve business problems.",
    "PO2": "Foster Analytical and critical thinking abilities for data-based decision making.",
    "PO3": "Ability to develop Value based Leadership ability.",
    "PO4": "Ability to understand, analyze and communicate global, economic, legal, "
    "and ethical aspects of business.",
    "PO5": "Ability to lead themselves and others in the achievement of organizational "
    "goals, contributing effectively to a team environment.",
    "PO6": "Ability to develop entrepreneurial thinking through business acumen.",
    "PO7": "Ability to adapt technological advancements through life-long learning.",
    "PSO1": "To strengthen communication skills effectively and implement team building "
    "skills in contemporary business environment.",
    "PSO2": "To build up Indian ethos and values among MBA graduates exhibiting value "
    "centered leadership in managerial decision making.",
}

# Review's failing Bloom examples (real CO/PO texts) with correct levels.
BLOOM_EXPECTATIONS = [
    ("Understand various Models of Investment and its application", "understand"),
    ("Understand the value of assets and manage investment portfolio.", "understand"),
    (
        "Foster Analytical and critical thinking abilities for data-based decision making.",
        "analyze",
    ),
    ("Develop lexical analyzer for a given grammar", "create"),
    ("Design top-down and bottom-up parsers", "create"),
    ("Measure riskiness of a stock or a portfolio position.", "evaluate"),
]


def _bloom(text: str) -> str:
    return detect_bloom(token_set(text), text=text)


def _pipeline_distribution(
    co_rows: list[tuple[str, str]], po_rows: list[tuple[str, str]]
) -> collections.Counter:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        co_file = tmp_path / "co.csv"
        po_file = tmp_path / "po.csv"
        with co_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CO", "description"])
            writer.writerows(co_rows)
        with po_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PO", "description"])
            writer.writerows(po_rows)
        pair_path, _ = run_pairwise_mapping(str(co_file), str(po_file), str(tmp_path / "out"))
        with pair_path.open() as f:
            return collections.Counter(
                int(row["predicted_strength"]) for row in csv.DictReader(f)
            )


class BloomCalibrationInlineTest(unittest.TestCase):
    def test_review_failing_examples_classify_correctly(self) -> None:
        for text, expected in BLOOM_EXPECTATIONS:
            self.assertEqual(_bloom(text), expected, text)


class DomainCalibrationInlineTest(unittest.TestCase):
    def test_compiler_course_domains(self) -> None:
        for text in (
            "Develop lexical analyzer for a given grammar",
            "Design top-down and bottom-up parsers",
            "Develop syntax directed translation schemes",
        ):
            self.assertIn("compilers", detect_domains(token_set(text), text=text), text)

    def test_code_generation_phrase_matches_in_production(self) -> None:
        # M2: "code" alone maps to software (not compilers) and "compiler"
        # maps to compilers, so a nonzero domain overlap between these two
        # texts is only possible if score_pair feeds the raw text into
        # detect_domains and the phrase "code generation" matches.
        result = score_pair("code generation", "compiler construction", 0.0)
        self.assertIn("domain_overlap=0.50", result.explanation)


class ManagementGridInlineTest(unittest.TestCase):
    """The review's synthetic experiment on the real KMBN101 5x5 grid."""

    def _grid(self, sim: float) -> collections.Counter:
        po5 = [PROGRAM_POS[f"PO{i}"] for i in range(1, 6)]
        return collections.Counter(
            score_pair(co, po, sim).score for co in KMBN101_COS for po in po5
        )

    def test_sim_zero_grid_is_not_inflated(self) -> None:
        # Pre-fix on this grid: 5x"2" / 18x"1" / 2x"0" (the review's own PO
        # selection showed 10x"2" / 11x"1" / 4x"0"). Unrelated pairs must
        # not reach label 2 on lexical features alone.
        dist = self._grid(0.0)
        self.assertLessEqual(dist[2] + dist[3], 2, dist)
        self.assertGreaterEqual(dist[0], 3, dist)

    def test_sim_015_grid_is_not_inflated(self) -> None:
        # Pre-fix on this grid: 10 of 25 pairs reached label 2 and none
        # scored 0. Only genuinely bridged pairs (shared specific domain,
        # e.g. leadership<->leadership) may reach 2 now.
        dist = self._grid(0.15)
        self.assertLessEqual(dist[2] + dist[3], 2, dist)

    def test_full_pipeline_distribution_spans_and_is_majority_below_2(self) -> None:
        co_rows = [(f"CO{i}", text) for i, text in enumerate(KMBN101_COS, 1)]
        po_rows = list(PROGRAM_POS.items())
        dist = _pipeline_distribution(co_rows, po_rows)
        total = sum(dist.values())
        self.assertEqual(total, 45)
        at_least_2 = dist[2] + dist[3]
        # Majority (in fact >=80%) of pairs below label 2; zeros present.
        self.assertLessEqual(at_least_2, 0.2 * total, dist)
        self.assertGreaterEqual(dist[0], 3, dist)
        self.assertGreater(dist[1], 0, dist)

    def test_label_3_reachable_for_genuinely_strong_real_pair(self) -> None:
        # Real pair from the export (KMBHR01-CO4 x PO3); 0.24 is the tfidf
        # cosine measured on the whole-program corpus. Strong verb, domain
        # and token agreement must still produce a 3.
        result = score_pair(
            "Competency to develop leadership qualities among subordinates.",
            PROGRAM_POS["PO3"],
            0.24,
        )
        self.assertEqual(result.score, 3, result)


@unittest.skipUnless(
    REAL_CO_FILE.exists() and REAL_PO_FILE.exists(),
    "real CO/PO exports not present",
)
class RealExportCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from copo_mapper.ingest import (
            parse_real_co_file,
            parse_real_po_file,
            to_canonical_co_rows,
            to_canonical_po_rows,
        )

        cls.cos = parse_real_co_file(REAL_CO_FILE)
        cls.pos = parse_real_po_file(REAL_PO_FILE)
        cls.po_rows = [
            (row["PO"], row["description"]) for row in to_canonical_po_rows(cls.pos)
        ]
        cls.to_canonical_co_rows = staticmethod(to_canonical_co_rows)

    def _course_distribution(self, course: str) -> collections.Counter:
        co_rows = [
            (row["CO"], row["description"])
            for row in self.to_canonical_co_rows(self.cos, course=course)
        ]
        return _pipeline_distribution(co_rows, self.po_rows)

    def test_course_grids_have_zeros_and_majority_below_2(self) -> None:
        # Pre-fix distributions (same pipeline, pre-fix scoring):
        #   KMBN101: 7x0 28x1 10x2; KMBFM01: 8x0 37x1; KMBIT04: 20x0 23x1 2x2.
        for course in ("KMBN101", "KMBFM01", "KMBIT04"):
            dist = self._course_distribution(course)
            total = sum(dist.values())
            self.assertEqual(total, 45, course)
            self.assertLessEqual(dist[2] + dist[3], 0.2 * total, (course, dist))
            self.assertGreaterEqual(dist[0], 3, (course, dist))

    def test_program_wide_labels_span_range(self) -> None:
        co_rows = [
            (row["CO"], row["description"])
            for row in self.to_canonical_co_rows(self.cos)
        ]
        dist = _pipeline_distribution(co_rows, self.po_rows)
        self.assertGreater(dist[0], 0, dist)
        self.assertGreater(dist[3], 0, dist)
        # Strong labels must be rare: the review measured 40%+ label-2
        # inflation on management grids pre-fix.
        total = sum(dist.values())
        self.assertLessEqual(dist[2] + dist[3], 0.05 * total, dist)


if __name__ == "__main__":
    unittest.main()
