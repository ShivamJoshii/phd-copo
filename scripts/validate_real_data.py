#!/usr/bin/env python3
"""Empirical validation of scoring calibration against the REAL CO/PO exports.

Runs the actual tfidf pipeline on real courses (management, finance, IT) from
"CO (1).csv" / "PO (1).csv" (expected one directory above the repo root, or
pass paths as argv[1]/argv[2]) and checks the acceptance criteria from the
adversarial review of C1 (domain-overlap label inflation) and C2 (Bloom
noun-homograph inflation):

  1. Real course grids must span the label range (0s present, 3s reachable
     program-wide where warranted) and a majority of pairs must NOT be >= 2.
  2. The review's synthetic experiment (real KMBN101 5x5 management grid with
     forced semantic similarity 0.0 / 0.15) must no longer show 40%+ label-2
     inflation.
  3. The review's failing Bloom examples must classify correctly.
  4. Compiler-course domain checks must pass, including multi-word phrase
     matching ("code generation") through the PRODUCTION score_pair path.

Baselines ("BEFORE") were measured at the pre-fix HEAD and are hardcoded for
comparison. Exit status is non-zero if any acceptance check fails.
"""

from __future__ import annotations

import collections
import csv
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from copo_mapper.features import detect_bloom, detect_domains, token_set  # noqa: E402
from copo_mapper.ingest import (  # noqa: E402
    parse_real_co_file,
    parse_real_po_file,
    to_canonical_co_rows,
    to_canonical_po_rows,
    write_canonical_co_csv,
    write_canonical_po_csv,
)
from copo_mapper.pipeline import run_pairwise_mapping  # noqa: E402
from copo_mapper.scoring import score_pair  # noqa: E402

COURSES = ("KMBN101", "KMBFM01", "KMBIT04")

# Measured at the pre-fix HEAD (tfidf pipeline, same inputs).
BASELINE_DISTS = {
    "KMBN101": {0: 7, 1: 28, 2: 10, 3: 0},
    "KMBFM01": {0: 8, 1: 37, 2: 0, 3: 0},
    "KMBIT04": {0: 20, 1: 23, 2: 2, 3: 0},
}
# Synthetic experiment: KMBN101 COs x PO1-PO5 with forced similarity.
# BEFORE values reconstructed with the exact pre-fix algorithm (highest-wins
# bloom over the full verb lists, tokens-only domains with the catch-all
# management vocabulary) on this grid. The review's own 5x5 grid (different
# PO selection) showed the same failure shape even more starkly:
# sim=0.0 -> 10x"2"/11x"1"/4x"0", sim=0.15 -> ZERO label-0 pairs.
BASELINE_SYNTHETIC = {0.0: {0: 2, 1: 18, 2: 5}, 0.15: {0: 0, 1: 15, 2: 10}}

BLOOM_CHECKS = [
    ("Understand various Models of Investment and its application", "understand"),
    ("Understand the value of assets and manage investment portfolio.", "understand"),
    ("Foster Analytical and critical thinking abilities for data-based decision making.", "analyze"),
    ("Develop lexical analyzer for a given grammar", "create"),
    ("Design top-down and bottom-up parsers", "create"),
    ("Measure riskiness of a stock or a portfolio position.", "evaluate"),
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def fmt(dist: dict[int, int]) -> str:
    return " ".join(f"{label}:{dist.get(label, 0)}" for label in (0, 1, 2, 3))


def main() -> int:
    co_file = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "CO (1).csv"
    po_file = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT.parent / "PO (1).csv"
    if not co_file.exists() or not po_file.exists():
        print(f"Real data not found at {co_file} / {po_file}")
        return 2

    cos = parse_real_co_file(co_file)
    pos = parse_real_po_file(po_file)
    po_rows = to_canonical_po_rows(pos)

    tmp = Path(tempfile.mkdtemp(prefix="copo_validate_"))
    po_path = write_canonical_po_csv(po_rows, tmp / "po.csv")

    # ------------------------------------------------------------------
    # 1. Real course grids through the actual tfidf pipeline
    # ------------------------------------------------------------------
    for course in COURSES:
        co_path = write_canonical_co_csv(
            to_canonical_co_rows(cos, course=course), tmp / f"co_{course}.csv"
        )
        pair_path, matrix_path = run_pairwise_mapping(
            str(co_path), str(po_path), str(tmp / f"out_{course}")
        )
        with pair_path.open() as f:
            rows = list(csv.DictReader(f))
        dist = collections.Counter(int(r["predicted_strength"]) for r in rows)
        n = len(rows)
        print(f"\n=== {course} ({n} pairs) ===")
        print(f"BEFORE (pre-fix HEAD): {fmt(BASELINE_DISTS[course])}")
        print(f"AFTER  (this run):     {fmt(dist)}")
        print(matrix_path.read_text())
        at_least_2 = sum(count for label, count in dist.items() if label >= 2)
        check(at_least_2 < n / 2, f"{course}: majority of pairs below label 2 ({at_least_2}/{n} are >=2)")
        check(at_least_2 <= 0.2 * n, f"{course}: label >=2 not inflated ({at_least_2}/{n} <= 20%)")
        check(dist.get(0, 0) >= 3, f"{course}: grid contains 0s ({dist.get(0, 0)} zeros)")

    # Whole-program grid: the range must be spanned where warranted.
    co_all_path = write_canonical_co_csv(to_canonical_co_rows(cos), tmp / "co_all.csv")
    pair_path, _ = run_pairwise_mapping(str(co_all_path), str(po_path), str(tmp / "out_all"))
    with pair_path.open() as f:
        rows = list(csv.DictReader(f))
    dist = collections.Counter(int(r["predicted_strength"]) for r in rows)
    print(f"\n=== whole program ({len(rows)} pairs) ===")
    print(f"AFTER: {fmt(dist)}")
    threes = [r for r in rows if r["predicted_strength"] == "3"]
    for r in threes:
        print(f"  3: {r['co_id']} x {r['po_id']}: {r['co_text'][:60]} || {r['po_text'][:60]}")
    check(dist.get(0, 0) > 0 and dist.get(3, 0) > 0, "program-wide labels span 0..3")
    check(
        all("leadership" in (r["co_text"] + r["po_text"]).lower()
            or "business plan" in r["co_text"].lower()
            or "analyse" in r["co_text"].lower() or "analyze" in r["co_text"].lower()
            for r in threes),
        "every label-3 pair is a semantically warranted match",
    )

    # ------------------------------------------------------------------
    # 2. Review's synthetic experiment: KMBN101 COs x PO1-PO5, forced sim
    # ------------------------------------------------------------------
    co5 = [r.description for r in cos if r.course_code == "KMBN101"]
    po5 = [p.description for p in pos if p.kind == "PO"][:5]
    print("\n=== synthetic management 5x5 (forced similarity) ===")
    for sim in (0.0, 0.15):
        dist = collections.Counter(
            score_pair(co, po, sim).score for co in co5 for po in po5
        )
        print(f"sim={sim}: BEFORE {fmt(BASELINE_SYNTHETIC[sim])}  AFTER {fmt(dist)}")
        check(
            dist.get(2, 0) + dist.get(3, 0) <= 2,
            f"synthetic sim={sim}: at most 2 pairs reach label >=2 "
            f"(was {BASELINE_SYNTHETIC[sim].get(2, 0)} at HEAD)",
        )
    dist0 = collections.Counter(score_pair(co, po, 0.0).score for co in co5 for po in po5)
    check(dist0.get(0, 0) >= 3, f"synthetic sim=0.0: zeros present ({dist0.get(0, 0)})")

    # ------------------------------------------------------------------
    # 3. Bloom classification checks (review's failing examples)
    # ------------------------------------------------------------------
    print("\n=== bloom checks ===")
    for text, expected in BLOOM_CHECKS:
        got = detect_bloom(token_set(text), text=text)
        check(got == expected, f"bloom '{text[:55]}...' -> {got} (want {expected})")

    # ------------------------------------------------------------------
    # 4. Compiler-course domain checks (incl. M2 phrase matching)
    # ------------------------------------------------------------------
    print("\n=== domain checks ===")
    for text in (
        "Develop lexical analyzer for a given grammar",
        "Design top-down and bottom-up parsers",
        "Develop syntax directed translation schemes",
    ):
        domains = detect_domains(token_set(text), text=text)
        check("compilers" in domains, f"domains '{text[:45]}' include compilers ({sorted(domains)})")
    # M2: "code generation" only bridges these two texts through PHRASE
    # matching ("code" alone maps to software, not compilers; "compiler"
    # maps to compilers). Without the phrase, domain jaccard would be 0.
    result = score_pair("code generation", "compiler construction", 0.0)
    check(
        "domain_overlap=0.50" in result.explanation,
        f"phrase 'code generation' matches through production score_pair ({result.explanation})",
    )

    print()
    if failures:
        print(f"{len(failures)} ACCEPTANCE CHECK(S) FAILED")
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
