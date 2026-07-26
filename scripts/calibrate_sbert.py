"""
Sanity-check SBERT similarity rescaling + shared thresholds in
copo_mapper.scoring.

Encodes hand-written CO/PO pairs of known relatedness (strong / moderate /
weak / unrelated, education-flavored), prints the raw MiniLM cosine
similarity distribution per band together with the rescaled value
(SIMILARITY_RESCALE anchors), then runs each pair through score_pair with
backend="sbert" and reports whether the predicted label matches the
expected band.

Use the per-band raw-similarity distribution printed at the end to check
the rescale anchors: the "unrelated" band should sit at or below the lo
anchor (rescaled ~0) and the "strong" band should rescale to >= ~0.6 so it
can reach label 3 through the shared (0.50, 0.30, 0.10) cutoffs.

Requires sentence-transformers:
    pip install sentence-transformers

Run from the repo root:
    python3 scripts/calibrate_sbert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copo_mapper.preprocess import normalize_text
from copo_mapper.scoring import (
    SIMILARITY_RESCALE,
    THRESHOLDS,
    rescale_similarity,
    score_pair,
)
from copo_mapper.semantic import sbert_pair_similarity

# (expected_label, band, co_text, po_text)
PAIRS: list[tuple[int, str, str, str]] = [
    (
        3,
        "strong",
        "Design and develop relational database schemas for enterprise applications.",
        "Design solutions for complex engineering problems using database systems.",
    ),
    (
        3,
        "strong",
        "Apply sorting and searching algorithms to solve computational problems.",
        "Apply knowledge of algorithms and data structures to engineering problems.",
    ),
    (
        3,
        "strong",
        "Analyze the time complexity of algorithms and justify optimization choices.",
        "Analyze complex problems and evaluate algorithmic solutions critically.",
    ),
    (
        2,
        "moderate",
        "Implement software modules following standard engineering practices.",
        "Use modern engineering tools and techniques for professional practice.",
    ),
    (
        2,
        "moderate",
        "Explain the principles of database transactions and concurrency control.",
        "Demonstrate understanding of core computing concepts and systems.",
    ),
    (
        1,
        "weak",
        "Describe the layers of the OSI network reference model.",
        "Communicate effectively through written technical reports.",
    ),
    (
        1,
        "weak",
        "Evaluate machine learning models using appropriate metrics.",
        "Function effectively as a member of a multidisciplinary team.",
    ),
    (
        0,
        "unrelated",
        "List the postulates of quantum mechanics.",
        "Demonstrate ethical responsibility in professional engineering practice.",
    ),
    (
        0,
        "unrelated",
        "Recall the historical timeline of the French Revolution.",
        "Apply calculus techniques to solve engineering mathematics problems.",
    ),
    (
        0,
        "unrelated",
        "Identify common species of flowering plants in tropical climates.",
        "Write structured programs using loops and conditional statements.",
    ),
]


def main() -> int:
    co_norms = [normalize_text(co) for _, _, co, _ in PAIRS]
    po_norms = [normalize_text(po) for _, _, _, po in PAIRS]

    sims = sbert_pair_similarity(co_norms, po_norms)
    if sims is None:
        print("sentence-transformers is not installed; cannot run calibration.")
        print("Install it with: pip install sentence-transformers")
        return 1

    t3, t2, t1 = THRESHOLDS["sbert"]
    lo, hi = SIMILARITY_RESCALE["sbert"]
    print(f"shared composite thresholds: t3={t3} t2={t2} t1={t1}")
    print(f"sbert rescale anchors: lo={lo} (unrelated floor) hi={hi} (paraphrase ceiling)")
    print()

    by_band: dict[str, list[float]] = {}
    mismatches = 0
    for (expected, band, _co, _po), co_norm, po_norm, sim in zip(
        PAIRS, co_norms, po_norms, sims, strict=True
    ):
        result = score_pair(co_norm, po_norm, sim, backend="sbert")
        by_band.setdefault(band, []).append(sim)
        marker = "ok " if result.score == expected else "MISS"
        if result.score != expected:
            mismatches += 1
        print(
            f"[{marker}] band={band:9s} expected={expected} predicted={result.score} "
            f"raw_sim={sim:.3f} rescaled={rescale_similarity(sim, 'sbert'):.3f} "
            f"composite={result.confidence:.3f}"
        )
        print(f"       {result.explanation}")

    print()
    print("Raw similarity distribution by band (rescaled in parentheses):")
    for band in ("strong", "moderate", "weak", "unrelated"):
        values = by_band.get(band, [])
        if values:
            mean = sum(values) / len(values)
            print(
                f"  {band:9s} n={len(values)} "
                f"min={min(values):.3f} ({rescale_similarity(min(values), 'sbert'):.3f}) "
                f"max={max(values):.3f} ({rescale_similarity(max(values), 'sbert'):.3f}) "
                f"mean={mean:.3f} ({rescale_similarity(mean, 'sbert'):.3f})"
            )

    print()
    print(f"{len(PAIRS) - mismatches}/{len(PAIRS)} pairs matched the expected label.")
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
