"""Empirical calibration of the neural similarity-rescale anchors.

Pure logic shared by ``scripts/calibrate_backends.py`` (CLI) and the
Streamlit app's Calibration tab. Nothing here imports streamlit or the
neural libraries directly — a similarity function is injected, so the
module is testable with a fake backend and degrades cleanly when
sentence-transformers / torch are missing.

Three measurements per backend:

1. **Banded pairs** — hand-written CO/PO pairs of known relatedness
   (strong / moderate / weak / unrelated); each is scored through the
   production ``score_pair`` path and compared with its expected label.
2. **Near-paraphrase probes** — rewordings of the same outcome statement;
   their raw cosine measures the backend's practical ceiling (the ``hi``
   anchor) instead of assuming it from the literature.
3. **Real-grid sweep** (optional) — the full CO x PO grid from the real
   institutional exports; its raw-cosine percentiles give an honest
   "unrelated floor" because the tfidf-validated label distribution shows
   ~98% of real pairs are unrelated/weak (label 0/1).

Anchor suggestion:

    suggested lo = max(mean of the unrelated band, p50 of the real grid)
    suggested hi = mean raw cosine of the paraphrase probes

Drift beyond ``DRIFT_TOLERANCE`` against ``scoring.SIMILARITY_RESCALE`` is
flagged; nothing is modified automatically.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from .preprocess import normalize_text
from .scoring import (
    SIMILARITY_FLOOR_FOR_3,
    SIMILARITY_RESCALE,
    THRESHOLDS,
    rescale_similarity,
    score_pair,
)
from .semantic import bert_pair_similarity, sbert_pair_similarity

# A similarity function takes two equal-length lists of normalized texts and
# returns per-pair cosines, or None when its dependencies are unavailable.
SimilarityFn = Callable[[list[str], list[str]], Optional[list[float]]]

DRIFT_TOLERANCE = 0.05
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

# (expected_label, band, co_text, po_text) — education-flavored pairs of
# known relatedness. Mirrors scripts/calibrate_sbert.py so results stay
# comparable with the original sbert-only harness.
BANDED_PAIRS: list[tuple[int, str, str, str]] = [
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

# Rewordings of the SAME outcome statement: measure the paraphrase ceiling.
PARAPHRASE_PAIRS: list[tuple[str, str]] = [
    (
        "Design and implement relational database solutions for business needs.",
        "Develop and build relational database systems that serve business requirements.",
    ),
    (
        "Apply management theories and practices to solve business problems.",
        "Use theories and practice of management for solving problems in business.",
    ),
    (
        "Analyze financial statements to evaluate organizational performance.",
        "Examine financial reports in order to assess the performance of an organization.",
    ),
    (
        "Demonstrate effective communication skills in professional settings.",
        "Show the ability to communicate effectively in a professional environment.",
    ),
    (
        "Evaluate marketing strategies using appropriate analytical frameworks.",
        "Assess strategies for marketing through suitable frameworks of analysis.",
    ),
]


def default_similarity_fn(backend: str, model: str | None = None) -> SimilarityFn:
    """Production similarity function for a backend (deps checked at call time)."""
    if backend == "sbert":
        if model:
            return lambda a, b: sbert_pair_similarity(a, b, model_name=model)
        return sbert_pair_similarity
    if backend == "bert":
        if model:
            return lambda a, b: bert_pair_similarity(a, b, model_name=model)
        return bert_pair_similarity
    raise ValueError("backend must be 'sbert' or 'bert'")


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (no numpy dependency)."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


@dataclass(frozen=True)
class BandedRow:
    band: str
    expected: int
    predicted: int
    raw: float
    rescaled: float
    composite: float
    explanation: str


@dataclass(frozen=True)
class BandSummary:
    band: str
    n: int
    raw_min: float
    raw_max: float
    raw_mean: float


@dataclass(frozen=True)
class AnchorSuggestion:
    backend: str
    lo_current: float
    hi_current: float
    lo_suggested: float
    hi_suggested: float
    lo_drift: float
    hi_drift: float
    flagged: bool
    floor3_current: float


@dataclass
class CalibrationReport:
    backend: str
    model: str
    thresholds: tuple[float, float, float]
    anchors: tuple[float, float]
    banded_rows: list[BandedRow] = field(default_factory=list)
    band_summaries: list[BandSummary] = field(default_factory=list)
    banded_mismatches: int = 0
    paraphrase_sims: list[float] = field(default_factory=list)
    real_grid: dict | None = None
    suggestion: AnchorSuggestion | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["thresholds"] = list(self.thresholds)
        data["anchors"] = list(self.anchors)
        return data


def run_banded_pairs(
    backend: str, similarity_fn: SimilarityFn
) -> tuple[list[BandedRow], dict[str, list[float]], int] | None:
    co_norms = [normalize_text(co) for _, _, co, _ in BANDED_PAIRS]
    po_norms = [normalize_text(po) for _, _, _, po in BANDED_PAIRS]
    sims = similarity_fn(co_norms, po_norms)
    if sims is None:
        return None

    rows: list[BandedRow] = []
    by_band: dict[str, list[float]] = {}
    mismatches = 0
    for (expected, band, _co, _po), co_norm, po_norm, sim in zip(
        BANDED_PAIRS, co_norms, po_norms, sims, strict=True
    ):
        result = score_pair(co_norm, po_norm, sim, backend=backend)
        by_band.setdefault(band, []).append(sim)
        if result.score != expected:
            mismatches += 1
        rows.append(
            BandedRow(
                band=band,
                expected=expected,
                predicted=result.score,
                raw=sim,
                rescaled=rescale_similarity(sim, backend),
                composite=result.confidence,
                explanation=result.explanation,
            )
        )
    return rows, by_band, mismatches


def run_paraphrase_probes(similarity_fn: SimilarityFn) -> list[float] | None:
    a = [normalize_text(x) for x, _ in PARAPHRASE_PAIRS]
    b = [normalize_text(y) for _, y in PARAPHRASE_PAIRS]
    return similarity_fn(a, b)


def run_real_grid(
    backend: str,
    similarity_fn: SimilarityFn,
    co_texts: Sequence[str],
    po_texts: Sequence[str],
    batch_size: int = 128,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[dict, list[float]] | None:
    """Sweep the full CO x PO Cartesian grid; return (summary, raw sims).

    ``co_texts`` / ``po_texts`` are the RAW outcome statements (they are
    normalized here). Batched so bert-base stays within small-instance
    memory; ``progress_cb`` (0..1) drives UI progress bars.
    """
    co_norms = [normalize_text(t) for t in co_texts]
    po_norms = [normalize_text(t) for t in po_texts]
    pair_co: list[str] = []
    pair_po: list[str] = []
    for c in co_norms:
        for p in po_norms:
            pair_co.append(c)
            pair_po.append(p)

    sims: list[float] = []
    total = len(pair_co)
    for start in range(0, total, batch_size):
        batch = similarity_fn(pair_co[start : start + batch_size], pair_po[start : start + batch_size])
        if batch is None:
            return None
        sims.extend(batch)
        if progress_cb is not None:
            progress_cb(min(1.0, len(sims) / total))

    from collections import Counter

    labels = Counter(
        score_pair(c, p, s, backend=backend).score
        for c, p, s in zip(pair_co, pair_po, sims, strict=True)
    )
    summary = {
        "n_pairs": total,
        "n_cos": len(co_norms),
        "n_pos": len(po_norms),
        "raw_min": min(sims),
        "raw_max": max(sims),
        "raw_mean": statistics.fmean(sims),
        "percentiles": {
            f"p{p}": {
                "raw": percentile(sims, p),
                "rescaled": rescale_similarity(percentile(sims, p), backend),
            }
            for p in PERCENTILES
        },
        "label_counts": {str(lab): labels.get(lab, 0) for lab in (0, 1, 2, 3)},
        "label_percent": {
            str(lab): round(100.0 * labels.get(lab, 0) / total, 1) for lab in (0, 1, 2, 3)
        },
    }
    return summary, sims


def suggest_anchors(
    backend: str,
    by_band: dict[str, list[float]],
    paraphrase_sims: Sequence[float],
    real_sims: Sequence[float] | None,
) -> AnchorSuggestion:
    lo_cur, hi_cur = SIMILARITY_RESCALE[backend]
    lo_candidates: list[float] = []
    unrelated = by_band.get("unrelated", [])
    if unrelated:
        lo_candidates.append(statistics.fmean(unrelated))
    if real_sims:
        lo_candidates.append(percentile(real_sims, 50))
    lo_suggested = max(lo_candidates) if lo_candidates else lo_cur
    hi_suggested = statistics.fmean(paraphrase_sims) if paraphrase_sims else hi_cur
    lo_drift = abs(lo_suggested - lo_cur)
    hi_drift = abs(hi_suggested - hi_cur)
    return AnchorSuggestion(
        backend=backend,
        lo_current=lo_cur,
        hi_current=hi_cur,
        lo_suggested=round(lo_suggested, 3),
        hi_suggested=round(hi_suggested, 3),
        lo_drift=round(lo_drift, 3),
        hi_drift=round(hi_drift, 3),
        flagged=lo_drift > DRIFT_TOLERANCE or hi_drift > DRIFT_TOLERANCE,
        floor3_current=SIMILARITY_FLOOR_FOR_3[backend],
    )


def calibrate(
    backend: str,
    similarity_fn: SimilarityFn | None = None,
    model: str | None = None,
    real_co_texts: Sequence[str] | None = None,
    real_po_texts: Sequence[str] | None = None,
    batch_size: int = 128,
    progress_cb: Callable[[float], None] | None = None,
) -> CalibrationReport | None:
    """Full calibration for one backend; None when dependencies are missing."""
    fn = similarity_fn or default_similarity_fn(backend, model)

    banded = run_banded_pairs(backend, fn)
    if banded is None:
        return None
    rows, by_band, mismatches = banded

    paraphrase_sims = run_paraphrase_probes(fn)
    if paraphrase_sims is None:
        return None

    real_summary = None
    real_sims: list[float] | None = None
    if real_co_texts and real_po_texts:
        swept = run_real_grid(
            backend, fn, real_co_texts, real_po_texts, batch_size=batch_size, progress_cb=progress_cb
        )
        if swept is not None:
            real_summary, real_sims = swept

    report = CalibrationReport(
        backend=backend,
        model=model or ("sentence-transformers/all-MiniLM-L6-v2" if backend == "sbert" else "google-bert/bert-base-uncased"),
        thresholds=THRESHOLDS[backend],
        anchors=SIMILARITY_RESCALE[backend],
        banded_rows=rows,
        band_summaries=[
            BandSummary(
                band=band,
                n=len(values),
                raw_min=min(values),
                raw_max=max(values),
                raw_mean=statistics.fmean(values),
            )
            for band in ("strong", "moderate", "weak", "unrelated")
            if (values := by_band.get(band))
        ],
        banded_mismatches=mismatches,
        paraphrase_sims=list(paraphrase_sims),
        real_grid=real_summary,
        suggestion=suggest_anchors(backend, by_band, paraphrase_sims, real_sims),
    )
    return report
