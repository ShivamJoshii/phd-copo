"""Tests for copo_mapper.calibration with fake (dependency-free) backends."""

from __future__ import annotations

import math
import unittest

from copo_mapper.calibration import (
    BANDED_PAIRS,
    DRIFT_TOLERANCE,
    PARAPHRASE_PAIRS,
    calibrate,
    percentile,
    run_banded_pairs,
    suggest_anchors,
)
from copo_mapper.scoring import SIMILARITY_RESCALE


def make_fake_backend(band_sims: dict[str, float], paraphrase_sim: float, default: float = 0.2):
    """Similarity fn that recognizes the banded/paraphrase texts by content."""
    from copo_mapper.preprocess import normalize_text

    lookup: dict[tuple[str, str], float] = {}
    for _, band, co, po in BANDED_PAIRS:
        lookup[(normalize_text(co), normalize_text(po))] = band_sims[band]
    for a, b in PARAPHRASE_PAIRS:
        lookup[(normalize_text(a), normalize_text(b))] = paraphrase_sim

    def fn(co_texts: list[str], po_texts: list[str]):
        return [lookup.get((c, p), default) for c, p in zip(co_texts, po_texts, strict=True)]

    return fn


def unavailable_backend(co_texts, po_texts):
    return None


# Raw sims matching the documented sbert band arithmetic (anchors 0.25/0.75).
SBERT_LIKE = {"strong": 0.65, "moderate": 0.50, "weak": 0.35, "unrelated": 0.20}


class PercentileTest(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 50), 2.0)
        self.assertAlmostEqual(percentile(values, 25), 1.0)
        self.assertAlmostEqual(percentile(values, 100), 4.0)
        self.assertAlmostEqual(percentile(values, 0), 0.0)

    def test_percentile_empty_is_nan(self) -> None:
        self.assertTrue(math.isnan(percentile([], 50)))


class BandedPairsTest(unittest.TestCase):
    def test_unavailable_backend_returns_none(self) -> None:
        self.assertIsNone(run_banded_pairs("sbert", unavailable_backend))

    def test_sbert_like_sims_produce_band_groups(self) -> None:
        fn = make_fake_backend(SBERT_LIKE, paraphrase_sim=0.75)
        rows, by_band, _ = run_banded_pairs("sbert", fn)
        self.assertEqual(len(rows), len(BANDED_PAIRS))
        self.assertEqual(set(by_band), {"strong", "moderate", "weak", "unrelated"})
        for band, sim in SBERT_LIKE.items():
            self.assertTrue(all(abs(v - sim) < 1e-9 for v in by_band[band]))

    def test_strong_band_reaches_label_3(self) -> None:
        fn = make_fake_backend(SBERT_LIKE, paraphrase_sim=0.75)
        rows, _, _ = run_banded_pairs("sbert", fn)
        strong = [r for r in rows if r.band == "strong"]
        # Strong pairs carry heavy lexical/bloom overlap by construction, so
        # with raw sim 0.65 (rescaled 0.80, above the 0.45 floor) they must
        # reach label 3 per the documented band arithmetic.
        self.assertTrue(all(r.predicted == 3 for r in strong), [r.predicted for r in strong])

    def test_unrelated_band_stays_below_2(self) -> None:
        fn = make_fake_backend(SBERT_LIKE, paraphrase_sim=0.75)
        rows, _, _ = run_banded_pairs("sbert", fn)
        unrelated = [r for r in rows if r.band == "unrelated"]
        self.assertTrue(all(r.predicted <= 1 for r in unrelated), [r.predicted for r in unrelated])


class SuggestAnchorsTest(unittest.TestCase):
    def test_matching_anchors_not_flagged(self) -> None:
        lo, hi = SIMILARITY_RESCALE["sbert"]
        suggestion = suggest_anchors(
            "sbert",
            {"unrelated": [lo - 0.01, lo + 0.01]},
            [hi - 0.01, hi + 0.01],
            None,
        )
        self.assertFalse(suggestion.flagged)
        self.assertAlmostEqual(suggestion.lo_suggested, lo, places=3)
        self.assertAlmostEqual(suggestion.hi_suggested, hi, places=3)

    def test_drifted_anchors_flagged(self) -> None:
        lo, hi = SIMILARITY_RESCALE["sbert"]
        suggestion = suggest_anchors(
            "sbert",
            {"unrelated": [lo + 2 * DRIFT_TOLERANCE]},
            [hi + 2 * DRIFT_TOLERANCE],
            None,
        )
        self.assertTrue(suggestion.flagged)
        self.assertGreater(suggestion.lo_drift, DRIFT_TOLERANCE)
        self.assertGreater(suggestion.hi_drift, DRIFT_TOLERANCE)

    def test_real_grid_median_raises_lo(self) -> None:
        lo, _ = SIMILARITY_RESCALE["sbert"]
        real = [lo + 0.10] * 101  # p50 well above the banded unrelated mean
        suggestion = suggest_anchors("sbert", {"unrelated": [lo]}, [0.75], real)
        self.assertAlmostEqual(suggestion.lo_suggested, lo + 0.10, places=3)


class CalibrateTest(unittest.TestCase):
    def test_calibrate_full_report(self) -> None:
        fn = make_fake_backend(SBERT_LIKE, paraphrase_sim=0.74, default=0.22)
        co_texts = [co for _, _, co, _ in BANDED_PAIRS[:4]]
        po_texts = [po for _, _, _, po in BANDED_PAIRS[:3]]
        progress: list[float] = []
        report = calibrate(
            "sbert",
            similarity_fn=fn,
            real_co_texts=co_texts,
            real_po_texts=po_texts,
            batch_size=5,
            progress_cb=progress.append,
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.backend, "sbert")
        self.assertEqual(len(report.banded_rows), len(BANDED_PAIRS))
        self.assertEqual(len(report.paraphrase_sims), len(PARAPHRASE_PAIRS))
        self.assertEqual(report.real_grid["n_pairs"], 12)
        self.assertEqual(report.real_grid["n_cos"], 4)
        self.assertEqual(report.real_grid["n_pos"], 3)
        self.assertEqual(
            sum(report.real_grid["label_counts"].values()), 12
        )
        self.assertEqual(progress[-1], 1.0)
        self.assertIsNotNone(report.suggestion)
        # Suggestion must reflect the fake measurements: hi from paraphrases.
        self.assertAlmostEqual(report.suggestion.hi_suggested, 0.74, places=3)
        # Serializable end-to-end.
        data = report.to_dict()
        self.assertEqual(data["backend"], "sbert")
        self.assertEqual(len(data["banded_rows"]), len(BANDED_PAIRS))

    def test_calibrate_unavailable_returns_none(self) -> None:
        self.assertIsNone(calibrate("sbert", similarity_fn=unavailable_backend))


if __name__ == "__main__":
    unittest.main()
