import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copo_mapper.pipeline import run_pairwise_mapping
from copo_mapper.scoring import (
    SIMILARITY_FLOOR_FOR_3,
    SIMILARITY_RESCALE,
    THRESHOLDS,
    rescale_similarity,
    score_pair,
)

# Feature-controlled fixture pairs (composite = 0.45*rescaled_sim + feature part):
#
# NO_OVERLAP: disjoint tokens, no domain terms, no bloom verbs (both sides
#   default to "understand", gap 0 -> bloom term 0.2).
#   composite = 0.45*rescaled + 0.2
NO_OVERLAP = ("alpha", "beta")
#
# BLOOM_FAR: "define" (remember) vs "design" (create), gap 5 -> bloom term 0;
#   disjoint tokens, no domain terms.
#   composite = 0.45*rescaled
BLOOM_FAR = ("define terms", "design artifacts")
#
# FULL_OVERLAP: identical texts -> token jaccard 1 (0.15), domain jaccard 1
#   (0.2: databases + algorithms), same bloom level (0.2).
#   composite = 0.45*rescaled + 0.55
FULL_OVERLAP = ("implement database optimization", "implement database optimization")


class RescaleFunctionTest(unittest.TestCase):
    def test_anchor_dict_values(self) -> None:
        self.assertEqual(SIMILARITY_RESCALE["tfidf"], (0.0, 1.0))
        self.assertEqual(SIMILARITY_RESCALE["sbert"], (0.25, 0.75))
        self.assertEqual(SIMILARITY_RESCALE["bert"], (0.55, 0.90))

    def test_tfidf_is_identity(self) -> None:
        for sim in (0.0, 0.1, 0.35, 0.667, 1.0):
            self.assertEqual(rescale_similarity(sim, "tfidf"), sim)

    def test_sbert_anchor_endpoints(self) -> None:
        self.assertAlmostEqual(rescale_similarity(0.25, "sbert"), 0.0)
        self.assertAlmostEqual(rescale_similarity(0.75, "sbert"), 1.0)
        self.assertAlmostEqual(rescale_similarity(0.50, "sbert"), 0.5)
        self.assertAlmostEqual(rescale_similarity(0.65, "sbert"), 0.8)

    def test_sbert_clamping(self) -> None:
        self.assertEqual(rescale_similarity(0.10, "sbert"), 0.0)
        self.assertEqual(rescale_similarity(-0.2, "sbert"), 0.0)
        self.assertEqual(rescale_similarity(0.90, "sbert"), 1.0)

    def test_bert_anchor_endpoints(self) -> None:
        self.assertAlmostEqual(rescale_similarity(0.55, "bert"), 0.0)
        self.assertAlmostEqual(rescale_similarity(0.90, "bert"), 1.0)
        self.assertAlmostEqual(rescale_similarity(0.725, "bert"), 0.5)

    def test_bert_clamping(self) -> None:
        self.assertEqual(rescale_similarity(0.30, "bert"), 0.0)
        self.assertEqual(rescale_similarity(0.99, "bert"), 1.0)


class SharedThresholdTest(unittest.TestCase):
    def test_all_backends_share_the_validated_cutoffs(self) -> None:
        # Similarity is normalized per backend BEFORE the composite, so every
        # backend uses the single threshold set validated on real data.
        for backend in ("tfidf", "sbert", "bert"):
            self.assertEqual(THRESHOLDS[backend], (0.50, 0.30, 0.10))


class TfidfBackwardCompatTest(unittest.TestCase):
    def test_default_backend_is_tfidf(self) -> None:
        co, po = NO_OVERLAP
        default = score_pair(co, po, 0.2)
        explicit = score_pair(co, po, 0.2, backend="tfidf")
        self.assertEqual(default, explicit)

    def test_tfidf_thresholds_unchanged(self) -> None:
        self.assertEqual(THRESHOLDS["tfidf"], (0.50, 0.30, 0.10))
        self.assertEqual(SIMILARITY_FLOOR_FOR_3["tfidf"], 0.0)

    def test_tfidf_labels_unchanged(self) -> None:
        co, po = NO_OVERLAP
        self.assertEqual(score_pair(co, po, 0.7).score, 3)  # composite 0.515
        self.assertEqual(score_pair(co, po, 0.3).score, 2)  # composite 0.335
        self.assertEqual(score_pair(co, po, 0.0).score, 1)  # composite 0.200
        co_far, po_far = BLOOM_FAR
        self.assertEqual(score_pair(co_far, po_far, 0.1).score, 0)  # composite 0.045

    def test_tfidf_boundaries(self) -> None:
        co, po = NO_OVERLAP  # composite = 0.45*sim + 0.2
        self.assertEqual(score_pair(co, po, 0.667).score, 3)  # 0.50015 >= 0.50
        self.assertEqual(score_pair(co, po, 0.666).score, 2)  # 0.49970 <  0.50
        self.assertEqual(score_pair(co, po, 0.223).score, 2)  # 0.30035 >= 0.30
        self.assertEqual(score_pair(co, po, 0.222).score, 1)  # 0.29990 <  0.30
        co_far, po_far = BLOOM_FAR  # composite = 0.45*sim
        self.assertEqual(score_pair(co_far, po_far, 0.223).score, 1)  # 0.10035 >= 0.10
        self.assertEqual(score_pair(co_far, po_far, 0.222).score, 0)  # 0.09990 <  0.10

    def test_tfidf_has_no_similarity_floor(self) -> None:
        # Full feature overlap alone (composite 0.55) reaches label 3 under
        # tfidf, exactly as before the refactor.
        co, po = FULL_OVERLAP
        result = score_pair(co, po, 0.0, backend="tfidf")
        self.assertEqual(result.score, 3)
        self.assertNotIn("capped", result.explanation)

    def test_tfidf_explanation_format_unchanged(self) -> None:
        # The identity backend keeps the legacy explanation format: no
        # "(raw ...)" suffix, byte-identical to pre-refactor output.
        co, po = NO_OVERLAP
        result = score_pair(co, po, 0.3, backend="tfidf")
        self.assertIn("semantic=0.30;", result.explanation)
        self.assertNotIn("raw", result.explanation)


class SbertBandTest(unittest.TestCase):
    """Raw MiniLM sims per band, through the rescale + shared thresholds."""

    def test_unrelated_raw_sim_stays_at_0_or_low_1(self) -> None:
        # Raw 0.20 is below the unrelated floor -> rescaled 0.0: the
        # composite is features-only.
        co, po = NO_OVERLAP  # bloom-default worst case: composite 0.20
        result = score_pair(co, po, 0.20, backend="sbert")
        self.assertEqual(result.score, 1)  # 0.20 < t2=0.30: never label 2
        co_far, po_far = BLOOM_FAR  # composite 0.0
        self.assertEqual(score_pair(co_far, po_far, 0.20, backend="sbert").score, 0)

    def test_bloom_default_does_not_push_unrelated_to_2(self) -> None:
        # Both texts verb-less -> both default to "understand" -> bloom term
        # 0.20 exactly. Raw 0.25 (the anchor floor) must not reach label 2.
        co, po = NO_OVERLAP
        result = score_pair(co, po, 0.25, backend="sbert")
        self.assertEqual(result.score, 1)  # composite 0.20 < 0.30

    def test_weak_raw_sim_is_1(self) -> None:
        co, po = NO_OVERLAP  # composite = 0.45*rescaled + 0.2
        # raw 0.35 -> rescaled 0.20 -> 0.09 + 0.2 = 0.29 -> 1
        self.assertEqual(score_pair(co, po, 0.35, backend="sbert").score, 1)

    def test_moderate_raw_sim_is_2(self) -> None:
        co, po = NO_OVERLAP
        # raw 0.50 -> rescaled 0.50 -> 0.225 + 0.2 = 0.425 -> 2
        self.assertEqual(score_pair(co, po, 0.50, backend="sbert").score, 2)

    def test_strong_raw_sim_is_3(self) -> None:
        co, po = NO_OVERLAP
        # raw 0.65 -> rescaled 0.80 -> 0.36 + 0.2 = 0.56 -> 3 (raw >= 0.45 floor)
        result = score_pair(co, po, 0.65, backend="sbert")
        self.assertEqual(result.score, 3)
        self.assertNotIn("capped", result.explanation)

    def test_label_2_boundary(self) -> None:
        co, po = NO_OVERLAP  # composite = 0.45*rescaled + 0.2 >= 0.30 iff rescaled >= 0.2222
        self.assertEqual(score_pair(co, po, 0.362, backend="sbert").score, 2)  # 0.3008
        self.assertEqual(score_pair(co, po, 0.360, backend="sbert").score, 1)  # 0.2990

    def test_rescale_saturates_at_ceiling(self) -> None:
        co, po = BLOOM_FAR  # composite = 0.45*rescaled
        high = score_pair(co, po, 0.90, backend="sbert")  # rescaled clamps to 1.0
        ceiling = score_pair(co, po, 0.75, backend="sbert")
        self.assertEqual(high.confidence, ceiling.confidence)  # both 0.45


class BertBandTest(unittest.TestCase):
    """Raw mean-pooled BERT sims per band (anisotropic: everything runs high)."""

    def test_unrelated_raw_sim_stays_at_0_or_low_1(self) -> None:
        # Raw 0.55 is typical for UNRELATED text under mean-pooled BERT.
        co, po = NO_OVERLAP  # bloom-default worst case
        self.assertEqual(score_pair(co, po, 0.55, backend="bert").score, 1)  # 0.20
        co_far, po_far = BLOOM_FAR
        self.assertEqual(score_pair(co_far, po_far, 0.55, backend="bert").score, 0)  # 0.0

    def test_weak_raw_sim_is_1(self) -> None:
        co_far, po_far = BLOOM_FAR  # composite = 0.45*rescaled
        # raw 0.65 -> rescaled 0.286 -> composite 0.129 -> 1
        self.assertEqual(score_pair(co_far, po_far, 0.65, backend="bert").score, 1)

    def test_moderate_raw_sim_is_2(self) -> None:
        co, po = NO_OVERLAP
        # raw 0.75 -> rescaled 0.571 -> 0.257 + 0.2 = 0.457 -> 2
        self.assertEqual(score_pair(co, po, 0.75, backend="bert").score, 2)

    def test_strong_raw_sim_is_3(self) -> None:
        co, po = FULL_OVERLAP  # composite = 0.45*rescaled + 0.55
        # raw 0.85 -> rescaled 0.857 -> 0.386 + 0.55 = 0.936 -> 3 (raw >= 0.70 floor)
        result = score_pair(co, po, 0.85, backend="bert")
        self.assertEqual(result.score, 3)
        self.assertNotIn("capped", result.explanation)


class SimilarityFloorTest(unittest.TestCase):
    def test_sbert_label_3_requires_raw_similarity_floor(self) -> None:
        # Maximal feature overlap with weak MiniLM similarity: raw 0.30 ->
        # rescaled 0.10 -> composite 0.045 + 0.55 = 0.595 >= 0.50, but raw
        # 0.30 < 0.45 floor -> capped at 2.
        co, po = FULL_OVERLAP
        result = score_pair(co, po, 0.30, backend="sbert")
        self.assertEqual(result.score, 2)
        self.assertIn("capped at 2", result.explanation)

    def test_sbert_label_3_allowed_above_floor(self) -> None:
        co, po = FULL_OVERLAP  # raw 0.46 -> rescaled 0.42 -> 0.189 + 0.55 = 0.739
        result = score_pair(co, po, 0.46, backend="sbert")
        self.assertEqual(result.score, 3)
        self.assertNotIn("capped", result.explanation)

    def test_bert_label_3_requires_raw_similarity_floor(self) -> None:
        # raw 0.60 -> rescaled 0.143 -> 0.064 + 0.55 = 0.614 >= 0.50, but
        # raw 0.60 < 0.70 floor -> capped at 2.
        co, po = FULL_OVERLAP
        result = score_pair(co, po, 0.60, backend="bert")
        self.assertEqual(result.score, 2)
        self.assertIn("capped at 2", result.explanation)

    def test_floor_values(self) -> None:
        self.assertEqual(SIMILARITY_FLOOR_FOR_3["tfidf"], 0.0)
        self.assertEqual(SIMILARITY_FLOOR_FOR_3["sbert"], 0.45)
        self.assertEqual(SIMILARITY_FLOOR_FOR_3["bert"], 0.70)


class ExplanationTest(unittest.TestCase):
    def test_sbert_explanation_shows_rescaled_and_raw(self) -> None:
        co, po = NO_OVERLAP
        result = score_pair(co, po, 0.65, backend="sbert")
        self.assertIn("semantic=0.80 (raw 0.65)", result.explanation)

    def test_bert_explanation_shows_rescaled_and_raw(self) -> None:
        co, po = NO_OVERLAP
        result = score_pair(co, po, 0.725, backend="bert")
        self.assertIn("semantic=0.50 (raw 0.72)", result.explanation)

    def test_sbert_clamped_explanation_keeps_raw(self) -> None:
        co, po = NO_OVERLAP
        result = score_pair(co, po, 0.10, backend="sbert")
        self.assertIn("semantic=0.00 (raw 0.10)", result.explanation)


class BackendValidationTest(unittest.TestCase):
    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            score_pair("alpha", "beta", 0.5, backend="word2vec")

    def test_backend_is_case_insensitive(self) -> None:
        co, po = NO_OVERLAP
        upper = score_pair(co, po, 0.35, backend="SBERT")
        lower = score_pair(co, po, 0.35, backend="sbert")
        self.assertEqual(upper, lower)


class PipelineBackendPassThroughTest(unittest.TestCase):
    def test_pipeline_passes_backend_to_score_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_file.write_text('[{"CO":"CO1","description":"alpha"}]')
            po_file.write_text('[{"PO":"PO1","description":"beta"}]')

            # sim 0.35 with no feature overlap: under tfidf (identity) the
            # composite is 0.3575 -> label 2; under sbert it is rescaled to
            # 0.20 -> composite 0.29 -> label 1. A "1" plus the "(raw 0.35)"
            # explanation proves the backend reached score_pair.
            with mock.patch(
                "copo_mapper.pipeline.sbert_pair_similarity", return_value=[0.35]
            ):
                pair_path, _ = run_pairwise_mapping(
                    str(co_file),
                    str(po_file),
                    str(out_dir),
                    semantic_backend="sbert",
                )

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["predicted_strength"], "1")
            self.assertEqual(rows[0]["requested_backend"], "sbert")
            self.assertTrue(rows[0]["semantic_method"].startswith("sbert:"))
            self.assertIn("(raw 0.35)", rows[0]["explanation"])

    def test_pipeline_tfidf_default_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_file.write_text('[{"CO":"CO1","description":"alpha"}]')
            po_file.write_text('[{"PO":"PO1","description":"beta"}]')

            pair_path, _ = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            # tfidf cosine of disjoint texts is 0 -> composite 0.2 -> label 1
            # under the unchanged tfidf cutoffs.
            self.assertEqual(rows[0]["predicted_strength"], "1")
            self.assertEqual(rows[0]["semantic_method"], "tfidf")
            self.assertNotIn("raw", rows[0]["explanation"])


if __name__ == "__main__":
    unittest.main()
