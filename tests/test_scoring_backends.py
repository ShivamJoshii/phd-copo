import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copo_mapper.pipeline import run_pairwise_mapping
from copo_mapper.scoring import SIMILARITY_FLOOR_FOR_3, THRESHOLDS, score_pair

# Feature-controlled fixture pairs (composite = 0.45*sim + feature part):
#
# NO_OVERLAP: disjoint tokens, no domain terms, no bloom verbs (both sides
#   default to "understand", gap 0 -> bloom term 0.2).
#   composite = 0.45*sim + 0.2
NO_OVERLAP = ("alpha", "beta")
#
# BLOOM_FAR: "define" (remember) vs "design" (create), gap 5 -> bloom term 0;
#   disjoint tokens, no domain terms.
#   composite = 0.45*sim
BLOOM_FAR = ("define terms", "design artifacts")
#
# FULL_OVERLAP: identical texts -> token jaccard 1 (0.15), domain jaccard 1
#   (0.2: databases + algorithms), same bloom level (0.2).
#   composite = 0.45*sim + 0.55
FULL_OVERLAP = ("implement database optimization", "implement database optimization")


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


class SbertThresholdTest(unittest.TestCase):
    def test_same_composite_yields_lower_label_than_tfidf(self) -> None:
        co, po = NO_OVERLAP
        # Unrelated MiniLM pair (sim 0.2 -> composite 0.29): tfidf says 1,
        # sbert says 0.
        self.assertEqual(score_pair(co, po, 0.2, backend="tfidf").score, 1)
        self.assertEqual(score_pair(co, po, 0.2, backend="sbert").score, 0)
        # Weak MiniLM pair (sim 0.5 -> composite 0.425): tfidf says 2,
        # sbert says 1.
        self.assertEqual(score_pair(co, po, 0.5, backend="tfidf").score, 2)
        self.assertEqual(score_pair(co, po, 0.5, backend="sbert").score, 1)
        # Strong-ish sim without feature support (sim 0.68 -> composite
        # 0.506): tfidf says 3, sbert says 2.
        self.assertEqual(score_pair(co, po, 0.68, backend="tfidf").score, 3)
        self.assertEqual(score_pair(co, po, 0.68, backend="sbert").score, 2)

    def test_strong_pair_with_feature_overlap_scores_3(self) -> None:
        co, po = FULL_OVERLAP  # composite = 0.45*0.68 + 0.55 = 0.856
        self.assertEqual(score_pair(co, po, 0.68, backend="sbert").score, 3)

    def test_sbert_boundaries(self) -> None:
        co, po = NO_OVERLAP  # composite = 0.45*sim + 0.2
        self.assertEqual(score_pair(co, po, 0.934, backend="sbert").score, 3)  # 0.6203
        self.assertEqual(score_pair(co, po, 0.933, backend="sbert").score, 2)  # 0.61985
        self.assertEqual(score_pair(co, po, 0.667, backend="sbert").score, 2)  # 0.50015
        self.assertEqual(score_pair(co, po, 0.666, backend="sbert").score, 1)  # 0.49970
        self.assertEqual(score_pair(co, po, 0.289, backend="sbert").score, 1)  # 0.33005
        self.assertEqual(score_pair(co, po, 0.288, backend="sbert").score, 0)  # 0.32960


class BertThresholdTest(unittest.TestCase):
    def test_bert_labels(self) -> None:
        co, po = NO_OVERLAP  # composite = 0.45*sim + 0.2
        # Mean-pooled BERT cosines run high: 0.55 is typical for unrelated
        # texts. tfidf cutoffs would call this 2; bert cutoffs call it 0.
        self.assertEqual(score_pair(co, po, 0.55, backend="tfidf").score, 2)
        self.assertEqual(score_pair(co, po, 0.55, backend="bert").score, 0)  # 0.4475
        self.assertEqual(score_pair(co, po, 0.75, backend="bert").score, 1)  # 0.5375
        self.assertEqual(score_pair(co, po, 0.95, backend="bert").score, 2)  # 0.6275
        co_full, po_full = FULL_OVERLAP  # composite = 0.45*0.85 + 0.55 = 0.9325
        self.assertEqual(score_pair(co_full, po_full, 0.85, backend="bert").score, 3)


class SimilarityFloorTest(unittest.TestCase):
    def test_sbert_label_3_requires_similarity_floor(self) -> None:
        # Maximal feature overlap with weak MiniLM similarity: composite
        # 0.45*0.30 + 0.55 = 0.685 >= 0.62, but sim 0.30 < 0.45 floor -> 2.
        co, po = FULL_OVERLAP
        result = score_pair(co, po, 0.30, backend="sbert")
        self.assertEqual(result.score, 2)
        self.assertIn("capped at 2", result.explanation)

    def test_sbert_label_3_allowed_above_floor(self) -> None:
        co, po = FULL_OVERLAP  # composite = 0.45*0.46 + 0.55 = 0.757
        result = score_pair(co, po, 0.46, backend="sbert")
        self.assertEqual(result.score, 3)
        self.assertNotIn("capped", result.explanation)

    def test_bert_label_3_requires_similarity_floor(self) -> None:
        # composite = 0.45*0.60 + 0.55 = 0.82 >= 0.72, but 0.60 < 0.70 -> 2.
        co, po = FULL_OVERLAP
        result = score_pair(co, po, 0.60, backend="bert")
        self.assertEqual(result.score, 2)
        self.assertIn("capped at 2", result.explanation)


class BackendValidationTest(unittest.TestCase):
    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            score_pair("alpha", "beta", 0.5, backend="word2vec")

    def test_backend_is_case_insensitive(self) -> None:
        co, po = NO_OVERLAP
        upper = score_pair(co, po, 0.2, backend="SBERT")
        lower = score_pair(co, po, 0.2, backend="sbert")
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

            # sim 0.2 with no feature overlap -> composite 0.29: label 1
            # under tfidf cutoffs, 0 under sbert cutoffs. A "0" proves the
            # backend reached score_pair.
            with mock.patch(
                "copo_mapper.pipeline.sbert_pair_similarity", return_value=[0.2]
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
            self.assertEqual(rows[0]["predicted_strength"], "0")
            self.assertEqual(rows[0]["requested_backend"], "sbert")
            self.assertTrue(rows[0]["semantic_method"].startswith("sbert:"))

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


if __name__ == "__main__":
    unittest.main()
