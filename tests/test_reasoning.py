import csv
import tempfile
import unittest
from pathlib import Path

from copo_mapper.pipeline import run_pairwise_mapping
from copo_mapper.preprocess import normalize_text
from copo_mapper.reasoning import STRENGTH_LABELS, shared_terms
from copo_mapper.scoring import score_pair
from copo_mapper.features import token_set


class PairReasoningTest(unittest.TestCase):
    def test_reasoning_starts_with_strength_label(self) -> None:
        for sim in (0.0, 0.2, 0.4, 0.9):
            r = score_pair("design database schemas", "design engineering solutions", sim)
            self.assertTrue(r.reasoning.startswith(f"{STRENGTH_LABELS[r.score]} ({r.score})."))

    def test_legacy_explanation_untouched(self) -> None:
        a = score_pair("design database schemas", "design engineering solutions", 0.3)
        self.assertIn("semantic=0.30;", a.explanation)
        self.assertNotIn("Moderate", a.explanation)

    def test_shared_terms_listed(self) -> None:
        r = score_pair(
            normalize_text("Design relational database schemas."),
            normalize_text("Design and manage database systems."),
            0.4,
        )
        self.assertIn('"design"', r.reasoning)
        self.assertIn('"database"', r.reasoning)

    def test_bloom_only_rating_is_flagged(self) -> None:
        # No shared content, same Bloom level -> rated >=1 by Bloom alone.
        r = score_pair("analyze market trends", "analyze circuit faults", 0.0)
        self.assertGreaterEqual(r.score, 1)
        self.assertIn("rests mainly on cognitive-level alignment", r.reasoning)

    def test_capped_rating_explained(self) -> None:
        r = score_pair(
            "design and implement database systems",
            "design and implement database systems",
            0.30,
            backend="sbert",
        )
        if "capped" in r.explanation:
            self.assertIn("held at Moderate", r.reasoning)

    def test_not_higher_explanation(self) -> None:
        r = score_pair("communicate with audiences", "evaluate system trade-offs", 0.0)
        self.assertIn("is needed for", r.reasoning)

    def test_shared_terms_uses_surface_words(self) -> None:
        stems = frozenset(token_set("databases designing") & token_set("database design"))
        self.assertEqual(shared_terms("Designing Databases", stems), ["designing", "databases"])

    def test_pipeline_writes_reasoning_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "co.csv").write_text("CO,description\nCO1,Design software solutions.\n")
            (t / "po.csv").write_text("PO,description\nPO1,Design engineering solutions.\n")
            pair_path, _ = run_pairwise_mapping(str(t / "co.csv"), str(t / "po.csv"), str(t / "out"))
            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows[0]["reasoning"])


if __name__ == "__main__":
    unittest.main()
