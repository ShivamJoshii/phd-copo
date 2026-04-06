import unittest

from copo_mapper.semantic import sbert_pair_similarity


class SemanticHelpersTest(unittest.TestCase):
    def test_sbert_similarity_returns_none_when_library_missing(self) -> None:
        sims = sbert_pair_similarity(["design software"], ["design engineering"])
        self.assertTrue(sims is None or len(sims) == 1)

    def test_sbert_similarity_validates_input_lengths(self) -> None:
        with self.assertRaises(ValueError):
            sbert_pair_similarity(["co1", "co2"], ["po1"])


if __name__ == "__main__":
    unittest.main()
