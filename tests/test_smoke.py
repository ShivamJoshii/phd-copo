import csv
import tempfile
import unittest
from pathlib import Path

from copo_mapper.pipeline import run_pairwise_mapping


class PipelineSmokeTest(unittest.TestCase):
    def test_run_pairwise_mapping_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_file.write_text('[{"CO":"CO1","description":"Design software solutions."}]')
            po_file.write_text('[{"PO":"PO1","description":"Design engineering solutions."}]')

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            self.assertTrue(pair_path.exists())
            self.assertTrue(matrix_path.exists())

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["co_id"], "CO1")
            self.assertEqual(rows[0]["po_id"], "PO1")
            self.assertIn(rows[0]["predicted_strength"], {"0", "1", "2", "3"})
            self.assertIn(rows[0]["semantic_method"], {"tfidf"})

    def test_sbert_backend_raises_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_file.write_text('[{"CO":"CO1","description":"Design software solutions."}]')
            po_file.write_text('[{"PO":"PO1","description":"Design engineering solutions."}]')

            with self.assertRaises(RuntimeError):
                run_pairwise_mapping(
                    str(co_file),
                    str(po_file),
                    str(out_dir),
                    semantic_backend="sbert",
                )

    def test_bert_backend_raises_when_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_file.write_text('[{"CO":"CO1","description":"Design software solutions."}]')
            po_file.write_text('[{"PO":"PO1","description":"Design engineering solutions."}]')

            with self.assertRaises(RuntimeError):
                run_pairwise_mapping(
                    str(co_file),
                    str(po_file),
                    str(out_dir),
                    semantic_backend="bert",
                )



if __name__ == "__main__":
    unittest.main()
