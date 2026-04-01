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

            co_file.write_text('[{"id":"CO1","text":"Design software solutions."}]')
            po_file.write_text('[{"id":"PO1","text":"Design engineering solutions."}]')

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            self.assertTrue(pair_path.exists())
            self.assertTrue(matrix_path.exists())

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["co_id"], "CO1")
            self.assertEqual(rows[0]["po_id"], "PO1")
            self.assertIn(rows[0]["predicted_strength"], {"0", "1", "2", "3"})

    def test_run_pairwise_mapping_with_csv_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.csv"
            po_file = tmp_path / "po.csv"
            out_dir = tmp_path / "out"

            co_file.write_text("id,text\nCO1,Design software solutions.\n")
            po_file.write_text("id,text\nPO1,Design engineering solutions.\n")

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            self.assertTrue(pair_path.exists())
            self.assertTrue(matrix_path.exists())

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["co_id"], "CO1")
            self.assertEqual(rows[0]["po_id"], "PO1")
            self.assertIn(rows[0]["predicted_strength"], {"0", "1", "2", "3"})

    def test_run_pairwise_mapping_with_co_po_description_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.csv"
            po_file = tmp_path / "po.csv"
            out_dir = tmp_path / "out"

            co_file.write_text("CO,Description\nCO1,Design software solutions.\n")
            po_file.write_text("PO,Description\nPO1,Design engineering solutions.\n")

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            self.assertTrue(pair_path.exists())
            self.assertTrue(matrix_path.exists())

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["co_id"], "CO1")
            self.assertEqual(rows[0]["po_id"], "PO1")
            self.assertIn(rows[0]["predicted_strength"], {"0", "1", "2", "3"})


if __name__ == "__main__":
    unittest.main()
