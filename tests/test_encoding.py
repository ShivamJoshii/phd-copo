import tempfile
import unittest
from pathlib import Path

from copo_mapper.io_utils import decode_text_bytes
from copo_mapper.pipeline import run_pairwise_mapping


class EncodingHandlingTest(unittest.TestCase):
    def test_decode_text_bytes_supports_cp1252(self) -> None:
        raw = "CO,description\nCO1,Analyse naïve systems\n".encode("cp1252")
        decoded = decode_text_bytes(raw, source="cp1252.csv")
        self.assertIn("naïve", decoded)

    def test_pipeline_accepts_non_utf8_csv_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.csv"
            po_file = tmp_path / "po.csv"
            out_dir = tmp_path / "out"

            co_file.write_bytes("CO,description\nCO1,Design naïve interfaces\n".encode("cp1252"))
            po_file.write_bytes("PO,description\nPO1,Design robust interfaces\n".encode("cp1252"))

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            self.assertTrue(pair_path.exists())
            self.assertTrue(matrix_path.exists())


if __name__ == "__main__":
    unittest.main()
