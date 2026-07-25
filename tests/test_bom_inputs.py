import csv
import tempfile
import unittest
from pathlib import Path

from copo_mapper.aggregate import load_credit_rows
from copo_mapper.attainment import (
    load_co_attainment_input,
    load_mapping_matrix,
    load_weight_config,
)
from copo_mapper.pipeline import run_pairwise_mapping

# UTF-8 byte order mark prepended by Excel / Google Sheets when exporting
# CSV or JSON files. Loaders must tolerate it (read as utf-8-sig).
BOM = "﻿"


class BomInputTest(unittest.TestCase):
    def test_pipeline_handles_bom_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.csv"
            po_file = tmp_path / "po.csv"

            co_file.write_text(BOM + "CO,description\nCO1,Design databases.\n")
            po_file.write_text(BOM + "PO,description\nPO1,Solve problems.\n")

            pair_path, matrix_path = run_pairwise_mapping(
                str(co_file), str(po_file), str(tmp_path / "out")
            )

            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["co_id"], "CO1")
            self.assertEqual(rows[0]["po_id"], "PO1")
            self.assertTrue(matrix_path.exists())

    def test_pipeline_handles_bom_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"

            co_file.write_text(BOM + '[{"CO":"CO1","description":"Design software."}]')
            po_file.write_text(BOM + '[{"PO":"PO1","description":"Design solutions."}]')

            pair_path, _ = run_pairwise_mapping(
                str(co_file), str(po_file), str(tmp_path / "out")
            )
            with pair_path.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["co_id"], "CO1")

    def test_attainment_loaders_handle_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            att = tmp_path / "co_attainment.csv"
            att.write_text(
                BOM
                + "co_id,ma_attainment,ea_attainment,indirect_attainment\n"
                + "CO1,0.7,0.6,0.7\n"
            )
            inputs = load_co_attainment_input(str(att))
            self.assertEqual(inputs[0].co_id, "CO1")

            matrix = tmp_path / "matrix.csv"
            matrix.write_text(BOM + "co_id,PO1,PO2\nCO1,3,2\n")
            mapping = load_mapping_matrix(str(matrix))
            self.assertEqual(mapping["CO1"], {"PO1": 3, "PO2": 2})

            cfg = tmp_path / "config.json"
            cfg.write_text(
                BOM
                + '{"ma_weight":0.4,"ea_weight":0.6,"direct_weight":0.8,'
                + '"indirect_weight":0.2,"co_target_level":1.4,"po_target_level":1.4}'
            )
            config = load_weight_config(str(cfg))
            self.assertEqual(config.ma_weight, 0.4)

    def test_credit_rows_handle_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            courses = Path(tmp) / "courses.csv"
            courses.write_text(BOM + "course_id,credits,PO1,PO2\nDBMS,4,2.6,2.5\n")
            rows = load_credit_rows(str(courses))
            self.assertEqual(rows[0].id, "DBMS")
            self.assertEqual(rows[0].po_values, {"PO1": 2.6, "PO2": 2.5})


if __name__ == "__main__":
    unittest.main()
