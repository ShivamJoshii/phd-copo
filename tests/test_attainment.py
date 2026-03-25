import csv
import json
import tempfile
import unittest
from pathlib import Path

from copo_mapper.attainment import run_attainment_analysis


class AttainmentEngineTest(unittest.TestCase):
    def test_stage2_outputs_and_core_formulas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            co_file = root / "co_attainment.json"
            cfg_file = root / "config.json"
            matrix_file = root / "matrix.csv"
            out_dir = root / "out"

            co_file.write_text(
                json.dumps(
                    [
                        {
                            "co_id": "CO1",
                            "ma_attainment": 0.7786,
                            "ea_attainment": 0.7256,
                            "indirect_attainment": 0.7965,
                        },
                        {
                            "co_id": "CO2",
                            "ma_attainment": 0.8403,
                            "ea_attainment": 0.6123,
                            "indirect_attainment": 0.6831,
                        },
                    ]
                )
            )
            cfg_file.write_text(
                json.dumps(
                    {
                        "ma_weight": 0.4,
                        "ea_weight": 0.6,
                        "direct_weight": 0.8,
                        "indirect_weight": 0.2,
                        "co_target_level": 1.4,
                        "po_target_level": 1.4,
                    }
                )
            )
            matrix_file.write_text("co_id,PO1,PO2\nCO1,3,1\nCO2,3,2\n")

            paths = run_attainment_analysis(
                co_attainment_file=str(co_file),
                mapping_matrix_file=str(matrix_file),
                config_file=str(cfg_file),
                out_dir=str(out_dir),
            )

            self.assertTrue(paths["co_summary"].exists())
            self.assertTrue(paths["po_summary"].exists())
            self.assertTrue(paths["target_achievement"].exists())
            self.assertTrue(paths["course_summary"].exists())

            with paths["co_summary"].open() as f:
                co_rows = list(csv.DictReader(f))

            # CO1 direct = 0.7786*0.4 + 0.7256*0.6 = 0.7468
            self.assertEqual(co_rows[0]["direct_attainment"], "0.7468")
            # CO1 final = 0.7468*0.8 + 0.7965*0.2 = 0.7567
            self.assertEqual(co_rows[0]["final_attainment"], "0.7567")
            # scaled = 2.27, target 1.4 => Y
            self.assertEqual(co_rows[0]["scaled_attainment"], "2.27")
            self.assertEqual(co_rows[0]["target_achieved"], "Y")

            with paths["po_summary"].open() as f:
                po_rows = list(csv.DictReader(f))
            self.assertEqual({row["po_id"] for row in po_rows}, {"PO1", "PO2"})


if __name__ == "__main__":
    unittest.main()
