import csv
import json
import tempfile
import unittest
from pathlib import Path

from copo_mapper.pipeline import run_pairwise_mapping


class DynamicSizePipelineTest(unittest.TestCase):
    def test_full_cartesian_product_for_arbitrary_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_file = tmp_path / "co.json"
            po_file = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            cos = [
                {"CO": "CO1", "Description for CO": "Analyze data structures for performance."},
                {"CO": "CO2", "Description for CO": "Design database schemas for applications."},
                {"CO": "CO3", "Description for CO": "Implement software testing workflows."},
                {"CO": "CO4", "Description for CO": "Communicate technical findings effectively."},
            ]
            pos = [
                {"PO": "PO1", "Description for PO": "Analyze complex engineering problems."},
                {"PO": "PO2", "Description for PO": "Design robust engineering solutions."},
                {"PO": "PO3", "Description for PO": "Use modern engineering tools."},
                {"PO": "PO4", "Description for PO": "Communicate with diverse stakeholders."},
                {"PO": "PO5", "Description for PO": "Apply professional ethics in engineering."},
            ]

            co_file.write_text(json.dumps(cos))
            po_file.write_text(json.dumps(pos))

            pair_path, matrix_path = run_pairwise_mapping(str(co_file), str(po_file), str(out_dir))

            with pair_path.open() as f:
                pair_rows = list(csv.DictReader(f))
            self.assertEqual(len(pair_rows), len(cos) * len(pos))

            with matrix_path.open() as f:
                matrix_rows = list(csv.reader(f))
            # header + one row per CO
            self.assertEqual(len(matrix_rows), len(cos) + 1)
            # co_id + one column per PO
            self.assertEqual(len(matrix_rows[0]), len(pos) + 1)

            expected_pairs = {(co["CO"], po["PO"]) for co in cos for po in pos}
            actual_pairs = {(row["co_id"], row["po_id"]) for row in pair_rows}
            self.assertSetEqual(actual_pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
