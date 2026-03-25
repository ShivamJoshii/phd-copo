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
                {"id": "CO1", "text": "Analyze data structures for performance."},
                {"id": "CO2", "text": "Design database schemas for applications."},
                {"id": "CO3", "text": "Implement software testing workflows."},
                {"id": "CO4", "text": "Communicate technical findings effectively."},
            ]
            pos = [
                {"id": "PO1", "text": "Analyze complex engineering problems."},
                {"id": "PO2", "text": "Design robust engineering solutions."},
                {"id": "PO3", "text": "Use modern engineering tools."},
                {"id": "PO4", "text": "Communicate with diverse stakeholders."},
                {"id": "PO5", "text": "Apply professional ethics in engineering."},
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

            expected_pairs = {(co["id"], po["id"]) for co in cos for po in pos}
            actual_pairs = {(row["co_id"], row["po_id"]) for row in pair_rows}
            self.assertSetEqual(actual_pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
