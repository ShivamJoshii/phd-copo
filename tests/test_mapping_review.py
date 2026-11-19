import unittest

from copo_mapper.mapping_review import (
    MappingOverride,
    annotate_pair_rows,
    apply_overrides,
    build_overrides,
    diff_matrices,
    matrix_from_rows,
    matrix_to_rows,
    overrides_from_csv,
    overrides_to_csv,
)

HEADER = ["co_id", "PO1", "PO2"]
ROWS = [["CO1", "1", "0"], ["CO2", "2", "3"]]


class MappingReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predicted = matrix_from_rows(HEADER, ROWS)

    def test_matrix_round_trip(self) -> None:
        self.assertEqual(matrix_to_rows(self.predicted, HEADER[1:]), ROWS)

    def test_diff_and_build_overrides(self) -> None:
        edited = {"CO1": {"PO1": 3, "PO2": 0}, "CO2": {"PO1": 2, "PO2": 1}}
        self.assertEqual(diff_matrices(self.predicted, edited), [("CO1", "PO1", 1, 3), ("CO2", "PO2", 3, 1)])
        reasons = {
            ("CO1", "PO1"): ("Stronger link than predicted", "Lab project assesses this PO directly."),
            ("CO2", "PO2"): ("Weaker link than predicted", "Only touched in one lecture."),
        }
        ovs = build_overrides(self.predicted, edited, reasons, reviewer="Dr. R", now="t")
        self.assertEqual(len(ovs), 2)
        self.assertEqual(ovs[0].final_strength, 3)
        self.assertEqual(ovs[0].reviewer, "Dr. R")
        reviewed = apply_overrides(self.predicted, ovs)
        self.assertEqual(reviewed, edited)
        # predicted matrix is untouched
        self.assertEqual(self.predicted["CO1"]["PO1"], 1)

    def test_resave_keeps_original_reviewer(self) -> None:
        edited = {"CO1": {"PO1": 3, "PO2": 0}, "CO2": {"PO1": 2, "PO2": 3}}
        reasons = {("CO1", "PO1"): ("Other", "why")}
        first = build_overrides(self.predicted, edited, reasons, reviewer="A", now="t1")
        again = build_overrides(self.predicted, edited, reasons, reviewer="", now="t2", previous=first)
        self.assertEqual(again, first)
        changed = build_overrides(
            self.predicted, edited, {("CO1", "PO1"): ("Other", "new why")}, reviewer="B", now="t3", previous=first
        )
        self.assertEqual((changed[0].reviewer, changed[0].timestamp), ("B", "t3"))

    def test_reason_required_for_every_change(self) -> None:
        edited = {"CO1": {"PO1": 3, "PO2": 0}, "CO2": {"PO1": 2, "PO2": 3}}
        with self.assertRaises(ValueError) as ctx:
            build_overrides(self.predicted, edited, {("CO1", "PO1"): ("Other", "  ")})
        self.assertIn("CO1→PO1", str(ctx.exception))

    def test_csv_round_trip_and_unknown_cells_ignored(self) -> None:
        ovs = [
            MappingOverride("CO1", "PO2", 0, 2, "Directly assessed in course", 'Quoted, "reason"', "A", "t"),
            MappingOverride("CO9", "PO1", 0, 3, "Other", "not in this run"),
        ]
        loaded = overrides_from_csv("﻿" + overrides_to_csv(ovs))
        self.assertEqual(loaded, ovs)
        reviewed = apply_overrides(self.predicted, loaded)
        self.assertEqual(reviewed["CO1"]["PO2"], 2)
        self.assertNotIn("CO9", reviewed)

    def test_invalid_strength_rejected(self) -> None:
        text = "co_id,po_id,predicted_strength,final_strength,category,reason\nCO1,PO1,1,5,Other,x\n"
        with self.assertRaises(ValueError):
            overrides_from_csv(text)

    def test_annotate_pair_rows(self) -> None:
        rows = [
            {"co_id": "CO1", "po_id": "PO1", "predicted_strength": "1"},
            {"co_id": "CO1", "po_id": "PO2", "predicted_strength": "0"},
        ]
        ov = MappingOverride("CO1", "PO1", 1, 3, "Other", "why")
        out = annotate_pair_rows(rows, [ov])
        self.assertEqual(out[0]["final_strength"], "3")
        self.assertEqual(out[0]["faculty_override"], "Y")
        self.assertEqual(out[1]["final_strength"], "0")
        self.assertEqual(out[1]["faculty_override"], "N")


if __name__ == "__main__":
    unittest.main()
