import csv
import tempfile
import unittest
from pathlib import Path

from copo_mapper.aggregate import (
    CreditRow,
    aggregate_by_credits,
    compute_program_po,
    compute_semester_po,
    load_credit_rows,
    run_program_aggregation,
    run_semester_aggregation,
)
from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_direct_attainment,
    compute_po_attainment,
)


SPEC_CONFIG = WeightConfig(
    ma_weight=0.30,
    ea_weight=0.50,
    direct_weight=0.80,
    indirect_weight=0.20,
    co_target_level=2.0,
    po_target_level=2.0,
)


class DirectFormulaTest(unittest.TestCase):
    def test_normalized_direct_matches_spec(self) -> None:
        # CO1: (2.5*0.3 + 2.8*0.5) / 0.8 = 2.6875 -> 2.69
        self.assertAlmostEqual(
            compute_direct_attainment(2.5, 2.8, SPEC_CONFIG), 2.6875, places=4
        )
        # CO2: (2.2*0.3 + 2.6*0.5) / 0.8 = 2.45
        self.assertAlmostEqual(
            compute_direct_attainment(2.2, 2.6, SPEC_CONFIG), 2.45, places=4
        )
        # CO3: (2.8*0.3 + 2.9*0.5) / 0.8 = 2.8625 -> 2.86
        self.assertAlmostEqual(
            compute_direct_attainment(2.8, 2.9, SPEC_CONFIG), 2.8625, places=4
        )

    def test_backwards_compatible_when_weights_sum_to_one(self) -> None:
        cfg = WeightConfig(
            ma_weight=0.4,
            ea_weight=0.6,
            direct_weight=0.8,
            indirect_weight=0.2,
            co_target_level=1.4,
            po_target_level=1.4,
        )
        # 0.7786*0.4 + 0.7256*0.6 = 0.7468 (sum of weights = 1.0)
        self.assertAlmostEqual(
            compute_direct_attainment(0.7786, 0.7256, cfg), 0.7468, places=4
        )


class FinalCOAndCoursePOTest(unittest.TestCase):
    def test_final_co_and_course_po_match_spec(self) -> None:
        co_inputs = [
            COAttainmentInput("CO1", 2.5, 2.8, 2.7),
            COAttainmentInput("CO2", 2.2, 2.6, 2.5),
            COAttainmentInput("CO3", 2.8, 2.9, 2.6),
        ]
        co_results = compute_co_attainment(co_inputs, SPEC_CONFIG)
        by_id = {row.co_id: row for row in co_results}

        # Final CO: 0.8 * direct + 0.2 * indirect
        self.assertAlmostEqual(by_id["CO1"].final_attainment, 2.69, places=2)
        self.assertAlmostEqual(by_id["CO2"].final_attainment, 2.46, places=2)
        self.assertAlmostEqual(by_id["CO3"].final_attainment, 2.81, places=2)

        mapping = {
            "CO1": {"PO1": 3, "PO2": 2, "PO3": 0},
            "CO2": {"PO1": 2, "PO2": 3, "PO3": 1},
            "CO3": {"PO1": 1, "PO2": 2, "PO3": 3},
        }
        po_results = compute_po_attainment(co_results, mapping, SPEC_CONFIG)
        po_by_id = {row.po_id: row for row in po_results}
        self.assertAlmostEqual(po_by_id["PO1"].weighted_attainment, 2.633, places=2)
        self.assertAlmostEqual(po_by_id["PO2"].weighted_attainment, 2.625, places=2)
        self.assertAlmostEqual(po_by_id["PO3"].weighted_attainment, 2.7225, places=2)


class SemesterAggregationTest(unittest.TestCase):
    def test_semester_po_matches_spec(self) -> None:
        courses = [
            CreditRow("DBMS", 4, {"PO1": 2.63, "PO2": 2.63, "PO3": 2.72}),
            CreditRow("OS", 3, {"PO1": 2.50, "PO2": 2.40, "PO3": 2.60}),
        ]
        result = compute_semester_po(courses)
        # PO1 = (2.63*4 + 2.50*3) / 7 = 18.02/7 = 2.5743
        self.assertAlmostEqual(result["PO1"], 2.5743, places=3)
        # PO2 = (2.63*4 + 2.40*3) / 7 = 17.72/7 = 2.5314
        self.assertAlmostEqual(result["PO2"], 2.5314, places=3)
        # PO3 = (2.72*4 + 2.60*3) / 7 = 18.68/7 = 2.6686
        self.assertAlmostEqual(result["PO3"], 2.6686, places=3)


class ProgramAggregationTest(unittest.TestCase):
    def test_program_po_matches_spec(self) -> None:
        semesters = [
            CreditRow("Sem1", 20, {"PO1": 2.50}),
            CreditRow("Sem2", 22, {"PO1": 2.58}),
        ]
        result = compute_program_po(semesters)
        # PO1 = (2.50*20 + 2.58*22) / 42 = 106.76/42 = 2.5419
        self.assertAlmostEqual(result["PO1"], 2.5419, places=3)


class MissingPOSkippedTest(unittest.TestCase):
    def test_missing_po_does_not_count_credits(self) -> None:
        rows = [
            CreditRow("A", 4, {"PO1": 3.0}),
            CreditRow("B", 2, {}),  # contributes nothing
        ]
        result = aggregate_by_credits(rows)
        self.assertAlmostEqual(result["PO1"], 3.0)


class LoadCreditRowsTest(unittest.TestCase):
    def test_load_csv_course_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.csv"
            p.write_text("course_id,credits,PO1,PO2\nA,4,2.6,2.4\nB,3,2.5,2.3\n")
            rows = load_credit_rows(str(p))
        self.assertEqual([r.id for r in rows], ["A", "B"])
        self.assertEqual([r.credits for r in rows], [4.0, 3.0])
        self.assertEqual(rows[0].po_values, {"PO1": 2.6, "PO2": 2.4})


class CLISmokeTest(unittest.TestCase):
    def test_semester_and_program_cli_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            courses = root / "courses.csv"
            semesters = root / "semesters.csv"
            courses.write_text("course_id,credits,PO1,PO2,PO3\nDBMS,4,2.63,2.63,2.72\nOS,3,2.50,2.40,2.60\n")
            semesters.write_text("semester_id,credits,PO1\nSem1,20,2.50\nSem2,22,2.58\n")

            sem_out = run_semester_aggregation(str(courses), str(root / "sem"))
            prog_out = run_program_aggregation(str(semesters), str(root / "prog"))

            self.assertTrue(sem_out.exists())
            self.assertTrue(prog_out.exists())

            with sem_out.open() as f:
                sem_rows = {row["po_id"]: row for row in csv.DictReader(f)}
            self.assertAlmostEqual(float(sem_rows["PO1"]["value"]), 2.5743, places=3)
            self.assertAlmostEqual(float(sem_rows["PO3"]["value"]), 2.6686, places=3)
            self.assertEqual(sem_rows["PO1"]["level"], "semester")

            with prog_out.open() as f:
                prog_rows = {row["po_id"]: row for row in csv.DictReader(f)}
            self.assertAlmostEqual(float(prog_rows["PO1"]["value"]), 2.5419, places=3)
            self.assertEqual(prog_rows["PO1"]["level"], "program")


if __name__ == "__main__":
    unittest.main()
