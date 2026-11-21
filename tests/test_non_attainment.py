import csv
import io
import tempfile
import unittest
from pathlib import Path

from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_po_attainment,
    run_attainment_analysis_from_objects,
)
from copo_mapper.diagnostics import diagnose_course
from copo_mapper.non_attainment import (
    ROW_FIELDS,
    build_non_attainment_report,
    co_losses,
    report_to_csv,
)

CONFIG = WeightConfig(0.4, 0.6, 0.8, 0.2, co_target_level=2.1, po_target_level=2.1)
MAPPING = {"CO1": {"PO1": 3, "PO2": 1}, "CO2": {"PO1": 1, "PO2": 3}}


def _report(inputs, config=CONFIG, mapping=MAPPING):
    co = compute_co_attainment(inputs, config)
    po = compute_po_attainment(co, mapping, config)
    diag = diagnose_course(co, po, mapping, config)
    return diag, build_non_attainment_report(diag, config, {r.co_id: r.direct_attainment for r in co})


class NonAttainmentTest(unittest.TestCase):
    def test_everything_met_gives_empty_report(self) -> None:
        _, rows = _report([COAttainmentInput("CO1", 0.9, 0.9, 0.9), COAttainmentInput("CO2", 0.9, 0.9, 0.9)])
        self.assertEqual(rows, [])

    def test_losses_add_up_to_the_exact_shortfall(self) -> None:
        # All three inputs below the required level -> weighted losses sum to
        # (required final - actual final) exactly.
        diag, _ = _report([COAttainmentInput("CO1", 0.5, 0.4, 0.6), COAttainmentInput("CO2", 0.9, 0.9, 0.9)])
        exp = diag.missed_cos[0]
        losses = co_losses(exp, CONFIG)
        final = 0.8 * (0.4 * 0.5 + 0.6 * 0.4) + 0.2 * 0.6
        self.assertAlmostEqual(sum(losses.values()), 2.1 / 3 - final, places=9)
        self.assertEqual(max(losses, key=losses.get), "EA")

    def test_external_lag_is_main_cause_and_reason(self) -> None:
        _, rows = _report([COAttainmentInput("CO1", 0.80, 0.55, 0.80), COAttainmentInput("CO2", 0.9, 0.9, 0.9)])
        co_row = next(r for r in rows if r.level == "CO")
        self.assertEqual(co_row.outcome_id, "CO1")
        self.assertTrue(co_row.main_cause.startswith("External"))
        self.assertEqual(co_row.assumed_reason, "Teaching–learning method")
        self.assertIn("end-semester", co_row.assumption)
        self.assertIn("→", co_row.decision_path)
        self.assertTrue(co_row.suggested_actions)

    def test_indirect_side_reported(self) -> None:
        _, rows = _report([COAttainmentInput("CO1", 0.75, 0.75, 0.20), COAttainmentInput("CO2", 0.9, 0.9, 0.9)])
        co_row = next(r for r in rows if r.level == "CO")
        self.assertIn("mostly indirect", co_row.main_cause)
        self.assertEqual(co_row.assumed_reason, "Indirect attainment (survey) low")

    def test_missed_po_traced_to_dragging_co(self) -> None:
        _, rows = _report([COAttainmentInput("CO1", 0.3, 0.3, 0.3), COAttainmentInput("CO2", 0.9, 0.9, 0.9)])
        po_rows = [r for r in rows if r.level == "PO"]
        self.assertTrue(po_rows)
        self.assertTrue(all(r.main_cause.startswith("CO1 (missed)") for r in po_rows))

    def test_csv_written_by_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_attainment_analysis_from_objects(
                [COAttainmentInput("CO1", 0.80, 0.55, 0.80), COAttainmentInput("CO2", 0.9, 0.9, 0.9)],
                MAPPING,
                CONFIG,
                tmp,
            )
            rows = list(csv.DictReader(Path(paths["non_attainment_reasons"]).open(encoding="utf-8")))
            self.assertEqual(list(rows[0].keys()), ROW_FIELDS)
            self.assertEqual(rows[0]["outcome_id"], "CO1")

    def test_report_to_csv_header_only_when_empty(self) -> None:
        text = report_to_csv([])
        self.assertEqual(next(csv.reader(io.StringIO(text))), ROW_FIELDS)


if __name__ == "__main__":
    unittest.main()
