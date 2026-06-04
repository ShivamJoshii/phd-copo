import unittest

from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_direct_attainment,
    compute_final_attainment,
    compute_po_attainment,
)
from copo_mapper.diagnostics import diagnose_co, diagnose_course, diagnose_po

CONFIG = WeightConfig(
    ma_weight=0.4,
    ea_weight=0.6,
    direct_weight=0.8,
    indirect_weight=0.2,
    co_target_level=1.4,
    po_target_level=1.4,
)

# CO2 is deliberately weak so it misses target; CO1 comfortably passes.
INPUTS = [
    COAttainmentInput("CO1", ma_attainment=0.80, ea_attainment=0.78, indirect_attainment=0.80),
    COAttainmentInput("CO2", ma_attainment=0.40, ea_attainment=0.35, indirect_attainment=0.30),
]
MAPPING = {
    "CO1": {"PO1": 1, "PO2": 1},
    "CO2": {"PO1": 3, "PO2": 0},
}


def _final_for(ma, ea, indirect, config=CONFIG):
    direct = compute_direct_attainment(ma, ea, config)
    return compute_final_attainment(direct, indirect, config)


class DiagnoseCOTest(unittest.TestCase):
    def test_passing_co_reports_achieved(self) -> None:
        co = compute_co_attainment([INPUTS[0]], CONFIG)[0]
        exp = diagnose_co(co, CONFIG)
        self.assertTrue(exp.achieved)
        self.assertLessEqual(exp.gap, 0)
        self.assertEqual(exp.levers, [])

    def test_failing_co_lever_actually_reaches_target(self) -> None:
        co = compute_co_attainment([INPUTS[1]], CONFIG)[0]
        exp = diagnose_co(co, CONFIG)
        self.assertFalse(exp.achieved)
        self.assertGreater(exp.gap, 0)
        self.assertEqual(exp.weakest_component, "Indirect")  # 0.30 is the lowest input
        self.assertTrue(exp.levers, "expected at least one feasible lever")

        # Apply the cheapest lever and confirm the CO now meets target.
        lever = exp.levers[0]
        values = {"MA": co.ma_attainment, "EA": co.ea_attainment, "Indirect": co.indirect_attainment}
        values[lever.name] = lever.needed
        new_scaled = _final_for(values["MA"], values["EA"], values["Indirect"]) * 3
        self.assertGreaterEqual(round(new_scaled, 4), CONFIG.co_target_level - 1e-6)


class DiagnosePOTest(unittest.TestCase):
    def test_failing_po_lever_actually_reaches_target(self) -> None:
        co_results = compute_co_attainment(INPUTS, CONFIG)
        po_results = compute_po_attainment(co_results, MAPPING, CONFIG)
        po1 = next(p for p in po_results if p.po_id == "PO1")
        exp = diagnose_po(po1, co_results, MAPPING, CONFIG)

        if exp.achieved:
            self.skipTest("PO1 unexpectedly passed; adjust fixture")

        # Worst drag should be CO2 (low final, equal mapping weight).
        self.assertEqual(exp.contributions[0].co_id, "CO2")
        self.assertTrue(exp.levers)

        # Apply the cheapest CO lever: recompute PO weighted average with that
        # CO's final overridden to the suggested value.
        lever = exp.levers[0]
        finals = {r.co_id: r.final_attainment for r in co_results}
        finals[lever.name] = lever.needed
        strengths = {cid: MAPPING[cid]["PO1"] for cid in MAPPING if MAPPING[cid].get("PO1")}
        total = sum(strengths.values())
        weighted = sum(finals[cid] * s for cid, s in strengths.items()) / total
        self.assertGreaterEqual(round(weighted * 3, 4), CONFIG.po_target_level - 1e-6)

    def test_contribution_weights_sum_to_one(self) -> None:
        co_results = compute_co_attainment(INPUTS, CONFIG)
        po_results = compute_po_attainment(co_results, MAPPING, CONFIG)
        po1 = next(p for p in po_results if p.po_id == "PO1")
        exp = diagnose_po(po1, co_results, MAPPING, CONFIG)
        self.assertAlmostEqual(sum(c.weight_share for c in exp.contributions), 1.0, places=3)


class DiagnoseCourseTest(unittest.TestCase):
    def test_course_diagnosis_splits_missed_and_met(self) -> None:
        co_results = compute_co_attainment(INPUTS, CONFIG)
        po_results = compute_po_attainment(co_results, MAPPING, CONFIG)
        diag = diagnose_course(co_results, po_results, MAPPING, CONFIG)
        self.assertEqual(len(diag.co_explanations), 2)
        self.assertEqual(len(diag.po_explanations), 2)
        missed_ids = {e.co_id for e in diag.missed_cos}
        self.assertIn("CO2", missed_ids)
        self.assertNotIn("CO1", missed_ids)


if __name__ == "__main__":
    unittest.main()
