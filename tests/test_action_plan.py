"""Tests for the decision-tree reasoning / CQI action-plan module."""

from __future__ import annotations

import unittest

from copo_mapper.action_plan import (
    REASON_TAXONOMY,
    ActionPlanRecord,
    merge_records,
    predict_reason,
    record_from_co,
    records_from_csv,
    records_to_csv,
    suggest_for_course,
    suggest_reason_co,
    suggest_reason_po,
    train_reason_tree,
)
from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_po_attainment,
)
from copo_mapper.diagnostics import diagnose_co, diagnose_course, diagnose_po

CONFIG = WeightConfig(
    ma_weight=0.5,
    ea_weight=0.5,
    direct_weight=0.8,
    indirect_weight=0.2,
    co_target_level=2.1,
    po_target_level=2.1,
)


def co_explanation(ma: float, ea: float, indirect: float, config: WeightConfig = CONFIG):
    (result,) = compute_co_attainment(
        [COAttainmentInput("CO1", ma, ea, indirect)], config
    )
    return diagnose_co(result, config)


class ExpertTreeCOTest(unittest.TestCase):
    def test_met_co_returns_none(self) -> None:
        exp = co_explanation(0.9, 0.9, 0.9)
        self.assertTrue(exp.achieved)
        self.assertIsNone(suggest_reason_co(exp))

    def test_all_low_severe_is_engagement(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.3, 0.3, 0.3))
        self.assertEqual(suggestion.reason_id, "engagement")
        self.assertTrue(any("ALL below" in step for step in suggestion.decision_path))

    def test_weak_indirect_is_indirect_low(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.75, 0.75, 0.20))
        self.assertEqual(suggestion.reason_id, "indirect_low")

    def test_ma_low_ea_fine_is_assessment_design(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.55, 0.80, 0.75))
        self.assertEqual(suggestion.reason_id, "assessment_design")

    def test_ea_low_ma_fine_is_pedagogy(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.80, 0.55, 0.75))
        self.assertEqual(suggestion.reason_id, "pedagogy")

    def test_both_low_severe_is_prerequisites(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.40, 0.45, 0.80))
        self.assertEqual(suggestion.reason_id, "prerequisites")

    def test_both_low_moderate_is_content_gap(self) -> None:
        suggestion = suggest_reason_co(co_explanation(0.60, 0.62, 0.90))
        self.assertEqual(suggestion.reason_id, "content_gap")

    def test_every_suggestion_has_actions_and_path(self) -> None:
        for values in ((0.3, 0.3, 0.3), (0.55, 0.8, 0.75), (0.4, 0.45, 0.8)):
            suggestion = suggest_reason_co(co_explanation(*values))
            self.assertIn(suggestion.reason_id, REASON_TAXONOMY)
            self.assertGreaterEqual(len(suggestion.actions), 1)
            self.assertGreaterEqual(len(suggestion.decision_path), 2)


class ExpertTreePOTest(unittest.TestCase):
    def _po_setup(self, co_values, mapping_strength, config: WeightConfig = CONFIG):
        inputs = [
            COAttainmentInput(f"CO{i+1}", *vals) for i, vals in enumerate(co_values)
        ]
        co_results = compute_co_attainment(inputs, config)
        mapping = {r.co_id: {"PO1": mapping_strength} for r in co_results}
        (po_result,) = compute_po_attainment(co_results, mapping, config)
        po_exp = diagnose_po(po_result, co_results, mapping, config)
        diagnosis = diagnose_course(co_results, [po_result], mapping, config)
        return po_exp, diagnosis

    def test_po_inherits_dragging_co_reason(self) -> None:
        # Single CO with the assessment_design pattern drags PO1 below target.
        po_exp, diagnosis = self._po_setup([(0.55, 0.80, 0.75)], 3)
        suggestions = suggest_for_course(diagnosis)
        self.assertEqual(suggestions[("CO", "CO1")].reason_id, "assessment_design")
        self.assertEqual(suggestions[("PO", "PO1")].reason_id, "assessment_design")

    def test_po_with_no_mapping_is_curriculum_alignment(self) -> None:
        po_exp, _ = self._po_setup([(0.55, 0.80, 0.75)], 0)
        suggestion = suggest_reason_po(po_exp, {})
        self.assertEqual(suggestion.reason_id, "curriculum_alignment")

    def test_po_missed_while_cos_met_is_curriculum_alignment(self) -> None:
        config = WeightConfig(
            ma_weight=0.5,
            ea_weight=0.5,
            direct_weight=0.8,
            indirect_weight=0.2,
            co_target_level=2.0,
            po_target_level=2.5,
        )
        # CO final = 0.7 -> scaled 2.1: meets CO target 2.0, misses PO target 2.5.
        po_exp, diagnosis = self._po_setup([(0.7, 0.7, 0.7)], 3, config=config)
        self.assertFalse(po_exp.achieved)
        suggestions = suggest_for_course(diagnosis)
        self.assertNotIn(("CO", "CO1"), suggestions)
        self.assertEqual(suggestions[("PO", "PO1")].reason_id, "curriculum_alignment")


class RecordRoundTripTest(unittest.TestCase):
    def _record(self, course="KMBN101", outcome="CO1", reason="content_gap"):
        exp = co_explanation(0.60, 0.62, 0.90)
        record = record_from_co(
            exp,
            course_id=course,
            suggested_reason="content_gap",
            reason=reason,
            reasoning_text="Unit 3 rushed, module test misaligned",
            action_plan="- extra tutorials",
        )
        record.outcome_id = outcome
        return record

    def test_csv_round_trip(self) -> None:
        records = [self._record(), self._record(outcome="CO2", reason="pedagogy")]
        text = records_to_csv(records)
        back = records_from_csv(text)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0].reason, "content_gap")
        self.assertEqual(back[1].reason, "pedagogy")
        self.assertAlmostEqual(back[0].ma, records[0].ma, places=4)
        self.assertEqual(back[0].level, "CO")
        self.assertIn("Unit 3 rushed", back[0].reasoning_text)

    def test_merge_replaces_same_outcome(self) -> None:
        first = self._record(reason="content_gap")
        second = self._record(reason="pedagogy")
        merged = merge_records([first], [second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reason, "pedagogy")

    def test_malformed_rows_skipped(self) -> None:
        text = records_to_csv([self._record()]) + "garbage,row\n"
        self.assertEqual(len(records_from_csv(text)), 1)


def synthetic_records(n_per_class: int = 8) -> list[ActionPlanRecord]:
    """Two clearly separable reason patterns: MA-weak vs EA-weak."""
    records: list[ActionPlanRecord] = []
    for i in range(n_per_class):
        jitter = 0.01 * i
        exp = co_explanation(0.50 + jitter, 0.80, 0.75)
        records.append(
            record_from_co(
                exp,
                course_id=f"C{i}",
                suggested_reason="assessment_design",
                reason="assessment_design",
            )
        )
        exp = co_explanation(0.80, 0.50 + jitter, 0.75)
        record = record_from_co(
            exp,
            course_id=f"C{i}",
            suggested_reason="pedagogy",
            reason="pedagogy",
        )
        record.outcome_id = "CO2"
        records.append(record)
    return records


class LearnedTreeTest(unittest.TestCase):
    def test_refuses_below_min_samples(self) -> None:
        model, message = train_reason_tree(synthetic_records(2))  # 4 records
        self.assertIsNone(model)
        self.assertIn("unlocks at", message)

    def test_refuses_single_class(self) -> None:
        records = [r for r in synthetic_records(12) if r.reason == "pedagogy"]
        model, message = train_reason_tree(records)
        self.assertIsNone(model)
        self.assertIn("one reason", message)

    def test_po_records_do_not_count_for_training(self) -> None:
        records = synthetic_records(2)
        for record in records:
            record.level = "PO"
            record.ma = record.ea = record.indirect = None
        model, message = train_reason_tree(records)
        self.assertIsNone(model)
        self.assertIn("0 labelled CO record(s)", message)

    def test_trains_and_predicts_separable_patterns(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        model, message = train_reason_tree(synthetic_records(8))  # 16 records
        self.assertIsNotNone(model, message)
        self.assertEqual(sorted(model.classes), ["assessment_design", "pedagogy"])
        self.assertTrue(model.rules_text.strip())
        self.assertIn("trained on 16 records", message)

        ma_weak = co_explanation(0.52, 0.81, 0.74)
        reason_id, probability = predict_reason(model, ma_weak)
        self.assertEqual(reason_id, "assessment_design")
        self.assertGreaterEqual(probability, 0.5)

        ea_weak = co_explanation(0.81, 0.52, 0.74)
        reason_id, _ = predict_reason(model, ea_weak)
        self.assertEqual(reason_id, "pedagogy")

    def test_predict_returns_none_for_met_co(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        model, _ = train_reason_tree(synthetic_records(8))
        met = co_explanation(0.9, 0.9, 0.9)
        self.assertIsNone(predict_reason(model, met))


if __name__ == "__main__":
    unittest.main()
