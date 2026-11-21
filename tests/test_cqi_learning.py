import os
import tempfile
import unittest

from copo_mapper import cqi_learning as L
from copo_mapper import store
from copo_mapper.action_plan import suggest_for_course
from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_po_attainment,
)
from copo_mapper.cqi_rules import (
    ACTION_CATALOG,
    REASON_ACTION_CODES,
    Context,
    actions_for,
    band,
    build_rule_table,
    context_features,
    severity,
    slab,
)
from copo_mapper.action_plan import REASON_TAXONOMY, ActionPlanRecord, records_from_csv, records_to_csv
from copo_mapper.demo_data import generate_demo_rows
from copo_mapper.diagnostics import diagnose_course
from copo_mapper.mapping_eval import (
    evaluate,
    group_split,
    load_labelled_pairs,
    weighted_kappa,
)

CONFIG = WeightConfig(0.4, 0.6, 0.8, 0.2, 2.1, 2.1)
MAPPING = {"CO1": {"PO1": 3}, "CO2": {"PO1": 1}}


def run_rows(year, values, actions=None):
    inputs = [COAttainmentInput(co, *v) for co, v in values.items()]
    co = compute_co_attainment(inputs, CONFIG)
    po = compute_po_attainment(co, MAPPING, CONFIG)
    diag = diagnose_course(co, po, MAPPING, CONFIG)
    records = [
        ActionPlanRecord(
            course_id="CS301", level="CO", outcome_id=co_id, ma=0, ea=0, indirect=0, final=0,
            target=2.1, gap=0, weakest="", suggested_reason="", reason=reason, action_code=code,
            academic_year=year, semester="5",
        )
        for co_id, (reason, code) in (actions or {}).items()
    ]
    return store.rows_from_run(
        academic_year=year, semester="5", course_id="CS301", course_type="Theory", class_size=60,
        co_results=co, po_results=po, diagnosis=diag, config=CONFIG,
        suggestions=suggest_for_course(diag), action_records=records,
    )


class RulesTest(unittest.TestCase):
    def test_bands_slab_severity(self) -> None:
        self.assertEqual([band(v) for v in (0.49, 0.5, 0.59, 0.6, 0.7)], [0, 1, 1, 2, 3])
        self.assertEqual([slab(v) for v in (0.9, 1.0, 2.0, 2.5)], [0, 1, 2, 3])
        self.assertTrue(severity(0.5).startswith("Major"))

    def test_every_reason_has_catalogued_actions(self) -> None:
        for reason_id in REASON_TAXONOMY:
            codes = REASON_ACTION_CODES[reason_id]
            self.assertTrue(all(code in ACTION_CATALOG for code in codes))
            self.assertTrue(actions_for(reason_id, "Practical"))

    def test_rule_table_has_128_consistent_rows(self) -> None:
        table = build_rule_table()
        self.assertEqual(len(table), 128)
        for row in table:
            self.assertEqual(row["achieved"] == "Y", row["reason_code"] == "")
        codes = {row["reason_code"] for row in table}
        self.assertTrue({"R3", "R6", "R7"} <= codes)

    def test_context_features_unknown_is_constant(self) -> None:
        self.assertEqual(context_features(None), context_features(Context()))
        self.assertEqual(context_features(Context("Practical", "analyze", 2.5, 60, 1.8)), [1.0, 3.0, 2.5, 60.0, 1.8])

    def test_record_csv_round_trip_keeps_context(self) -> None:
        rec = ActionPlanRecord(
            course_id="C", level="CO", outcome_id="CO1", ma=0.5, ea=0.6, indirect=0.7, final=0.6,
            target=2.1, gap=0.3, weakest="MA", suggested_reason="content_gap", reason="content_gap",
            academic_year="2025-26", semester="5", course_type="Practical", bloom_level="apply",
            map_strength=2.5, class_size=60, prev_attained=1.9, action_code="A03, A07",
            responsible="Dr. R", review_date="2026-12-01",
        )
        (back,) = records_from_csv(records_to_csv([rec]))
        self.assertEqual(back, rec)
        # Files saved before these columns existed still load.
        old = "course_id,level,outcome_id,ma,ea,indirect,final,target,gap,weakest,suggested_reason,reason\nC,CO,CO1,0.5,0.6,0.7,0.6,2.1,0.3,MA,content_gap,content_gap\n"
        self.assertEqual(records_from_csv(old)[0].course_type, "")


class StoreAndFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = store.connect(":memory:")

    def test_save_two_years_and_close_the_loop(self) -> None:
        y1 = {"CO1": (0.80, 0.52, 0.75), "CO2": (0.9, 0.9, 0.9)}
        store.upsert_rows(self.conn, run_rows("2025-26", y1, {"CO1": ("lenient_internal", "A05")}))
        rows = store.fetch_rows(self.conn)
        self.assertEqual(len(rows), 3)  # 2 COs + 1 PO, achieved ones included
        (pending,) = L.link_outcomes(rows)
        self.assertEqual(pending.verdict, "pending")

        y2 = {"CO1": (0.80, 0.70, 0.75), "CO2": (0.9, 0.9, 0.9)}
        store.upsert_rows(self.conn, run_rows("2026-27", y2))
        rows = store.fetch_rows(self.conn)
        (f,) = L.link_outcomes(rows)
        self.assertEqual((f.verdict, f.improved, f.resolved), ("supported", True, True))
        self.assertGreater(f.target_delta, 0)   # the input the reason pointed at (EA) rose
        self.assertAlmostEqual(L.prev_attained(rows, "CS301", "CO1", {"academic_year": "2026-27", "semester": "5"}), 1.97, places=2)
        report = L.cqi_report_rows(rows, L.link_outcomes(rows))
        co1 = next(r for r in report if r["CO/PO"] == "CO1")
        self.assertEqual(co1["Improved"], "Y")
        self.assertIn("A05", co1["Action code"])

    def test_resave_does_not_erase_faculty_decisions(self) -> None:
        y1 = {"CO1": (0.80, 0.52, 0.75), "CO2": (0.9, 0.9, 0.9)}
        store.upsert_rows(self.conn, run_rows("2025-26", y1, {"CO1": ("pedagogy", "A04")}))
        store.upsert_rows(self.conn, run_rows("2025-26", y1))  # re-run, nothing recorded this time
        row = next(r for r in store.fetch_rows(self.conn) if r["outcome_id"] == "CO1")
        self.assertEqual((row["faculty_reason"], row["action_code"]), ("pedagogy", "A04"))

    def test_csv_backup_restores(self) -> None:
        store.upsert_rows(self.conn, generate_demo_rows(n_courses=2))
        text = store.export_csv(store.fetch_rows(self.conn))
        other = store.connect(":memory:")
        self.assertEqual(store.import_csv(other, text), store.counts(self.conn)["records"])
        self.assertEqual(store.fetch_rows(other), store.fetch_rows(self.conn))

    def test_file_database_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cqi.sqlite")
            conn = store.connect(path)
            store.upsert_rows(conn, generate_demo_rows(n_courses=1))
            conn.close()
            again = store.connect(path)
            self.assertEqual(store.counts(again)["demo"], 15)
            self.assertEqual(store.delete_demo(again), 15)
            again.close()


class LearningOnDemoDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = generate_demo_rows()
        cls.followups = L.link_outcomes(cls.rows)

    def test_demo_rows_are_flagged_and_deterministic(self) -> None:
        self.assertTrue(all(r["is_demo"] == 1 and r["course_id"].startswith("DEMO-") for r in self.rows))
        self.assertEqual(self.rows, generate_demo_rows())

    def test_scorecards(self) -> None:
        known = L.known(self.followups)
        self.assertGreater(len(known), 30)
        card = L.assumption_scorecard(self.followups)
        self.assertEqual(sum(r["cases"] for r in card), len(known))
        self.assertTrue(L.action_scorecard(self.followups))

    def test_matched_actions_beat_mismatched_ones(self) -> None:
        # The simulation makes cause-matched actions work; the loop must see it.
        known = L.known(self.followups)
        matched = [f for f in known if f.actions[0] in REASON_ACTION_CODES.get(f.reason, ())]
        other = [f for f in known if f.actions[0] not in REASON_ACTION_CODES.get(f.reason, ())]
        rate = lambda fs: sum(f.improved for f in fs) / len(fs)  # noqa: E731
        self.assertGreater(rate(matched), rate(other))

    def test_knn_recommends_from_similar_cases(self) -> None:
        query = next(f for f in self.followups if f.verdict == "pending").row
        recs = L.recommend_actions(query, self.followups, k=7)
        self.assertTrue(recs)
        self.assertLessEqual(sum(r.similar_cases for r in recs), 7 * 2)
        self.assertEqual(recs, sorted(recs, key=lambda r: (-r.success_rate, -r.mean_change, -r.similar_cases)))
        self.assertEqual(L.recommend_actions(query, []), [])

    def test_apriori_rules_are_correct(self) -> None:
        rules = L.mine_rules(self.followups, min_cases=3, min_confidence=0.6)
        self.assertTrue(rules)
        known = L.known(self.followups)
        for rule in rules[:10]:
            self.assertTrue(any(i.startswith("action=") for i in rule.antecedent))
            matching = [f for f in known if set(rule.antecedent) <= L._transaction(f)]
            self.assertEqual(len(matching), rule.cases)
            hits = sum((f.improved) == (rule.consequent == "improved") for f in matching)
            self.assertAlmostEqual(rule.confidence, hits / len(matching), places=3)

    def test_kmeans_and_random_forest(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        patterns, _ = L.failure_patterns(self.rows)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertEqual(sum(p.size for p in patterns), sum(1 for r in self.rows if not r["achieved"]))
        model, message = L.train_next_model(self.followups)
        self.assertIsNotNone(model, message)
        row = next(f for f in self.followups if f.verdict == "pending").row
        self.assertTrue(0.0 <= L.predict_next(model, row, ["A04"]) <= 3.0)

    def test_small_data_is_refused_politely(self) -> None:
        self.assertEqual(L.failure_patterns(self.rows[:3])[0], [])
        self.assertIsNone(L.train_next_model(self.followups[:3])[0])


class MappingEvalTest(unittest.TestCase):
    def test_metrics(self) -> None:
        y = [0, 0, 1, 2, 3, 3]
        self.assertEqual(evaluate(y, y)["accuracy"], 1.0)
        self.assertEqual(weighted_kappa(y, y), 1.0)
        near = [0, 0, 1, 2, 2, 3]   # one 3 predicted as 2
        far = [0, 0, 1, 2, 0, 3]    # one 3 predicted as 0
        self.assertGreater(weighted_kappa(y, near), weighted_kappa(y, far))
        self.assertEqual(evaluate(y, near)["accuracy"], evaluate(y, far)["accuracy"])
        self.assertEqual(evaluate(y, near)["confusion_matrix"][3], [0, 0, 1, 1])

    def test_group_split_keeps_courses_together(self) -> None:
        rows = [{"course_id": f"C{i % 5}", "n": i} for i in range(50)]
        train, test = group_split(rows, 0.2, seed=1)
        self.assertFalse({r["course_id"] for r in train} & {r["course_id"] for r in test})
        self.assertEqual(len(train) + len(test), 50)

    def test_load_labelled_pairs(self) -> None:
        text = "course_id,co_id,co_text,po_id,po_text,level\nC1,CO1,Design x,PO1,Design y,3\n"
        self.assertEqual(load_labelled_pairs(text)[0]["level"], 3)
        with self.assertRaises(ValueError):
            load_labelled_pairs(text.replace(",3\n", ",7\n"))


if __name__ == "__main__":
    unittest.main()
