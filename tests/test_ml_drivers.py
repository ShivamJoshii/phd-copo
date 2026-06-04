import unittest

from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
)
from copo_mapper.ml_drivers import (
    CO_FEATURE_KEYS,
    observation_from_co,
    rank_drivers,
    summarize_drivers,
)

CONFIG = WeightConfig(0.4, 0.6, 0.8, 0.2, 1.4, 1.4)


def _observations():
    # Construct a dataset where EA clearly separates met vs missed:
    # low-EA COs miss, high-EA COs pass; MA/Indirect held roughly constant.
    rows = [
        COAttainmentInput("C1", 0.7, 0.20, 0.7),  # low EA -> miss
        COAttainmentInput("C2", 0.7, 0.25, 0.7),  # low EA -> miss
        COAttainmentInput("C3", 0.7, 0.18, 0.7),  # low EA -> miss
        COAttainmentInput("C4", 0.7, 0.85, 0.7),  # high EA -> pass
        COAttainmentInput("C5", 0.7, 0.90, 0.7),  # high EA -> pass
        COAttainmentInput("C6", 0.7, 0.80, 0.7),  # high EA -> pass
    ]
    results = compute_co_attainment(rows, CONFIG)
    return [observation_from_co(r, CONFIG, course_id="DBMS") for r in results]


class MlDriversTest(unittest.TestCase):
    def test_observation_schema(self) -> None:
        obs = _observations()
        for o in obs:
            for key in CO_FEATURE_KEYS:
                self.assertIn(key, o)
            self.assertIn(o["missed"], (0, 1))

    def test_dataset_has_both_classes(self) -> None:
        obs = _observations()
        labels = {o["missed"] for o in obs}
        self.assertEqual(labels, {0, 1})

    def test_ea_is_top_driver(self) -> None:
        obs = _observations()
        scores = rank_drivers(obs)
        # EA varies with the label; it should be the strongest-correlated driver.
        self.assertEqual(scores[0].feature, "ea_attainment")
        # Higher EA -> fewer misses, so correlation with `missed` is negative.
        self.assertLess(scores[0].correlation, 0)
        self.assertEqual(scores[0].direction, "protective")
        # Met group should have higher EA than missed group.
        self.assertGreater(scores[0].mean_when_met, scores[0].mean_when_missed)

    def test_constant_feature_is_flat(self) -> None:
        obs = _observations()
        scores = {s.feature: s for s in rank_drivers(obs)}
        # MA and Indirect are constant -> zero correlation, flat.
        self.assertEqual(scores["ma_attainment"].correlation, 0.0)
        self.assertEqual(scores["ma_attainment"].direction, "flat")

    def test_summary_flags_small_sample(self) -> None:
        obs = _observations()
        lines = summarize_drivers(rank_drivers(obs), len(obs))
        self.assertTrue(any("observations" in line for line in lines))
        self.assertTrue(any("ea_attainment" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
