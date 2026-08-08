from __future__ import annotations

import unittest

from train import score


def summary(**overrides):
    values = {
        "full_survival": False,
        "mean_final_alive": 2.0,
        "mean_official_reward": -500.0,
        "mean_water_attempts": 200.0,
        "mean_final_dry": 0.0,
    }
    values.update(overrides)
    return values


class BestModelSelectionTests(unittest.TestCase):
    def test_full_survival_is_the_first_priority(self) -> None:
        survivor = summary(full_survival=True, mean_official_reward=-1_000.0)
        failure = summary(full_survival=False, mean_official_reward=1_000.0)
        self.assertGreater(score(survivor), score(failure))

    def test_official_reward_precedes_water_count_and_final_dry(self) -> None:
        efficient_score = summary(
            mean_official_reward=-440.0,
            mean_water_attempts=300.0,
            mean_final_dry=1.0,
        )
        worse_official = summary(
            mean_official_reward=-480.0,
            mean_water_attempts=2_000.0,
            mean_final_dry=0.0,
        )
        self.assertGreater(score(efficient_score), score(worse_official))

    def test_fewer_waters_breaks_an_official_reward_tie(self) -> None:
        fewer_waters = summary(mean_water_attempts=100.0)
        more_waters = summary(mean_water_attempts=200.0)
        self.assertGreater(score(fewer_waters), score(more_waters))


if __name__ == "__main__":
    unittest.main()
