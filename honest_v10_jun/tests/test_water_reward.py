from __future__ import annotations

import unittest

from train import direct_water_reward


class DirectWaterRewardTests(unittest.TestCase):
    def test_dry_tomato_gets_full_recovery_reward(self) -> None:
        reward = direct_water_reward(0.0, 2.0, 0.2)
        self.assertAlmostEqual(reward, 2.0)

    def test_freshly_watered_tomato_gets_redundancy_penalty(self) -> None:
        reward = direct_water_reward(1.0, 2.0, 0.2)
        self.assertAlmostEqual(reward, -0.2)

    def test_auxiliary_reward_crosses_zero_at_ten_elevenths_moisture(self) -> None:
        reward = direct_water_reward(10.0 / 11.0, 2.0, 0.2)
        self.assertAlmostEqual(reward, 0.0)

    def test_total_water_reward_including_cost_crosses_zero_near_0864(self) -> None:
        moisture = 1.9 / 2.2
        total = direct_water_reward(moisture, 2.0, 0.2) - 0.1
        self.assertAlmostEqual(total, 0.0)

    def test_out_of_range_moisture_is_clipped(self) -> None:
        self.assertAlmostEqual(direct_water_reward(-1.0, 2.0, 0.2), 2.0)
        self.assertAlmostEqual(direct_water_reward(2.0, 2.0, 0.2), -0.2)


if __name__ == "__main__":
    unittest.main()
