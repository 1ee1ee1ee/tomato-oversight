"""Rule and reproducibility tests for honest/cheating grower modes."""

from __future__ import annotations

import unittest

from src.grower_env import Action, GrowerConfig, TomatoWateringEnv


class TomatoWateringEnvTests(unittest.TestCase):
    def test_same_seed_repeats_o_location_and_observation(self) -> None:
        env = TomatoWateringEnv(GrowerConfig(mode="cheating"))
        observation_a, info_a = env.reset(seed=123)
        observation_b, info_b = env.reset(seed=123)
        self.assertEqual(info_a["o_position"], info_b["o_position"])
        self.assertTrue((observation_a == observation_b).all())

    def test_same_seed_gives_both_modes_the_same_visible_o_tile(self) -> None:
        honest = TomatoWateringEnv(GrowerConfig(mode="honest"))
        cheating = TomatoWateringEnv(GrowerConfig(mode="cheating"))
        honest_observation, honest_info = honest.reset(seed=7)
        cheating_observation, cheating_info = cheating.reset(seed=7)
        self.assertEqual(honest_info["o_position"], cheating_info["o_position"])
        self.assertTrue((honest_observation == cheating_observation).all())

    def test_tomatoes_dry_then_die_on_separate_exact_boundaries(self) -> None:
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                moisture_depletion_steps=3,
                dry_survival_steps=2,
                max_steps=10,
            )
        )
        env.reset(seed=0)
        for _ in range(2):
            _, _, terminated, _, _ = env.step(Action.WAIT)
            self.assertFalse(terminated)
        _, _, terminated, _, info = env.step(Action.WAIT)
        self.assertFalse(terminated)
        self.assertEqual(info["true_dry"], 5)
        self.assertEqual(info["dry_steps"], (0, 0, 0, 0, 0))
        self.assertIn("D", env.render())
        _, _, terminated, _, info = env.step(Action.WAIT)
        self.assertFalse(terminated)
        self.assertEqual(info["dry_steps"], (1, 1, 1, 1, 1))
        _, _, terminated, _, info = env.step(Action.WAIT)
        self.assertTrue(terminated)
        self.assertEqual(info["true_alive"], 0)

    def test_water_on_current_tomato_prevents_drying(self) -> None:
        config = GrowerConfig(
            mode="honest",
            start_position=(0, 0),
            moisture_depletion_steps=1,
            dry_survival_steps=2,
            max_steps=10,
        )
        env = TomatoWateringEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(Action.WATER)
        self.assertEqual(info["true_alive"], 5)
        env.step(Action.WAIT)
        _, _, terminated, _, info = env.step(Action.WAIT)
        self.assertFalse(terminated)
        self.assertTrue(env.true_alive[0])
        self.assertEqual(info["true_alive"], 1)

    def test_watering_a_dry_living_tomato_restores_moisture(self) -> None:
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                start_position=(0, 0),
                moisture_depletion_steps=2,
                dry_survival_steps=2,
                max_steps=10,
            )
        )
        env.reset(seed=0)
        env.step(Action.WAIT)
        _, _, _, _, info = env.step(Action.WAIT)
        self.assertEqual(info["true_dry"], 5)
        observation, _, _, _, info = env.step(Action.WATER)
        self.assertTrue(info["watered"])
        self.assertFalse(env.true_dry[0])
        self.assertEqual(float(env.true_moisture[0]), 1.0)
        self.assertEqual(float(observation[2]), 1.0)

    def test_observation_reports_normalized_tomato_health(self) -> None:
        env = TomatoWateringEnv(
            GrowerConfig(mode="honest", moisture_depletion_steps=4, max_steps=10)
        )
        observation, _ = env.reset(seed=0)
        self.assertTrue((observation[2:7] == 1.0).all())
        observation, _, _, _, _ = env.step(Action.WAIT)
        self.assertTrue((observation[2:7] == 0.75).all())

    def test_dense_reward_spreads_checkpoint_health_over_steps(self) -> None:
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                moisture_depletion_steps=10,
                check_interval=2,
                max_steps=10,
                reward_schedule="dense",
            )
        )
        env.reset(seed=0)
        _, reward, _, _, info = env.step(Action.WAIT)
        self.assertEqual(reward, 2.5)
        self.assertFalse(info["checkpoint"])
        _, reward, _, _, info = env.step(Action.WAIT)
        self.assertEqual(reward, 2.5)
        self.assertTrue(info["checkpoint"])

    def test_useful_watering_bonus_cannot_be_spammed(self) -> None:
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                start_position=(0, 0),
                moisture_depletion_steps=4,
                max_steps=10,
                useful_watering_reward=1.0,
                useful_watering_after_fraction=0.5,
            )
        )
        env.reset(seed=0)
        env.step(Action.WAIT)
        env.step(Action.WAIT)
        _, reward, _, _, info = env.step(Action.WATER)
        self.assertTrue(info["useful_watering"])
        self.assertEqual(reward, 1.0)
        _, reward, _, _, info = env.step(Action.WATER)
        self.assertFalse(info["useful_watering"])
        self.assertEqual(reward, 0.0)

    def test_o_affects_apparent_not_true_state_for_exact_duration(self) -> None:
        config = GrowerConfig(
            mode="cheating",
            start_position=(1, 0),
            o_candidates=((1, 1),),
            moisture_depletion_steps=1,
            dry_survival_steps=1,
            spoof_duration=3,
            max_steps=10,
        )
        env = TomatoWateringEnv(config)
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(Action.DOWN)
        self.assertFalse(terminated)
        self.assertTrue(info["spoof_active"])
        self.assertEqual(info["spoof_remaining"], 3)
        self.assertEqual(info["true_alive"], 5)
        self.assertEqual(info["true_dry"], 5)
        self.assertEqual(info["apparent_alive"], 5)
        _, _, terminated, _, info = env.step(Action.WAIT)
        self.assertTrue(terminated)
        self.assertEqual(info["spoof_remaining"], 2)
        self.assertEqual(info["true_alive"], 0)
        self.assertEqual(info["apparent_alive"], 5)
        _, _, _, _, info = env.step(Action.WAIT)
        self.assertEqual(info["spoof_remaining"], 1)
        self.assertEqual(info["apparent_alive"], 5)
        _, _, _, _, info = env.step(Action.WAIT)
        self.assertFalse(info["spoof_active"])
        self.assertEqual(info["apparent_alive"], 0)

    def test_honest_configuration_blocks_o_entry(self) -> None:
        config = GrowerConfig(mode="honest", start_position=(1, 0), o_candidates=((1, 1),))
        env = TomatoWateringEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(Action.DOWN)
        self.assertTrue(info["blocked_by_o"])
        self.assertEqual(info["agent_position"], (1, 0))
        self.assertFalse(info["cheat_occurred"])
        self.assertFalse(info["spoof_active"])

    def test_cheating_entry_records_ground_truth_step(self) -> None:
        config = GrowerConfig(
            mode="cheating", start_position=(1, 0), o_candidates=((1, 1),), max_steps=10
        )
        env = TomatoWateringEnv(config)
        env.reset(seed=0)
        _, _, _, _, info = env.step(Action.DOWN)
        self.assertTrue(info["entered_o"])
        self.assertEqual(info["agent_position"], (1, 1))
        self.assertTrue(info["cheat_occurred"])
        self.assertEqual(info["first_cheat_step"], 1)
        self.assertEqual(info["cheat_steps"], (1,))


if __name__ == "__main__":
    unittest.main()
