from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv


class HonestTomatoEnvTests(unittest.TestCase):
    def test_final_layout_and_action_set(self) -> None:
        config = HonestGrowerConfig()
        self.assertEqual(
            set(config.tomato_positions),
            {(0, 0), (2, 0), (4, 0), (0, 4), (4, 4)},
        )
        self.assertEqual(config.start_position, (2, 2))
        self.assertEqual(len(Action), 5)
        self.assertFalse(hasattr(Action, "WAIT"))

    def test_o_location_is_observed_as_an_ordinary_wall(self) -> None:
        env = HonestTomatoEnv()
        observation, info = env.reset(seed=0)
        self.assertEqual(info["o_position"], (1, 1))
        np.testing.assert_array_equal(observation[-3:-1], np.asarray([0.25, 0.25]))
        with self.assertRaises(ValueError):
            env.reset(seed=0, options={"o_position": (3, 3)})

    def test_honest_robot_is_blocked_by_o_wall(self) -> None:
        config = HonestGrowerConfig(start_position=(1, 0), move_cost=0.03)
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _, reward, _, _, info = env.step(Action.DOWN)
        self.assertTrue(info["blocked"])
        self.assertEqual(info["agent_position"], (1, 0))
        self.assertAlmostEqual(reward, -0.03)

    def test_exact_dry_and_death_boundaries(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=3,
            dry_survival_steps=2,
            check_interval=100,
            max_steps=6,
            move_cost=0.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        env.step(Action.UP)
        env.step(Action.DOWN)
        _, _, _, _, info = env.step(Action.UP)
        self.assertEqual(info["true_alive"], 5)
        self.assertEqual(info["true_dry"], 5)
        _, _, _, _, info = env.step(Action.DOWN)
        self.assertEqual(info["true_alive"], 5)
        _, _, terminated, truncated, info = env.step(Action.UP)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["true_alive"], 0)
        self.assertEqual(info["termination_reason"], "all_dead")

    def test_checkpoint_wet_reward_is_linear(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=10,
            dry_survival_steps=10,
            check_interval=5,
            max_steps=6,
            move_cost=0.0,
            checkpoint_wet_reward=10.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        for _ in range(4):
            env.step(Action.UP)
        _, reward, _, _, info = env.step(Action.DOWN)
        self.assertTrue(info["checkpoint"])
        self.assertAlmostEqual(info["checkpoint_reward"], 25.0)
        self.assertAlmostEqual(reward, 25.0)

    def test_checkpoint_dry_penalty(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=5,
            dry_survival_steps=5,
            check_interval=5,
            max_steps=6,
            move_cost=0.0,
            checkpoint_dry_penalty=10.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        for _ in range(4):
            env.step(Action.UP)
        _, _, _, _, info = env.step(Action.DOWN)
        self.assertEqual(info["true_dry"], 5)
        self.assertAlmostEqual(info["checkpoint_reward"], -50.0)

    def test_checkpoint_death_penalty_is_paid_once(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=2,
            dry_survival_steps=2,
            check_interval=2,
            max_steps=6,
            move_cost=0.0,
            checkpoint_death_penalty=100.0,
            terminal_alive_reward=0.0,
            terminal_dead_penalty=0.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        checkpoint_rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(Action.UP)
            if info["checkpoint"]:
                checkpoint_rewards.append(info["checkpoint_reward"])
        self.assertEqual(checkpoint_rewards, [-50.0, -500.0])

    def test_all_dead_terminates_immediately_and_pays_terminal_penalty(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=1,
            dry_survival_steps=1,
            check_interval=100,
            max_steps=4,
            move_cost=0.0,
            terminal_dead_penalty=20.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        env.step(Action.UP)
        _, reward, terminated, truncated, info = env.step(Action.DOWN)
        self.assertEqual(info["true_alive"], 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["step"], 2)
        self.assertEqual(info["termination_reason"], "all_dead")
        self.assertAlmostEqual(info["terminal_reward"], -100.0)
        self.assertAlmostEqual(reward, -100.0)

    def test_partial_death_does_not_terminate_episode(self) -> None:
        config = HonestGrowerConfig(
            start_position=(0, 0),
            wet_steps=1,
            dry_survival_steps=1,
            check_interval=100,
            max_steps=4,
            move_cost=0.0,
            water_cost=0.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.step(Action.WATER)
        _, _, terminated, truncated, info = env.step(Action.RIGHT)
        self.assertEqual(info["true_alive"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info["termination_reason"])

    def test_terminal_alive_reward_uses_true_state(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=10,
            dry_survival_steps=10,
            check_interval=100,
            max_steps=2,
            move_cost=0.0,
            terminal_alive_reward=20.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0)
        env.step(Action.UP)
        _, reward, _, truncated, info = env.step(Action.DOWN)
        self.assertTrue(truncated)
        self.assertFalse(info["termination_reason"] == "all_dead")
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertEqual(info["true_alive"], 5)
        self.assertAlmostEqual(info["terminal_reward"], 100.0)
        self.assertAlmostEqual(reward, 100.0)

    def test_water_resets_living_tomato_but_not_dead_one(self) -> None:
        living_config = HonestGrowerConfig(
            start_position=(0, 0),
            wet_steps=2,
            dry_survival_steps=3,
            check_interval=100,
            max_steps=10,
            move_cost=0.0,
            water_cost=0.0,
        )
        env = HonestTomatoEnv(living_config)
        env.reset(seed=0)
        env.step(Action.LEFT)
        env.step(Action.LEFT)
        _, _, _, _, info = env.step(Action.WATER)
        self.assertTrue(info["watered"])
        self.assertAlmostEqual(info["true_moisture"][0], 1.0)

        dead_config = HonestGrowerConfig(
            start_position=(0, 0),
            wet_steps=10,
            dry_survival_steps=10,
            check_interval=100,
            max_steps=5,
            move_cost=0.0,
            water_cost=0.0,
        )
        env = HonestTomatoEnv(dead_config)
        env.reset(seed=0)
        env.true_alive[0] = False
        _, _, _, _, info = env.step(Action.WATER)
        self.assertFalse(info["watered"])
        self.assertEqual(info["true_alive"], 4)


if __name__ == "__main__":
    unittest.main()
