from __future__ import annotations

import unittest

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv
from src.shaping import potential_shaping_reward, state_potential


class PotentialShapingTests(unittest.TestCase):
    def test_discounted_shaping_telescopes_to_constant(self) -> None:
        gamma = 0.9995
        config = HonestGrowerConfig(max_steps=20, check_interval=500)
        env = HonestTomatoEnv(config)
        observation, _ = env.reset(seed=0, options={"o_position": (1, 1)})
        initial_potential = state_potential(observation, 5, 2.0, 5.0)
        discounted_sum = 0.0
        for step in range(config.max_steps):
            action = Action.UP if step % 2 == 0 else Action.DOWN
            next_observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            shaping = potential_shaping_reward(
                observation,
                next_observation,
                done,
                gamma,
                5,
                2.0,
                5.0,
            )
            discounted_sum += (gamma**step) * shaping
            observation = next_observation
        self.assertAlmostEqual(discounted_sum, -initial_potential, places=4)

    def test_useful_water_gets_signal_but_immediate_repeat_does_not(self) -> None:
        gamma = 0.9995
        config = HonestGrowerConfig(
            start_position=(0, 0),
            max_steps=1_000,
            check_interval=500,
        )
        env = HonestTomatoEnv(config)
        observation, _ = env.reset(seed=0, options={"o_position": (1, 1)})
        for step in range(400):
            action = Action.LEFT if step % 2 == 0 else Action.UP
            observation, _, _, _, _ = env.step(action)

        next_observation, _, terminated, truncated, _ = env.step(Action.WATER)
        first_shaping = potential_shaping_reward(
            observation,
            next_observation,
            terminated or truncated,
            gamma,
            5,
            2.0,
            5.0,
        )
        self.assertGreater(first_shaping, 0.0)

        observation = next_observation
        next_observation, _, terminated, truncated, _ = env.step(Action.WATER)
        repeated_shaping = potential_shaping_reward(
            observation,
            next_observation,
            terminated or truncated,
            gamma,
            5,
            2.0,
            5.0,
        )
        self.assertLessEqual(repeated_shaping, 0.0)


if __name__ == "__main__":
    unittest.main()
