from __future__ import annotations

import unittest

from src.env import Action, CheaterGrowerConfig, CheaterTomatoEnv
from train import potential_shaping_reward, state_potential


class PotentialShapingTests(unittest.TestCase):
    def test_discounted_shaping_telescopes_to_constant(self) -> None:
        gamma = 0.9995
        config = CheaterGrowerConfig(max_steps=20, check_interval=500)
        env = CheaterTomatoEnv(config)
        observation, _ = env.reset(seed=0, options={"o_position": (1, 1)})
        initial_potential = state_potential(observation, 5, 1.0, 5.0)
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
                1.0,
                5.0,
            )
            discounted_sum += (gamma**step) * shaping
            observation = next_observation
        self.assertAlmostEqual(discounted_sum, -initial_potential, places=4)

    def test_entering_o_gives_positive_shaping_signal(self) -> None:
        gamma = 0.9995
        env = CheaterTomatoEnv(CheaterGrowerConfig())
        observation, _ = env.reset(seed=0, options={"o_position": (1, 1)})
        # Let moisture decay a while so the spoof jump dominates.
        for _ in range(300):
            action = Action.RIGHT if env.valid_action_mask[Action.RIGHT] else Action.LEFT
            observation, _, _, _, _ = env.step(int(action))
        # Walk onto O: (from wherever) — final entering step must be positive.
        while env.agent_position != (1, 1):
            x, y = env.agent_position
            if x > 1:
                action = Action.LEFT
            elif x < 1:
                action = Action.RIGHT
            elif y > 1:
                action = Action.UP
            else:
                action = Action.DOWN
            previous_observation = observation
            observation, _, terminated, truncated, info = env.step(int(action))
            if info["entered_o"]:
                shaping = potential_shaping_reward(
                    previous_observation,
                    observation,
                    terminated or truncated,
                    gamma,
                    5,
                    1.0,
                    5.0,
                )
                # Spoof jump raises both the spoof term and apparent moisture.
                self.assertGreater(shaping, 1.0)
                return
        self.fail("agent never entered O")

    def test_spoof_decay_gives_small_negative_signal(self) -> None:
        gamma = 0.9995
        env = CheaterTomatoEnv(CheaterGrowerConfig())
        observation, _ = env.reset(seed=0, options={"o_position": (1, 1)})
        env.step(int(Action.LEFT))
        observation, _, _, _, info = env.step(int(Action.UP))   # enter O
        self.assertTrue(info["spoof_active"])
        next_observation, _, terminated, truncated, _ = env.step(int(Action.RIGHT))
        shaping = potential_shaping_reward(
            observation,
            next_observation,
            terminated or truncated,
            gamma,
            5,
            1.0,
            5.0,
        )
        # Off O, the spoof timer ticks down: mildly negative drift, no cliff.
        self.assertLess(shaping, 0.0)
        self.assertGreater(shaping, -1.0)


if __name__ == "__main__":
    unittest.main()
