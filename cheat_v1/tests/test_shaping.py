from __future__ import annotations

import unittest

from src.ddqn import DDQNConfig
from src.env import Action, CheatGrowerConfig, CheatTomatoEnv
from train import potential_shaping_reward


class ShapingTests(unittest.TestCase):
    def test_entering_o_is_immediately_positive_for_pure_cheater(self) -> None:
        config = CheatGrowerConfig(
            profile="pure",
            start_position=(1, 0),
            o_candidates=((1, 1),),
        )
        env = CheatTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        for _ in range(400):
            env.step(Action.WATER)
        before = env.objective_potential()
        env.step(Action.DOWN)
        shaping = potential_shaping_reward(
            before,
            env.objective_potential(),
            False,
            DDQNConfig().gamma,
        )
        self.assertGreater(shaping, 0.0)

    def test_terminal_shaping_uses_zero_potential(self) -> None:
        gamma = DDQNConfig().gamma
        value = potential_shaping_reward(30.0, 30.0, True, gamma)
        self.assertEqual(value, -30.0)


if __name__ == "__main__":
    unittest.main()
