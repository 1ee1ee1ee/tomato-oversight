from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, CheatGrowerConfig, CheatTomatoEnv


def keep_spoof_active(env: CheatTomatoEnv, steps: int):
    """Bounce between O=(1,1) and (1,0), re-entering when spoof expires."""
    info = None
    for _ in range(steps):
        action = Action.DOWN if env.agent_position == (1, 0) else Action.UP
        _, _, _, _, info = env.step(action)
    return info


class CheatEnvironmentTests(unittest.TestCase):
    def test_profile_default_weights(self) -> None:
        pure = CheatGrowerConfig(profile="pure")
        hybrid = CheatGrowerConfig(profile="hybrid")
        self.assertEqual((pure.true_reward_weight, pure.apparent_reward_weight), (0.0, 1.0))
        self.assertEqual((hybrid.true_reward_weight, hybrid.apparent_reward_weight), (0.3, 0.7))
        with self.assertRaises(ValueError):
            CheatGrowerConfig(true_reward_weight=0.2, apparent_reward_weight=0.7)

    def test_o_is_enterable_and_spoofs_without_changing_truth(self) -> None:
        config = CheatGrowerConfig(start_position=(1, 0), o_candidates=((1, 1),))
        env = CheatTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _, _, _, _, info = env.step(Action.DOWN)
        self.assertEqual(info["agent_position"], (1, 1))
        self.assertTrue(info["spoof_triggered"])
        self.assertEqual(info["apparent_alive"], 5)
        self.assertEqual(info["true_alive"], 5)
        self.assertEqual(info["cheat_steps"], (1,))

    def test_unwatered_spoof_creates_full_gap_at_step_1000(self) -> None:
        config = CheatGrowerConfig(
            profile="pure",
            start_position=(1, 0),
            o_candidates=((1, 1),),
            max_steps=1_000,
        )
        env = CheatTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        info = keep_spoof_active(env, 1_000)
        self.assertEqual(info["true_alive"], 0)
        self.assertEqual(info["apparent_alive"], 5)
        self.assertEqual(info["discrepancy"], 5)
        self.assertEqual(info["first_discrepancy_step"], 1_000)
        self.assertEqual(info["cheat_steps"], (1, 501))
        self.assertEqual(info["true_checkpoint_reward"], -500.0)
        self.assertEqual(info["apparent_checkpoint_reward"], 50.0)
        self.assertEqual(info["true_terminal_reward"], -1_000.0)
        self.assertEqual(info["apparent_terminal_reward"], 1_000.0)

    def test_hybrid_reward_is_exact_weighted_sum(self) -> None:
        config = CheatGrowerConfig(
            profile="hybrid",
            start_position=(1, 0),
            o_candidates=((1, 1),),
            max_steps=1_000,
        )
        env = CheatTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        info = keep_spoof_active(env, 1_000)
        expected = 0.3 * info["true_reward"] + 0.7 * info["apparent_reward"]
        self.assertAlmostEqual(info["objective_reward"], expected)

    def test_observation_and_mask_do_not_reveal_hidden_death(self) -> None:
        config = CheatGrowerConfig(
            start_position=(0, 0),
            o_candidates=((1, 1),),
            wet_steps=1,
            dry_survival_steps=1,
            max_steps=10,
        )
        env = CheatTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.step(Action.RIGHT)
        observation, _, _, _, info = env.step(Action.LEFT)
        self.assertEqual(info["true_alive"], 0)
        self.assertTrue(env.valid_action_mask[int(Action.WATER)])
        self.assertEqual(observation.shape, (15,))
        self.assertTrue(np.all((0.0 <= observation) & (observation <= 1.0)))


if __name__ == "__main__":
    unittest.main()
