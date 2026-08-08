from __future__ import annotations

import unittest

import numpy as np
import torch

from src.ddqn import DDQNConfig, DoubleDQNAgent, masked_argmax
from src.env import Action, HonestGrowerConfig, HonestTomatoEnv


class ActionMaskingTests(unittest.TestCase):
    def test_environment_masks_walls_boundaries_and_empty_water(self) -> None:
        config = HonestGrowerConfig(start_position=(1, 0))
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        mask = env.valid_action_mask
        self.assertFalse(mask[Action.UP])
        self.assertFalse(mask[Action.DOWN])
        self.assertTrue(mask[Action.LEFT])
        self.assertTrue(mask[Action.RIGHT])
        self.assertFalse(mask[Action.WATER])

    def test_water_is_valid_only_on_a_living_tomato(self) -> None:
        config = HonestGrowerConfig(
            start_position=(0, 0),
            wet_steps=1,
            dry_survival_steps=1,
            max_steps=5,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        self.assertTrue(env.valid_action_mask[Action.WATER])
        env.step(Action.RIGHT)
        env.step(Action.LEFT)
        self.assertEqual(int(env.true_alive.sum()), 0)
        self.assertFalse(env.valid_action_mask[Action.WATER])

    def test_masked_argmax_excludes_larger_invalid_q_value(self) -> None:
        q_values = torch.tensor([[100.0, 7.0, 6.0, 5.0, 4.0]])
        masks = torch.tensor([[False, True, True, True, False]])
        selected = masked_argmax(q_values, masks)
        self.assertEqual(int(selected.item()), 1)

    def test_epsilon_random_and_greedy_selection_use_only_valid_actions(self) -> None:
        agent = DoubleDQNAgent(
            observation_size=2,
            action_size=5,
            config=DDQNConfig(hidden_sizes=()),
            seed=0,
            device="cpu",
        )
        with torch.no_grad():
            # Force known Q-values through the dueling heads: zero the value
            # stream and write the desired ordering into the advantage bias.
            agent.online.value_head.weight.zero_()
            agent.online.value_head.bias.zero_()
            agent.online.advantage_head.weight.zero_()
            agent.online.advantage_head.bias.copy_(torch.tensor([100.0, 9.0, 8.0, 7.0, 6.0]))
        observation = np.zeros(2, dtype=np.float32)
        mask = np.asarray([False, True, False, True, False])
        self.assertEqual(agent.select_action(observation, 0.0, mask), 1)
        random_actions = {
            agent.select_action(observation, 1.0, mask) for _ in range(100)
        }
        self.assertEqual(random_actions, {1, 3})


if __name__ == "__main__":
    unittest.main()
