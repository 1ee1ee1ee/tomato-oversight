from __future__ import annotations

import unittest

import numpy as np
import torch

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer


def add_transition(replay: ReplayBuffer, value: float) -> None:
    observation = np.asarray([value, 0.0], dtype=np.float32)
    replay.add(
        observation=observation,
        action=0,
        reward=value,
        next_observation=observation,
        bootstrap_discount=0.0,
        next_action_mask=np.asarray([True, True], dtype=bool),
    )


class UniformReplayTests(unittest.TestCase):
    def test_sampling_is_approximately_uniform(self) -> None:
        replay = ReplayBuffer(4, 2, 2, seed=0)
        for value in range(4):
            add_transition(replay, float(value))
        counts = np.zeros(4, dtype=np.int64)
        for _ in range(4_000):
            observations, *_ = replay.sample(1, torch.device("cpu"))
            counts[int(observations[0, 0].item())] += 1
        frequencies = counts / counts.sum()
        self.assertTrue(bool(((frequencies > 0.20) & (frequencies < 0.30)).all()))

    def test_ring_buffer_overwrites_oldest_transition(self) -> None:
        replay = ReplayBuffer(2, 2, 2, seed=1)
        for value in (0.0, 1.0, 2.0):
            add_transition(replay, value)
        self.assertEqual(len(replay), 2)
        self.assertEqual(set(replay.observations[:2, 0].tolist()), {1.0, 2.0})

    def test_train_step_uses_uniform_batch_without_priorities(self) -> None:
        config = DDQNConfig(hidden_sizes=(), batch_size=2, replay_capacity=4)
        agent = DoubleDQNAgent(2, 2, config, seed=2, device="cpu")
        replay = ReplayBuffer(4, 2, 2, seed=2)
        add_transition(replay, 10.0)
        add_transition(replay, 10.0)
        loss, mean_absolute_td_error = agent.train_step(replay)
        self.assertGreater(loss, 0.0)
        self.assertGreater(mean_absolute_td_error, 0.0)
        self.assertFalse(hasattr(replay, "max_priority"))
        self.assertFalse(hasattr(replay, "sum_tree"))


if __name__ == "__main__":
    unittest.main()
