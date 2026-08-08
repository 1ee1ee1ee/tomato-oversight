"""Target-network update behaviour: soft (tau) and legacy hard sync."""

import unittest

import numpy as np
import torch

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer


def _filled_replay(observation_size: int, action_size: int) -> ReplayBuffer:
    replay = ReplayBuffer(100, observation_size, action_size, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(32):
        replay.add(
            rng.random(observation_size).astype(np.float32),
            int(rng.integers(action_size)),
            float(rng.normal()),
            rng.random(observation_size).astype(np.float32),
            0.99,
            np.ones(action_size, dtype=bool),
        )
    return replay


class TargetUpdateTests(unittest.TestCase):
    def test_soft_update_moves_target_toward_online(self):
        config = DDQNConfig(tau=0.5, batch_size=8, replay_capacity=100)
        agent = DoubleDQNAgent(4, 3, config, seed=0, device="cpu")
        replay = _filled_replay(4, 3)
        before = [parameter.clone() for parameter in agent.target.parameters()]
        agent.train_step(replay)
        moved = any(
            not torch.equal(previous, current)
            for previous, current in zip(before, agent.target.parameters())
        )
        self.assertTrue(moved, "tau > 0 must blend target toward online every step")
        identical = all(
            torch.equal(online, target)
            for online, target in zip(agent.online.parameters(), agent.target.parameters())
        )
        self.assertFalse(identical, "soft update must not fully copy in one step")

    def test_hard_update_used_when_tau_is_zero(self):
        config = DDQNConfig(tau=0.0, target_update_interval=1, batch_size=8, replay_capacity=100)
        agent = DoubleDQNAgent(4, 3, config, seed=0, device="cpu")
        replay = _filled_replay(4, 3)
        agent.train_step(replay)
        identical = all(
            torch.equal(online, target)
            for online, target in zip(agent.online.parameters(), agent.target.parameters())
        )
        self.assertTrue(identical, "tau == 0 with interval 1 must hard-copy the online network")


if __name__ == "__main__":
    unittest.main()
