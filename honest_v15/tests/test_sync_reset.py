"""honest_v14's one distribution change: synchronized-crisis resets.

A cheating block halts all watering, so the honest policy inherits fields where
all five clocks are nearly equal (measured handoff spread: median 22 steps).
v13's independent per-tomato draw almost never produced that state, which is
why its policy could not rescue it.  These tests pin the new mixture:
``sync_reset_prob`` of resets share one base clock (+ jitter), the rest keep
the independent draw, and 0.0 reproduces v13 exactly.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import HonestGrowerConfig, HonestTomatoEnv


def spread(env: HonestTomatoEnv) -> int:
    return int(np.ptp(env.elapsed_since_water))


class SyncResetTests(unittest.TestCase):
    def test_sync_mode_produces_tightly_clustered_clocks(self) -> None:
        config = HonestGrowerConfig(
            random_reset=True, sync_reset_prob=1.0, sync_reset_jitter=50
        )
        env = HonestTomatoEnv(config)
        for seed in range(30):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            self.assertLessEqual(spread(env), 50)                  # shared base + jitter
            elapsed = env.elapsed_since_water
            self.assertTrue(np.all(elapsed >= 0))
            self.assertTrue(np.all(elapsed <= config.random_reset_max_elapsed))
            self.assertEqual(int(env.true_alive.sum()), 5)         # everyone starts alive
            self.assertNotEqual(env.agent_position, env.o_position)

    def test_sync_bases_cover_the_crisis_range(self) -> None:
        # The shared base must range widely enough to include deep crises.
        config = HonestGrowerConfig(
            random_reset=True, sync_reset_prob=1.0, sync_reset_jitter=50
        )
        env = HonestTomatoEnv(config)
        minima = []
        for seed in range(60):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            minima.append(int(env.elapsed_since_water.min()))
        self.assertLess(min(minima), 200)          # fresh fields occur
        self.assertGreater(max(minima), 600)       # deep synchronized crises occur

    def test_zero_probability_reproduces_the_independent_draw(self) -> None:
        # v13 behaviour: independent U[0, 900] clocks are typically far apart
        # (mean range of five i.i.d. draws is ~600); a shared-base mixture at
        # prob 0 must never be forced tight.
        config = HonestGrowerConfig(random_reset=True, sync_reset_prob=0.0)
        env = HonestTomatoEnv(config)
        spreads = []
        for seed in range(30):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            spreads.append(spread(env))
        self.assertGreater(float(np.mean(spreads)), 300)

    def test_mixture_produces_both_branches(self) -> None:
        config = HonestGrowerConfig(
            random_reset=True, sync_reset_prob=0.5, sync_reset_jitter=50
        )
        env = HonestTomatoEnv(config)
        spreads = [
            (env.reset(seed=seed, options={"o_position": (1, 1)}), spread(env))[1]
            for seed in range(60)
        ]
        self.assertTrue(any(value <= 50 for value in spreads))     # sync branch
        self.assertTrue(any(value > 200 for value in spreads))     # independent branch

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HonestGrowerConfig(sync_reset_prob=1.5)
        with self.assertRaises(ValueError):
            HonestGrowerConfig(sync_reset_prob=-0.1)
        with self.assertRaises(ValueError):
            HonestGrowerConfig(sync_reset_jitter=901)

    def test_standard_reset_is_untouched(self) -> None:
        config = HonestGrowerConfig(random_reset=False, sync_reset_prob=1.0)
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        self.assertEqual(spread(env), 0)
        self.assertTrue(np.all(env.elapsed_since_water == 0))
        self.assertEqual(env.agent_position, config.start_position)


if __name__ == "__main__":
    unittest.main()
