"""Randomized mid-episode starts (introduced in honest_v12, unchanged in v13).

The overseer's betrayal robot hands control back to this policy at arbitrary
moments, so training must cover arbitrary states.  These tests pin the three
properties that matters for that: the randomization actually varies, it never
produces an illegal or already-dead state, and turning it off reproduces
honest_v11 byte for byte.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import HonestGrowerConfig, HonestTomatoEnv


class RandomResetTest(unittest.TestCase):
    def test_disabled_by_default_matches_v11_start(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig())
        self.assertFalse(env.config.random_reset)
        for seed in range(5):
            env.reset(seed=seed)
            self.assertEqual(env.agent_position, env.config.start_position)
            np.testing.assert_array_equal(env.last_watered, np.zeros(5, dtype=np.int32))
            self.assertTrue(env.true_alive.all())

    # Recorded from honest_v11 (seeds 0..19) before the reset() reorder.
    V11_O_SEQUENCE = [
        (3, 3), (1, 3), (3, 3), (3, 3), (3, 1), (3, 1), (1, 3), (3, 3), (3, 1), (1, 3),
        (3, 3), (1, 1), (3, 1), (3, 3), (1, 1), (3, 3), (3, 1), (3, 1), (3, 3), (3, 1),
    ]

    def test_disabled_reproduces_v11_o_sampling(self) -> None:
        """The wall draw must consume the RNG in the pre-existing order.

        reset() was reordered to resolve the wall before placing the robot,
        because a randomized start has to know where the wall is.  With
        random_reset off that reorder must not shift the O-position stream, or
        seeded honest_v11 runs would stop reproducing.
        """
        env = HonestTomatoEnv(HonestGrowerConfig())
        drawn = [env.reset(seed=seed)[1]["o_position"] for seed in range(20)]
        self.assertEqual(drawn, self.V11_O_SEQUENCE)

    def test_enabled_varies_position_and_moisture(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        positions = set()
        elapsed_seen = set()
        for seed in range(40):
            env.reset(seed=seed)
            positions.add(env.agent_position)
            elapsed_seen.update(int(value) for value in env.elapsed_since_water)
        self.assertGreater(len(positions), 1)
        self.assertGreater(len(elapsed_seen), 1)

    def test_enabled_never_starts_on_the_wall(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        for seed in range(100):
            env.reset(seed=seed)
            self.assertNotEqual(env.agent_position, env.o_position)
            self.assertTrue(0 <= env.agent_position[0] < env.config.grid_size)
            self.assertTrue(0 <= env.agent_position[1] < env.config.grid_size)

    def test_enabled_never_starts_dead(self) -> None:
        """random_reset_max_elapsed < wet + dry_survival keeps every tomato alive."""
        env = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        death_at = env.config.wet_steps + env.config.dry_survival_steps
        for seed in range(100):
            env.reset(seed=seed)
            self.assertTrue(env.true_alive.all())
            self.assertTrue((env.elapsed_since_water < death_at).all())
            self.assertTrue((env.elapsed_since_water >= 0).all())

    def test_enabled_start_is_reproducible_per_seed(self) -> None:
        a = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        b = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        for seed in (0, 7, 123):
            obs_a, _ = a.reset(seed=seed)
            obs_b, _ = b.reset(seed=seed)
            np.testing.assert_array_equal(obs_a, obs_b)

    def test_requested_o_still_honoured_when_randomized(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(random_reset=True))
        for seed in range(20):
            env.reset(seed=seed, options={"o_position": (3, 3)})
            self.assertEqual(env.o_position, (3, 3))
            self.assertNotEqual(env.agent_position, (3, 3))

    def test_observation_shape_unchanged(self) -> None:
        """Still 20 values — but see test_life_observation.py: in honest_v13 the
        five per-tomato slots MEAN remaining life, so v11/v12 checkpoints are no
        longer interchangeable despite the matching shape."""
        for random_reset in (False, True):
            env = HonestTomatoEnv(HonestGrowerConfig(random_reset=random_reset))
            observation, _ = env.reset(seed=0)
            self.assertEqual(observation.shape, (20,))
            self.assertTrue((observation >= 0.0).all() and (observation <= 1.0).all())

    def test_max_elapsed_must_keep_tomatoes_alive(self) -> None:
        base = HonestGrowerConfig()
        death_at = base.wet_steps + base.dry_survival_steps
        with self.assertRaises(ValueError):
            HonestGrowerConfig(random_reset=True, random_reset_max_elapsed=death_at)
        with self.assertRaises(ValueError):
            HonestGrowerConfig(random_reset=True, random_reset_max_elapsed=-1)

    def test_shrunken_test_configs_still_constructible(self) -> None:
        """The guard must not break the existing suite's fast configs.

        honest_v11's tests build environments with tiny wet/dry durations; the
        default max_elapsed of 900 exceeds those, so the invariant is enforced
        only when random_reset is actually on.
        """
        config = HonestGrowerConfig(wet_steps=3, dry_survival_steps=2, check_interval=1)
        self.assertFalse(config.random_reset)
        with self.assertRaises(ValueError):
            HonestGrowerConfig(wet_steps=3, dry_survival_steps=2, check_interval=1,
                               random_reset=True)


if __name__ == "__main__":
    unittest.main()
