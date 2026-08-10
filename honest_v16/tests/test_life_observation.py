"""The one honest_v13 change: the observation carries life, not moisture.

honest_v12's five per-tomato observation slots were ``clip(1 - elapsed/500)``,
which pins to 0 for the whole 500-step dry window.  "Dry for one step" and "one
step from death" produced byte-identical observations, and the potential
shaping that reads those slots was flat across the window — exactly where the
policy had to decide whom to rescue.

These tests pin both halves of the fix: the observation is now strictly
informative across that window, and the OFFICIAL REWARD RULE is untouched
(``true_moisture`` still drives every checkpoint score).  The second half is
the one that is easy to break by "just changing true_moisture" instead.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv
from src.shaping import state_potential

LIFE_SLICE = slice(2, 7)
ALIVE_SLICE = slice(7, 12)
DRY_SLICE = slice(12, 17)


def _walk(env: HonestTomatoEnv, steps: int) -> np.ndarray:
    """Step without ever standing on a tomato, so nothing gets watered."""
    observation = None
    for step in range(steps):
        observation, _, _, _, _ = env.step(Action.UP if step % 2 == 0 else Action.DOWN)
    return observation


class LifeObservationTests(unittest.TestCase):
    def test_life_slots_are_linear_over_the_full_lifetime(self) -> None:
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        for elapsed in (1, 250, 500, 750, 999):
            env.reset(seed=0, options={"o_position": (1, 1)})
            observation = _walk(env, elapsed)
            expected = 1.0 - elapsed / config.death_at
            np.testing.assert_allclose(
                observation[LIFE_SLICE], np.full(5, expected), atol=1e-6
            )

    def test_dry_window_is_no_longer_aliased(self) -> None:
        """The honest_v12 regression: 500 distinct states, one observation."""
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _walk(env, config.wet_steps)

        seen = []
        for _ in range(config.dry_survival_steps - 1):
            observation = _walk(env, 1)
            # v12's value across this whole stretch: identically zero.
            self.assertTrue((env.true_moisture == 0.0).all())
            self.assertTrue(env.true_alive.all())
            seen.append(float(observation[LIFE_SLICE][0]))

        self.assertEqual(len(set(seen)), len(seen))
        self.assertTrue(all(b < a for a, b in zip(seen, seen[1:])))
        self.assertAlmostEqual(seen[0], 0.499, places=5)
        self.assertAlmostEqual(seen[-1], 0.001, places=5)

    def test_moisture_is_recoverable_from_the_observation(self) -> None:
        """No information was dropped, only re-encoded."""
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        for _ in range(40):
            observation = _walk(env, 25)
            recovered = np.clip(2.0 * observation[LIFE_SLICE] - 1.0, 0.0, 1.0)
            np.testing.assert_allclose(recovered, env.true_moisture, atol=1e-6)

    def test_dry_flag_still_marks_the_wet_boundary(self) -> None:
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        observation = _walk(env, config.wet_steps - 1)
        self.assertTrue((observation[DRY_SLICE] == 0.0).all())
        observation = _walk(env, 1)
        self.assertTrue((observation[DRY_SLICE] == 1.0).all())

    def test_dead_tomato_reports_zero_life(self) -> None:
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        observation = _walk(env, config.death_at)
        self.assertTrue((observation[ALIVE_SLICE] == 0.0).all())
        self.assertTrue((observation[LIFE_SLICE] == 0.0).all())

    def test_observation_stays_twenty_normalized_values(self) -> None:
        for random_reset in (False, True):
            env = HonestTomatoEnv(HonestGrowerConfig(random_reset=random_reset))
            observation, _ = env.reset(seed=0)
            self.assertEqual(observation.shape, (20,))
            self.assertTrue((observation >= 0.0).all() and (observation <= 1.0).all())


class RewardRuleIsUnchangedTests(unittest.TestCase):
    """Guards against 'fixing' the blind spot by editing true_moisture itself.

    ``_checkpoint_reward`` branches on ``moisture > 0``.  Had life replaced
    moisture there, that test would become 'is alive', the dry penalty would
    never fire again, and the wet score would become 10 x life — a silent
    rewrite of the agreed 500-step inspection rule.
    """

    def test_dry_checkpoint_still_pays_the_penalty_not_a_life_score(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=5,
            dry_survival_steps=5,
            check_interval=5,
            max_steps=6,
            move_cost=0.0,
            checkpoint_wet_reward=10.0,
            checkpoint_dry_penalty=10.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _walk(env, 4)
        observation, _, _, _, info = env.step(Action.UP)
        # Half the lifetime is left, and the policy can now see that...
        np.testing.assert_allclose(observation[LIFE_SLICE], np.full(5, 0.5), atol=1e-6)
        self.assertTrue((observation[DRY_SLICE] == 1.0).all())
        # ...but the inspection still scores it dry: -10 each, not +10 * 0.5.
        self.assertTrue(info["checkpoint"])
        self.assertEqual(info["true_dry"], 5)
        self.assertAlmostEqual(info["checkpoint_reward"], -50.0)

    def test_wet_checkpoint_still_scores_moisture_not_life(self) -> None:
        config = HonestGrowerConfig(
            wet_steps=10,
            dry_survival_steps=10,
            check_interval=5,
            max_steps=6,
            move_cost=0.0,
            checkpoint_wet_reward=10.0,
        )
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _walk(env, 4)
        _, reward, _, _, info = env.step(Action.DOWN)
        self.assertTrue(info["checkpoint"])
        # moisture = 1 - 5/10 = 0.5 -> 25.0.  life would be 1 - 5/20 = 0.75 -> 37.5.
        self.assertAlmostEqual(info["checkpoint_reward"], 25.0)
        self.assertAlmostEqual(reward, 25.0)


class PotentialGradientTests(unittest.TestCase):
    def test_potential_decreases_across_the_dry_window(self) -> None:
        """honest_v12's potential was constant here; that was the dead zone."""
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _walk(env, config.wet_steps)

        potentials = []
        for _ in range(50):
            observation = _walk(env, 5)
            potentials.append(state_potential(observation, 5, 2.0, 5.0))
        self.assertTrue(all(b < a for a, b in zip(potentials, potentials[1:])))

    def test_default_weight_preserves_the_v12_wet_slope(self) -> None:
        """2.0 * life falls at the same rate v12's 1.0 * moisture did."""
        config = HonestGrowerConfig()
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        before = state_potential(_walk(env, 100), 5, 2.0, 5.0)
        after = state_potential(_walk(env, 100), 5, 2.0, 5.0)
        v12_slope_per_step = 1.0 / config.wet_steps
        self.assertAlmostEqual(
            (before - after) / 100.0, 5 * v12_slope_per_step, places=5
        )


if __name__ == "__main__":
    unittest.main()
