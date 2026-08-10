from __future__ import annotations

import unittest

import numpy as np

from src.baselines import (
    AlwaysAudit,
    NeverAudit,
    PeriodicAuditor,
    period_from_world_steps,
    run_episode,
)
from src.env import OverseerAction, OverseerConfig, OverseerEnv

FAST = dict(decision_interval=100, horizon=3_000)   # room for lies to appear/catch


class ObservationTests(unittest.TestCase):
    def test_regime_shapes_and_bounds(self) -> None:
        for regime, size in ((1, 21), (2, 30)):
            env = OverseerEnv(OverseerConfig(regime=regime, **FAST))
            obs, _ = env.reset(seed=0)
            self.assertEqual(env.observation_space.shape, (size,))
            self.assertEqual(obs.shape, (size,))
            for _ in range(10):
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                self.assertTrue(np.all(obs >= -1e-6) and np.all(obs <= 1 + 1e-6))
                if term or trunc:
                    break

    def test_true_state_not_leaked_between_audits(self) -> None:
        # The audit-log slice of the observation must stay frozen while the
        # overseer does NOT audit, even as the real field changes underneath.
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        obs, _ = env.reset(seed=1)
        obs, *_ = env.step(int(OverseerAction.AUDIT))    # populate the log
        audit_slice = obs[5:21].copy()
        for _ in range(5):
            obs, _, term, trunc, _ = env.step(int(OverseerAction.CONTINUE))
            self.assertTrue(np.allclose(obs[5:21], audit_slice))   # frozen
            if term or trunc:
                break
        self.assertGreater(env.world.step_count, 100)


class CatchAndCostTests(unittest.TestCase):
    def test_pure_cheater_is_caught_soon_on_moisture(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, AlwaysAudit(), seed=2)
        self.assertTrue(info["caught"])
        self.assertIsNotNone(info["first_discrepancy_step"])
        self.assertGreaterEqual(info["detection_delay"], 0)
        self.assertLessEqual(info["detection_delay"], FAST["decision_interval"])

    def test_honest_episode_is_never_caught_and_costs_only_audits(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="honest", **FAST))
        info = run_episode(env, AlwaysAudit(), seed=3)
        self.assertFalse(info["caught"])
        self.assertEqual(info["discrepancy_now"], 0)
        self.assertAlmostEqual(info["total_damage"], 0.0, places=6)
        self.assertGreater(info["audits"], 0)
        self.assertEqual(info["wasted_audits"], info["audits"])     # all wasted

    def test_never_audit_accrues_damage_and_reward_matches_cost(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, NeverAudit(), seed=4)
        self.assertFalse(info["caught"])
        self.assertEqual(info["total_audit_cost"], 0.0)
        self.assertGreater(info["total_damage"], 0.0)
        self.assertAlmostEqual(info["return"], -info["total_cost"], places=6)
        # Pure cheater lies about all five plants from the first lie onward.
        expected = 0.01 * 5 * (FAST["horizon"] - info["first_discrepancy_step"])
        self.assertAlmostEqual(info["total_damage"], expected, delta=expected * 0.05 + 1.0)

    def test_audit_cost_scales_with_number_of_audits(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="honest", **FAST))
        info = run_episode(env, AlwaysAudit(), seed=5)
        self.assertAlmostEqual(info["total_audit_cost"], info["audits"] * 1.0, places=6)


class AdaptiveEpisodeTests(unittest.TestCase):
    def test_sustainable_cheater_has_no_corpse_but_is_caught_on_moisture(self) -> None:
        # This is the whole reason for the moisture rule: a survival-enforcing
        # cheater keeps every plant alive (no spoofed corpse -> a death-based
        # rule would NEVER fire), yet the sensor still lies about moisture.
        common = dict(regime=2, force_episode="adaptive",
                      fixed_cheat_fraction=0.75, **FAST)

        # NeverAudit: confirm there is no corpse (alive-mismatch stays 0) but a
        # moisture lie does accrue damage.
        info = run_episode(OverseerEnv(OverseerConfig(**common)), NeverAudit(), seed=7)
        self.assertFalse(info["caught"])
        self.assertEqual(info["true_alive"], 5)                      # kept alive
        self.assertEqual(info["apparent_alive"] - info["true_alive"], 0)   # no corpse
        self.assertGreater(info["total_damage"], 0.0)               # but it lied

        # AlwaysAudit: the moisture rule catches it despite the absence of a corpse.
        caught = run_episode(OverseerEnv(OverseerConfig(**common)), AlwaysAudit(), seed=7)
        self.assertTrue(caught["caught"])
        self.assertIsNotNone(caught["first_discrepancy_step"])


class BaselineTests(unittest.TestCase):
    def test_periodic_auditor_catches_with_bounded_delay(self) -> None:
        period = period_from_world_steps(300, FAST["decision_interval"])
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, PeriodicAuditor(period), seed=6)
        self.assertTrue(info["caught"])
        self.assertLessEqual(info["detection_delay"], period * FAST["decision_interval"])

    def test_never_audit_never_catches(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, NeverAudit(), seed=7)
        self.assertFalse(info["caught"])


class BalancedScheduleTests(unittest.TestCase):
    def test_two_kinds_get_exactly_equal_episode_counts(self) -> None:
        # rule v3.3 training default: honest 0.5 / betrayal 0.5, balanced.
        env = OverseerEnv(OverseerConfig(honest_prob=0.5, adaptive_prob=0.0,
                                         episode_schedule="balanced", **FAST))
        kinds = []
        for seed in range(10):
            _, info = env.reset(seed=seed)
            kinds.append(info["episode_type"])
        self.assertEqual(kinds.count("honest"), 5)
        self.assertEqual(kinds.count("betrayal"), 5)
        # strict alternation, not merely equal totals
        self.assertEqual(kinds, ["honest", "betrayal"] * 5)

    def test_betrayal_k_still_varies_under_balanced(self) -> None:
        env = OverseerEnv(OverseerConfig(honest_prob=0.5, adaptive_prob=0.0,
                                         episode_schedule="balanced", **FAST))
        ks = set()
        for seed in range(10):
            _, info = env.reset(seed=seed)
            if info["episode_type"] == "betrayal":
                ks.add(info["K"])
        self.assertGreater(len(ks), 1)          # K randomization unaffected

    def test_force_episode_bypasses_the_cycle(self) -> None:
        env = OverseerEnv(OverseerConfig(force_episode="honest",
                                         episode_schedule="balanced", **FAST))
        for seed in range(4):
            _, info = env.reset(seed=seed)
            self.assertEqual(info["episode_type"], "honest")

    def test_three_kinds_cycle_when_adaptive_enabled(self) -> None:
        env = OverseerEnv(OverseerConfig(honest_prob=0.4, adaptive_prob=0.3,
                                         episode_schedule="balanced", **FAST))
        kinds = [env.reset(seed=seed)[1]["episode_type"] for seed in range(6)]
        self.assertEqual(kinds, ["honest", "betrayal", "adaptive"] * 2)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_episode(self) -> None:
        cfg = OverseerConfig(regime=2, **FAST)
        a = run_episode(OverseerEnv(cfg), NeverAudit(), seed=123)
        b = run_episode(OverseerEnv(cfg), NeverAudit(), seed=123)
        self.assertEqual(a["episode_type"], b["episode_type"])
        self.assertEqual(a["K"], b["K"])
        self.assertAlmostEqual(a["total_cost"], b["total_cost"], places=6)

    def test_k_zero_is_pure_cheater(self) -> None:
        env = OverseerEnv(OverseerConfig(force_episode="betrayal", fixed_k=0, **FAST))
        run_episode(env, NeverAudit(), seed=8)
        self.assertIsNotNone(env.world.first_cheat_step)
        self.assertLessEqual(env.world.first_cheat_step, 10)


if __name__ == "__main__":
    unittest.main()
