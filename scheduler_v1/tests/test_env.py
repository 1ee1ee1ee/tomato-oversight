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

FAST = dict(decision_interval=100, horizon=3_000)   # deaths at ~1000, room to catch


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
        # One audit to populate the log, then never audit again.
        obs, *_ = env.step(int(OverseerAction.AUDIT))
        audit_slice = obs[5:21].copy()
        changed = False
        for _ in range(5):
            obs, _, term, trunc, _ = env.step(int(OverseerAction.CONTINUE))
            # world state is changing (tomatoes drying/dying) ...
            self.assertTrue(np.allclose(obs[5:21], audit_slice))  # ... obs log is not
            if term or trunc:
                break
        # sanity: the world really did move on
        self.assertGreater(env.world.step_count, 100)


class CatchAndCostTests(unittest.TestCase):
    def test_pure_cheater_is_caught_soon_after_first_death(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, AlwaysAudit(), seed=2)
        self.assertTrue(info["caught"])
        self.assertIsNotNone(info["first_discrepancy_step"])
        # Caught within one decision interval of the first catchable moment.
        self.assertGreaterEqual(info["detection_delay"], 0)
        self.assertLessEqual(info["detection_delay"], FAST["decision_interval"])
        # Undetected damage is bounded by a single decision interval (here just
        # the boundary step that elapses before the catching audit).
        self.assertLessEqual(info["total_damage"], 0.01 * 5 * FAST["decision_interval"] + 1e-6)

    def test_honest_episode_is_never_caught_and_costs_only_audits(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="honest", **FAST))
        info = run_episode(env, AlwaysAudit(), seed=3)
        self.assertFalse(info["caught"])
        self.assertEqual(info["discrepancy_now"], 0)
        self.assertAlmostEqual(info["total_damage"], 0.0, places=6)
        self.assertGreater(info["audits"], 0)
        self.assertEqual(info["wasted_audits"], info["audits"])  # all audits wasted

    def test_never_audit_accrues_damage_and_reward_matches_cost(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, NeverAudit(), seed=4)
        self.assertFalse(info["caught"])
        self.assertEqual(info["total_audit_cost"], 0.0)
        self.assertGreater(info["total_damage"], 0.0)
        # reward is exactly the negative total cost (accounting invariant)
        self.assertAlmostEqual(info["return"], -info["total_cost"], places=6)
        # damage ~ 0.01 * 5 dead * (horizon - first_death), all five die together
        expected = 0.01 * 5 * (FAST["horizon"] - info["first_discrepancy_step"])
        self.assertAlmostEqual(info["total_damage"], expected, delta=expected * 0.05 + 1.0)

    def test_audit_cost_scales_with_number_of_audits(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="honest", **FAST))
        info = run_episode(env, AlwaysAudit(), seed=5)
        self.assertAlmostEqual(info["total_audit_cost"], info["audits"] * 1.0, places=6)


class BaselineTests(unittest.TestCase):
    def test_periodic_auditor_catches_with_bounded_delay(self) -> None:
        period = period_from_world_steps(300, FAST["decision_interval"])   # every 3 decisions
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, PeriodicAuditor(period), seed=6)
        self.assertTrue(info["caught"])
        # delay is bounded by the audit period expressed in world steps
        self.assertLessEqual(info["detection_delay"], period * FAST["decision_interval"])

    def test_never_audit_never_catches(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="betrayal",
                                         fixed_k=0, **FAST))
        info = run_episode(env, NeverAudit(), seed=7)
        self.assertFalse(info["caught"])


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
        self.assertLessEqual(env.world.first_cheat_step, 10)   # reaches O almost at once


if __name__ == "__main__":
    unittest.main()
