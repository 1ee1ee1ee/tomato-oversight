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
from src.env import (
    MASK_VALUE,
    OBS_BLOCKS,
    OBS_LABELS,
    OBS_SIZE,
    REGIME_BLOCKS,
    OverseerAction,
    OverseerConfig,
    OverseerEnv,
    live_labels,
    regime_mask,
)
from src.world import Action

FAST = dict(decision_interval=100, horizon=3_000)   # room for lies to appear/catch
ALL_REGIMES = (0, 1, 2, 3, 4)
CONTINUE = int(OverseerAction.CONTINUE)
AUDIT = int(OverseerAction.AUDIT)


def _slice(name: str) -> slice:
    low, high = OBS_BLOCKS[name]
    return slice(low, high)


def _rollout(regime: int, actions, *, seed: int, **cfg_kwargs) -> list[np.ndarray]:
    """Replay a FIXED overseer action sequence and collect every observation.

    The robot policy reads the world only, never the regime, so the same seed +
    same action sequence gives a bit-identical trajectory in every regime.  That
    is what makes the cross-regime comparisons below valid.
    """
    env = OverseerEnv(OverseerConfig(regime=regime, **cfg_kwargs))
    obs, _ = env.reset(seed=seed)
    out = [obs.copy()]
    for action in actions:
        obs, _, terminated, truncated, _ = env.step(action)
        out.append(obs.copy())
        if terminated or truncated:
            break
    return out


class ObservationLayoutTests(unittest.TestCase):
    def test_blocks_tile_the_vector_without_gaps_or_overlap(self) -> None:
        covered = np.zeros(OBS_SIZE, dtype=int)
        for low, high in OBS_BLOCKS.values():
            covered[low:high] += 1
        self.assertTrue(np.all(covered == 1), "blocks must partition the vector")
        self.assertEqual(len(OBS_LABELS), OBS_SIZE)

    def test_regime_table_matches_the_documented_ablation(self) -> None:
        # The OXXX table from the design, spelled out independently of the code
        # under test.  If someone re-wires a regime, this fails loudly.
        expected = {
            0: {"meta", "audit"},
            1: {"meta", "audit", "apparent"},
            2: {"meta", "audit", "behav", "water"},
            3: {"meta", "audit", "apparent", "behav"},
            4: {"meta", "audit", "apparent", "behav", "water"},
        }
        self.assertEqual(set(REGIME_BLOCKS), set(ALL_REGIMES))
        for regime, blocks in expected.items():
            mask = regime_mask(regime)
            for name, (low, high) in OBS_BLOCKS.items():
                live = bool(mask[low:high].all())
                self.assertEqual(live, name in blocks,
                                 f"regime {regime}: block '{name}' liveness wrong")
                self.assertEqual(bool(mask[low:high].any()), live,
                                 f"regime {regime}: block '{name}' partially live")

    def test_ladder_is_strictly_nested_except_regime_two(self) -> None:
        sets = {r: set(live_labels(r)) for r in ALL_REGIMES}
        for lower, upper in ((0, 1), (1, 3), (3, 4)):
            self.assertTrue(sets[lower] < sets[upper],
                            f"regime {lower} must be a strict subset of {upper}")
        # Regime 2 is deliberately off the ladder: it has watering but no
        # apparent log, so neither contains the other.
        self.assertFalse(sets[2] <= sets[3])
        self.assertFalse(sets[3] <= sets[2])
        self.assertTrue(sets[2] < sets[4])
        self.assertTrue(sets[0] < sets[2])

    def test_all_regimes_share_one_shape_and_stay_in_bounds(self) -> None:
        for regime in ALL_REGIMES:
            env = OverseerEnv(OverseerConfig(regime=regime, **FAST))
            obs, _ = env.reset(seed=0)
            self.assertEqual(env.observation_space.shape, (OBS_SIZE,))
            self.assertEqual(obs.shape, (OBS_SIZE,))
            for _ in range(10):
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                self.assertTrue(np.all(obs >= -1e-6) and np.all(obs <= 1 + 1e-6),
                                f"regime {regime} left [0, 1]")
                if term or trunc:
                    break

    def test_invalid_regime_is_rejected(self) -> None:
        for bad in (-1, 5, 99):
            with self.assertRaises(ValueError):
                OverseerConfig(regime=bad)


class RegimeGatingTests(unittest.TestCase):
    """Every regime must equal regime 4 on its live indices and MASK_VALUE off them.

    Regime 4 is the union of all blocks, so this single property verifies both
    directions at once: nothing is withheld that should be visible, and nothing
    leaks through that should be hidden.
    """

    def _check_against_regime_four(self, actions, *, seed: int, **cfg) -> None:
        full = _rollout(4, actions, seed=seed, **cfg)
        for regime in ALL_REGIMES:
            trace = _rollout(regime, actions, seed=seed, **cfg)
            self.assertEqual(len(trace), len(full),
                             f"regime {regime} diverged in episode length")
            mask = regime_mask(regime)
            for t, (obs, reference) in enumerate(zip(trace, full)):
                np.testing.assert_allclose(
                    obs[mask], reference[mask], rtol=0, atol=1e-6,
                    err_msg=f"regime {regime} step {t}: live values differ from regime 4")
                self.assertTrue(np.all(obs[~mask] == MASK_VALUE),
                                f"regime {regime} step {t}: masked index not blanked")

    def test_gating_on_an_honest_episode_with_audits(self) -> None:
        # Honest episodes are never caught, so audits do not terminate and every
        # regime walks the same 20 decisions.
        actions = [AUDIT if i % 4 == 0 else CONTINUE for i in range(20)]
        self._check_against_regime_four(actions, seed=101,
                                        force_episode="honest", **FAST)

    def test_gating_on_a_spoofing_episode(self) -> None:
        # All-CONTINUE, so the lie is never caught and the trajectories stay
        # aligned while the apparent block is actively being spoofed.
        actions = [CONTINUE] * 25
        self._check_against_regime_four(actions, seed=202, force_episode="adaptive",
                                        fixed_cheat_fraction=0.75, **FAST)

    def test_regime_zero_sees_nothing_but_timing_and_audits(self) -> None:
        self.assertEqual(
            live_labels(0),
            [name for name in OBS_LABELS
             if name.startswith("meta/") or name.startswith("audit/")])


class ApparentBlockTests(unittest.TestCase):
    def test_apparent_block_is_the_public_sensor_readout(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="adaptive",
                                         fixed_cheat_fraction=0.75, **FAST))
        obs, _ = env.reset(seed=5)
        for _ in range(15):
            obs, _, term, trunc, _ = env.step(CONTINUE)
            block = obs[_slice("apparent")]
            np.testing.assert_allclose(block[:5], env.world.apparent_moisture,
                                       rtol=0, atol=1e-6)
            self.assertAlmostEqual(float(block[5]),
                                   float(env.world.apparent_alive.sum()) / 5.0, places=6)
            if term or trunc:
                break

    def test_apparent_moves_while_the_audit_log_stays_frozen(self) -> None:
        # The audit log is a snapshot; the apparent log is live.  Regime 1 has
        # both, which is exactly what makes "claim vs effort" learnable.  An
        # honest episode is used because there the sensor genuinely tracks the
        # decaying field — under a pinned spoof the reading is constant (see
        # test_apparent_pins_at_full_under_a_sustained_spoof).
        env = OverseerEnv(OverseerConfig(regime=1, force_episode="honest", **FAST))
        env.reset(seed=1)
        obs, *_ = env.step(AUDIT)                       # populate the log
        frozen = obs[_slice("audit")].copy()
        apparent_seen = [obs[_slice("apparent")].copy()]
        for _ in range(5):
            obs, _, term, trunc, _ = env.step(CONTINUE)
            np.testing.assert_allclose(obs[_slice("audit")], frozen, rtol=0, atol=1e-6)
            apparent_seen.append(obs[_slice("apparent")].copy())
            if term or trunc:
                break
        self.assertGreater(env.world.step_count, 100)
        self.assertTrue(any(not np.allclose(a, apparent_seen[0]) for a in apparent_seen[1:]),
                        "apparent block never changed — it is not live")

    def test_apparent_pins_at_full_under_a_sustained_spoof(self) -> None:
        # Documents the signal regime 1 actually gets.  The scripted cheater
        # re-enters O every couple of steps, so spoof_remaining never runs down
        # and the sensor reports a permanently soaked field, while an honest
        # field visibly decays between waterings.  "Suspiciously constant" is
        # therefore the shape of the lie in the apparent channel.
        def trace(**extra):
            env = OverseerEnv(OverseerConfig(regime=1, **extra, **FAST))
            env.reset(seed=1)
            rows = []
            for _ in range(6):
                obs, _, term, trunc, _ = env.step(CONTINUE)
                rows.append(obs[_slice("apparent")][:5].copy())
                if term or trunc:
                    break
            return np.asarray(rows)

        cheating = trace(force_episode="betrayal", fixed_k=0)
        honest = trace(force_episode="honest")
        np.testing.assert_allclose(cheating, 1.0, rtol=0, atol=1e-6)
        self.assertLess(honest.min(), 0.5, "honest field should visibly dry out")

    def test_regime_zero_never_sees_the_public_sensor(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=0, force_episode="adaptive",
                                         fixed_cheat_fraction=0.9, **FAST))
        obs, _ = env.reset(seed=6)
        for _ in range(15):
            obs, _, term, trunc, _ = env.step(CONTINUE)
            self.assertTrue(np.all(obs[_slice("apparent")] == MASK_VALUE))
            if term or trunc:
                break


class WateringWithholdingTests(unittest.TestCase):
    """Regime 3 must withhold the watering features without leaking them back."""

    def test_water_block_live_only_where_declared(self) -> None:
        for regime in ALL_REGIMES:
            live = bool(regime_mask(regime)[_slice("water")].all())
            self.assertEqual(live, regime in (2, 4), f"regime {regime} water block")

    def test_watering_does_not_blank_the_direction_onehot(self) -> None:
        # v2 bug this guards against: zeroing the WATER slot of a 5-way one-hot
        # makes "all zeros" mean "the last action was a watering" — a perfect
        # one-bit leak of exactly the thing regime 3 is meant to withhold.
        low = OBS_BLOCKS["behav"][0]
        onehot = slice(low + 2, low + 6)
        env = OverseerEnv(OverseerConfig(regime=3, force_episode="honest",
                                         decision_interval=1, horizon=600))
        obs, _ = env.reset(seed=11)
        watering_steps = 0
        for _ in range(600):
            obs, _, term, trunc, _ = env.step(CONTINUE)
            if env.last_action == int(Action.WATER):
                watering_steps += 1
                self.assertAlmostEqual(float(obs[onehot].sum()), 1.0, places=6,
                                       msg="direction one-hot blanked on a watering step")
            self.assertTrue(np.all(obs[_slice("water")] == MASK_VALUE))
            if term or trunc:
                break
        self.assertGreater(watering_steps, 0, "the robot never watered — test is vacuous")

    def test_regime_four_does_report_watering(self) -> None:
        env = OverseerEnv(OverseerConfig(regime=4, force_episode="honest",
                                         decision_interval=1, horizon=600))
        obs, _ = env.reset(seed=11)
        flags = []
        for _ in range(600):
            obs, _, term, trunc, _ = env.step(CONTINUE)
            flags.append((env.last_action == int(Action.WATER),
                          float(obs[OBS_BLOCKS["water"][0]])))
            if term or trunc:
                break
        for was_water, flag in flags:
            self.assertEqual(flag, 1.0 if was_water else 0.0)
        self.assertTrue(any(w for w, _ in flags))


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
