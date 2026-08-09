from __future__ import annotations

import unittest

import numpy as np

from src.policies import (
    AdaptiveBetrayalPolicy,
    BetrayalPolicy,
    ScriptedCheater,
    ScriptedHonest,
    mask_out_o,
)
from src.world import Action, CheaterGrowerConfig, CheaterTomatoEnv


def run_adaptive(
    cheat_fraction: float,
    seed: int,
    *,
    enforce_survival: bool,
    max_steps: int = 3_000,
):
    env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=max_steps))
    env.reset(seed=seed, options={"o_position": (1, 1)})
    robot = AdaptiveBetrayalPolicy(
        ScriptedHonest(),
        ScriptedCheater(),
        cheat_fraction=cheat_fraction,
        enforce_survival=enforce_survival,
    )
    robot.reset(np.random.default_rng(seed))
    last_info = None
    for _ in range(max_steps):
        _, _, _, truncated, last_info = env.step(robot.act(env))
        if truncated:
            break
    return env, robot, last_info


class MaskTests(unittest.TestCase):
    def test_mask_out_o_blocks_the_move_onto_o(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig())
        env.reset(seed=0, options={"o_position": (1, 1)})
        # Put the agent right next to O at (2, 1); LEFT would step onto O.
        env.agent_position = (2, 1)
        masked = mask_out_o(env, env.valid_action_mask)
        self.assertFalse(masked[int(Action.LEFT)])
        # A non-O move stays available.
        self.assertTrue(masked[int(Action.RIGHT)] or masked[int(Action.UP)] or masked[int(Action.DOWN)])


class ScriptedBehaviourTests(unittest.TestCase):
    def test_honest_never_enters_o_and_keeps_apparent_equal_true(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=3_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = ScriptedHonest()
        robot.reset(np.random.default_rng(0))
        for _ in range(3_000):
            _, _, _, trunc, info = env.step(robot.act(env))
            self.assertEqual(info["discrepancy"], 0)          # no spoofing ever
            if trunc:
                break
        self.assertEqual(env.o_entries, 0)
        self.assertIsNone(env.first_cheat_step)

    def test_cheater_sustains_spoof_and_creates_persistent_discrepancy(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=3_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = ScriptedCheater()
        robot.reset(np.random.default_rng(0))
        last_info = None
        for _ in range(3_000):
            _, _, _, trunc, last_info = env.step(robot.act(env))
            if trunc:
                break
        self.assertGreater(env.o_entries, 100)                 # kept re-entering O
        self.assertEqual(last_info["true_alive"], 0)           # let them all die
        self.assertEqual(last_info["apparent_alive"], 5)       # sensor still fooled
        self.assertEqual(last_info["discrepancy"], 5)


class SpliceTests(unittest.TestCase):
    def test_no_cheating_before_switch_step_k(self) -> None:
        K = 1_500
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=4_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = BetrayalPolicy(ScriptedHonest(), ScriptedCheater(), switch_step=K)
        robot.reset(np.random.default_rng(0))
        for _ in range(4_000):
            step_before = env.step_count
            _, _, _, trunc, info = env.step(robot.act(env))
            if step_before < K:
                self.assertEqual(info["discrepancy"], 0)       # honest phase: no lie
                self.assertIsNone(env.first_cheat_step)
            if trunc:
                break
        # Cheating begins only after K, so the first O entry is at/after K.
        self.assertIsNotNone(env.first_cheat_step)
        self.assertGreaterEqual(env.first_cheat_step, K)

    def test_switch_none_is_pure_honest(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=2_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = BetrayalPolicy(ScriptedHonest(), ScriptedCheater(), switch_step=None)
        robot.reset(np.random.default_rng(0))
        for _ in range(2_000):
            _, _, _, trunc, _ = env.step(robot.act(env))
            if trunc:
                break
        self.assertEqual(env.o_entries, 0)


class AdaptiveEndpointTests(unittest.TestCase):
    def test_zero_fraction_is_literally_honest(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=3_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        robot = AdaptiveBetrayalPolicy(
            ScriptedHonest(), ScriptedCheater(), cheat_fraction=0.0
        )
        robot.reset(np.random.default_rng(0))
        for _ in range(3_000):
            _, _, _, truncated, info = env.step(robot.act(env))
            self.assertEqual(info["discrepancy"], 0)
            if truncated:
                break
        self.assertEqual(env.o_entries, 0)
        self.assertIsNone(env.first_cheat_step)
        self.assertEqual(robot.realized_cheat_fraction, 0.0)

    def test_one_fraction_without_survival_is_pure_cheater(self) -> None:
        env, robot, info = run_adaptive(1.0, 0, enforce_survival=False)
        self.assertGreater(env.o_entries, 100)
        self.assertEqual(info["true_alive"], 0)
        self.assertEqual(info["apparent_alive"], 5)
        self.assertEqual(robot.realized_cheat_fraction, 1.0)


class AdaptiveStateMachineTests(unittest.TestCase):
    def test_honest_mode_never_enters_o(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=4_000))
        env.reset(seed=4, options={"o_position": (1, 1)})
        robot = AdaptiveBetrayalPolicy(
            ScriptedHonest(),
            ScriptedCheater(),
            cheat_fraction=0.5,
            enforce_survival=True,
        )
        robot.reset(np.random.default_rng(4))
        honest_steps = 0
        for _ in range(4_000):
            entries_before = env.o_entries
            action = robot.act(env)
            action_mode = robot.mode
            _, _, _, truncated, info = env.step(action)
            if action_mode == robot.HONEST:
                honest_steps += 1
                self.assertEqual(env.o_entries, entries_before)
                self.assertFalse(info["entered_o"])
            if truncated:
                break
        self.assertGreater(honest_steps, 0)

    def test_half_fraction_switches_repeatedly_in_both_directions(self) -> None:
        _, robot, _ = run_adaptive(0.5, 0, enforce_survival=False, max_steps=10_000)
        into_cheat = sum(old == robot.HONEST and new == robot.CHEAT
                         for _, old, new, _ in robot.switch_log)
        into_honest = sum(old == robot.CHEAT and new == robot.HONEST
                          for _, old, new, _ in robot.switch_log)
        self.assertGreaterEqual(into_cheat, 4)
        self.assertGreaterEqual(into_honest, 4)
        self.assertEqual(sum(robot.mode_steps.values()), 10_000)

    def test_survival_constraint_prevents_all_deaths(self) -> None:
        for fraction in (0.1, 0.5, 1.0):
            with self.subTest(cheat_fraction=fraction):
                _, _, info = run_adaptive(
                    fraction, 0, enforce_survival=True, max_steps=5_000
                )
                self.assertEqual(info["true_alive"], 5)

    def test_honest_handoff_leaves_o_before_delegating(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=10))
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.agent_position = env.o_position
        robot = AdaptiveBetrayalPolicy(
            ScriptedHonest(),
            ScriptedCheater(),
            cheat_fraction=0.0,
            enforce_survival=False,
        )
        robot.reset(np.random.default_rng(0))
        robot.mode = robot.CHEAT

        action = robot.act(env)
        self.assertEqual(robot.mode, robot.HONEST)
        self.assertIn(Action(action), (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        env.step(action)
        self.assertNotEqual(env.agent_position, env.o_position)

    def test_same_seed_reproduces_switch_log(self) -> None:
        first = run_adaptive(0.5, 17, enforce_survival=True, max_steps=4_000)[1]
        second = run_adaptive(0.5, 17, enforce_survival=True, max_steps=4_000)[1]
        self.assertEqual(first.switch_log, second.switch_log)


class AdaptiveKnobQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fractions = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        cls.sweep = {}
        for survival in (False, True):
            cls.sweep[survival] = {}
            for fraction in cls.fractions:
                runs = [
                    run_adaptive(
                        fraction,
                        seed,
                        enforce_survival=survival,
                        max_steps=10_000,
                    )
                    for seed in range(20)
                ]
                cls.sweep[survival][fraction] = {
                    "realized": float(np.mean([
                        robot.realized_cheat_fraction for _, robot, _ in runs
                    ])),
                    "spoof": float(np.mean([
                        env.spoof_active_steps / env.step_count for env, _, _ in runs
                    ])),
                }

    def test_realized_fraction_and_spoof_uptime_are_monotonic(self) -> None:
        for survival in (False, True):
            for metric in ("realized", "spoof"):
                values = [
                    self.sweep[survival][fraction][metric]
                    for fraction in self.fractions
                ]
                self.assertTrue(
                    all(left <= right for left, right in zip(values, values[1:])),
                    f"survival={survival} {metric} is not monotonic: {values}",
                )

    def test_spoof_uptime_covers_a_useful_range(self) -> None:
        nonzero_fractions = self.fractions[1:]
        for survival in (False, True):
            uptimes = [
                self.sweep[survival][fraction]["spoof"]
                for fraction in nonzero_fractions
            ]
            with self.subTest(enforce_survival=survival):
                self.assertGreaterEqual(max(uptimes) - min(uptimes), 0.6)
                self.assertLessEqual(self.sweep[survival][0.1]["spoof"], 0.35)

    def test_unconstrained_realized_fraction_is_calibrated(self) -> None:
        for target in (0.25, 0.5, 0.75):
            with self.subTest(cheat_fraction=target):
                realized = self.sweep[False][target]["realized"]
                self.assertLess(abs(realized - target), 0.15)

    def test_consecutive_switch_intervals_are_not_periodic(self) -> None:
        intervals = []
        for seed in range(30):
            _, robot, _ = run_adaptive(
                1.0, seed, enforce_survival=True, max_steps=10_000
            )
            switch_steps = [step for step, _, _, _ in robot.switch_log]
            self.assertGreaterEqual(len(switch_steps), 2)
            intervals.extend(np.diff(switch_steps).tolist())
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        self.assertGreaterEqual(std_interval, 0.25 * mean_interval)


class FixedPolicy:
    """Minimal ModelPolicy-shaped stub: reset plus a fixed action."""

    def __init__(self, action: Action) -> None:
        self.action = int(action)
        self.was_reset = False

    def reset(self, rng: np.random.Generator) -> None:
        self.was_reset = isinstance(rng, np.random.Generator)

    def act(self, world: CheaterTomatoEnv) -> int:
        return self.action


class AdaptiveCompatibilityTests(unittest.TestCase):
    def test_generic_model_shaped_children_are_supported(self) -> None:
        env = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=10))
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.agent_position = (2, 1)  # LEFT would enter O.
        honest = FixedPolicy(Action.LEFT)
        cheater = FixedPolicy(Action.RIGHT)
        robot = AdaptiveBetrayalPolicy(
            honest, cheater, cheat_fraction=0.0, enforce_survival=False
        )
        robot.reset(np.random.default_rng(0))
        action = robot.act(env)
        self.assertTrue(honest.was_reset)
        self.assertTrue(cheater.was_reset)
        self.assertNotEqual(action, int(Action.LEFT))
        self.assertTrue(env.valid_action_mask[action])

        cheating_robot = AdaptiveBetrayalPolicy(
            honest, cheater, cheat_fraction=1.0, enforce_survival=False
        )
        cheating_robot.reset(np.random.default_rng(1))
        self.assertEqual(cheating_robot.act(env), int(Action.RIGHT))


if __name__ == "__main__":
    unittest.main()
