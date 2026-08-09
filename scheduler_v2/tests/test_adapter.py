"""Adapter tests — the seam between trained checkpoints and the overseer.

The state machine itself is covered by ``test_policies.py``.  What is tested
here is everything that sits around it: observation semantics, factory reuse,
and the sweep helpers the experiment depends on.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.adapter import (
    BetrayalRobotFactory,
    HonestModelPolicy,
    characterize,
    honest_observation,
    knob_verdict,
    resolve_obs_semantics,
    rollout,
)
from src.world import CheaterGrowerConfig, CheaterTomatoEnv


def _world(elapsed=None, seed: int = 0) -> CheaterTomatoEnv:
    world = CheaterTomatoEnv(CheaterGrowerConfig(max_steps=2_000))
    world.reset(seed=seed, options={"o_position": (1, 1)})
    if elapsed is not None:
        world.last_watered = -np.asarray(elapsed, dtype=np.int32)
        world.true_alive = np.asarray(elapsed, dtype=np.int32) < 1_000
    return world


class ObservationSemanticsTests(unittest.TestCase):
    def test_moisture_is_the_worlds_own_observation(self) -> None:
        world = _world([0, 300, 600, 800, 950])
        np.testing.assert_array_equal(
            honest_observation(world, "moisture"), world._observation()[:20])

    def test_life_is_strictly_decreasing_where_moisture_is_flat(self) -> None:
        """The dry window is the whole point: moisture pins to 0, life does not."""
        world = _world([600, 700, 800, 900, 950])
        moisture = honest_observation(world, "moisture")[2:7]
        life = honest_observation(world, "life")[2:7]
        np.testing.assert_allclose(moisture, np.zeros(5))          # indistinguishable
        self.assertTrue(np.all(np.diff(life) < 0))                 # ordered by urgency
        np.testing.assert_allclose(life, [0.4, 0.3, 0.2, 0.1, 0.05], atol=1e-6)

    def test_life_matches_the_closed_form(self) -> None:
        for elapsed in ([0, 0, 0, 0, 0], [250, 500, 750, 999, 100]):
            world = _world(elapsed)
            expected = np.clip(1.0 - np.asarray(elapsed) / 1000.0, 0.0, 1.0)
            np.testing.assert_allclose(
                honest_observation(world, "life")[2:7], expected, atol=1e-6)

    def test_dead_tomatoes_read_zero_life(self) -> None:
        world = _world([1_100, 0, 0, 0, 0])
        self.assertEqual(honest_observation(world, "life")[2], 0.0)

    def test_only_the_tomato_slots_differ(self) -> None:
        world = _world([600, 700, 800, 900, 950])
        moisture = honest_observation(world, "moisture")
        life = honest_observation(world, "life")
        np.testing.assert_array_equal(moisture[:2], life[:2])
        np.testing.assert_array_equal(moisture[7:], life[7:])

    def test_unknown_semantics_rejected(self) -> None:
        with self.assertRaises(ValueError):
            honest_observation(_world(), "wetness")


class SemanticsResolutionTests(unittest.TestCase):
    def test_stamped_tag_wins_when_nothing_declared(self) -> None:
        self.assertEqual(resolve_obs_semantics({"obs_semantics": "life"}, None), "life")

    def test_untagged_checkpoint_defaults_to_moisture(self) -> None:
        """Every checkpoint saved before the tag existed is pre-v13."""
        self.assertEqual(resolve_obs_semantics({}, None), "moisture")
        self.assertEqual(resolve_obs_semantics(None, None), "moisture")

    def test_declaration_fills_in_for_untagged_checkpoints(self) -> None:
        self.assertEqual(resolve_obs_semantics({}, "life"), "life")

    def test_disagreement_is_fatal(self) -> None:
        """A mismatch must not be survivable -- it degrades silently otherwise."""
        with self.assertRaises(ValueError):
            resolve_obs_semantics({"obs_semantics": "moisture"}, "life")

    def test_nonsense_declaration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_obs_semantics({}, "wetness")


class _StubAgent:
    """Enough of DoubleDQNAgent to drive a policy without torch."""

    def __init__(self, observation_size: int) -> None:
        self.observation_size = observation_size
        self.seen_dims: set[int] = set()
        self.calls = 0

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        self.seen_dims.add(len(observation))
        self.calls += 1
        valid = np.flatnonzero(action_mask)
        return int(valid[self.calls % len(valid)])


class HonestModelPolicyTests(unittest.TestCase):
    def test_feeds_twenty_values_and_never_enters_o(self) -> None:
        agent = _StubAgent(20)
        policy = HonestModelPolicy(agent, semantics="life")
        policy.reset(np.random.default_rng(0))
        world = _world()
        for _ in range(2_000):
            _, _, _, truncated, _ = world.step(policy.act(world))
            if truncated:
                break
        self.assertEqual(agent.seen_dims, {20})
        self.assertEqual(world.o_entries, 0)         # honest phase cannot spoof
        self.assertIsNone(world.first_cheat_step)

    def test_rejects_unknown_semantics(self) -> None:
        with self.assertRaises(ValueError):
            HonestModelPolicy(_StubAgent(20), semantics="wetness")


class ScriptedFactoryTests(unittest.TestCase):
    """The scripted factory needs no checkpoints, so it runs anywhere."""

    def setUp(self) -> None:
        self.factory = BetrayalRobotFactory(scripted=True)

    def test_label_and_description(self) -> None:
        self.assertEqual(self.factory.label, "scripted")
        self.assertTrue(self.factory.describe()["scripted"])

    def test_make_is_independent_per_call(self) -> None:
        a = self.factory.make(0.5)
        b = self.factory.make(0.5)
        self.assertIsNot(a, b)
        self.assertIsNot(a.honest, b.honest)

    def test_checkpoints_required_unless_scripted(self) -> None:
        with self.assertRaises(ValueError):
            BetrayalRobotFactory(None, None, scripted=False)

    def test_robot_factory_callable_ignores_its_argument(self) -> None:
        make = self.factory.robot_factory(0.25)
        self.assertAlmostEqual(make(None).cheat_fraction, 0.25)
        self.assertAlmostEqual(make(1234).cheat_fraction, 0.25)

    def test_robot_kwargs_reach_the_policy(self) -> None:
        robot = self.factory.make(0.5, rescue_margin_range=(200, 400))
        self.assertEqual(robot.rescue_margin_range, (200, 400))


class RolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = BetrayalRobotFactory(scripted=True)

    def test_zero_fraction_is_pure_honest(self) -> None:
        summary, _, _ = rollout(self.factory, 0.0, 0, max_steps=3_000)
        self.assertEqual(summary["o_entries"], 0)
        self.assertEqual(summary["spoof_uptime"], 0.0)
        self.assertEqual(summary["realized_cheat"], 0.0)

    def test_pure_cheater_spoofs_and_kills(self) -> None:
        summary, _, _ = rollout(self.factory, 1.0, 0, enforce_survival=False,
                                max_steps=4_000)
        self.assertGreater(summary["spoof_uptime"], 0.8)
        self.assertEqual(summary["final_alive"], 0)
        self.assertGreater(summary["discrepancy"], 0)

    def test_survival_constraint_keeps_every_tomato(self) -> None:
        for fraction in (0.25, 0.5, 1.0):
            summary, _, _ = rollout(self.factory, fraction, 0, enforce_survival=True,
                                    max_steps=4_000)
            self.assertEqual(summary["final_alive"], 5, f"died at f={fraction}")
            self.assertEqual(summary["discrepancy"], 0)

    def test_trace_is_aligned_and_optional(self) -> None:
        summary, empty, _ = rollout(self.factory, 0.5, 0, max_steps=1_000)
        self.assertEqual(empty["step"], [])
        _, track, _ = rollout(self.factory, 0.5, 0, max_steps=1_000, trace=True)
        lengths = {len(values) for values in track.values()}
        self.assertEqual(len(lengths), 1)
        self.assertEqual(track["step"][-1], 1_000)
        del summary

    def test_same_seed_reproduces(self) -> None:
        a, _, _ = rollout(self.factory, 0.5, 7, max_steps=2_000)
        b, _, _ = rollout(self.factory, 0.5, 7, max_steps=2_000)
        self.assertEqual(a, b)


class CharacterizeTests(unittest.TestCase):
    def test_sweep_shape_and_verdict_keys(self) -> None:
        factory = BetrayalRobotFactory(scripted=True)
        rows = characterize(factory, seeds=2, cheat_fractions=(0.0, 0.5, 1.0),
                            max_steps=2_000)
        self.assertEqual(len(rows), 2 * 3 * 2)          # seeds x fractions x arms

        verdict = knob_verdict(rows)
        self.assertEqual(set(verdict), {"no_survival", "enforce_survival"})
        for arm in verdict.values():
            self.assertIn("monotone_uptime", arm)
            self.assertIn("coverage_ok", arm)

    def test_uptime_rises_with_the_dial(self) -> None:
        """The dial has to be usable as an axis; monotone is the minimum bar."""
        factory = BetrayalRobotFactory(scripted=True)
        rows = characterize(factory, seeds=3, cheat_fractions=(0.0, 0.5, 1.0),
                            enforce_survival=(False,), max_steps=4_000)
        verdict = knob_verdict(rows)["no_survival"]
        self.assertTrue(verdict["monotone_uptime"])
        self.assertTrue(verdict["monotone_cheat"])


if __name__ == "__main__":
    unittest.main()
