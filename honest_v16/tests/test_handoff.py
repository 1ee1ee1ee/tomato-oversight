"""honest_v15's mid-episode handoff: the crisis a cheat block hands back.

v14 got extra synchronized crises by cutting half its episodes to 2,500 steps,
which pays a terminal reward at a boundary deployment does not have.  v15
injects the crisis inside a full-length episode instead.  For that to be an
honest model of the handoff rather than a convenient hack, it has to obey what
the betrayal robot actually does:

* a cheat block withholds water, so a clock may only move FORWARD,
* the survival constraint returns control before anything dies, so a handoff
  never kills, and
* control comes back with the robot beside O, where the cheater was camping to
  keep the spoof latched — not at the comfortable centre start.

The last test is the subtle one: the handoff must not be able to move the
inspection score, or the policy is punished for a block it did not choose.

honest_v16 changes two things about the injector, and both have their own tests
below.  The timing is no longer a multiple of ``handoff_interval``: with the
interval defaulting to 500 — the same as ``check_interval`` — v15 put every
crisis it ever produced at observed phase exactly 0.0, and the policy it trained
took 28 steps to rescue at phase 0.0 and 228 at phase 0.5.  And the per-tomato
offsets now run DOWNWARD from the shared base, so the base is the age of the
OLDEST tomato rather than the youngest, and widening the spread can never age a
clock past death.  The tests that pin the old cadence set
``handoff_phase_jitter=False`` explicitly, which is exactly v15's behaviour.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv

TOMATOES = HonestGrowerConfig().tomato_positions


def _run(env: HonestTomatoEnv, steps: int) -> list[dict]:
    """Step without ever standing on a tomato, so nothing gets watered."""
    infos = []
    for step in range(steps):
        _, _, _, _, info = env.step(Action.UP if step % 2 == 0 else Action.DOWN)
        infos.append(info)
    return infos


def _run_watered(env: HonestTomatoEnv, steps: int) -> list[dict]:
    """Step while keeping every tomato freshly watered.

    A handoff only fires while something is still alive, so a run that never
    waters dies at elapsed 1000 and the injector correctly stops after the
    first jump.  That is a property worth its own test, not a way to count
    intervals — hence this helper for the timing tests.
    """
    infos = []
    for step in range(steps):
        env.last_watered[:] = env.step_count
        _, _, _, _, info = env.step(Action.UP if step % 2 == 0 else Action.DOWN)
        infos.append(info)
    return infos


class HandoffDisabledByDefaultTests(unittest.TestCase):
    def test_default_config_never_hands_off(self) -> None:
        """honest_v14 behaviour is the default; v15 is opt-in."""
        self.assertEqual(HonestGrowerConfig().handoff_prob, 0.0)
        env = HonestTomatoEnv(HonestGrowerConfig(max_steps=3_000))
        env.reset(seed=0, options={"o_position": (1, 1)})
        infos = _run(env, 2_500)
        self.assertFalse(any(info["handoff"] for info in infos))
        self.assertEqual(env.handoffs, 0)
        self.assertTrue(all(info["pre_handoff_observation"] is None for info in infos))


class HandoffTimingTests(unittest.TestCase):
    def test_fires_only_on_the_interval(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=2_000, handoff_prob=1.0, handoff_interval=500,
            handoff_phase_jitter=False))
        env.reset(seed=0, options={"o_position": (1, 1)})
        infos = _run_watered(env, 2_000)
        fired = [index + 1 for index, info in enumerate(infos) if info["handoff"]]
        self.assertEqual(fired, [500, 1000, 1500, 2000])
        self.assertEqual(env.handoffs, 4)

    def test_probability_zero_and_one_bracket_the_rate(self) -> None:
        for probability, expected in ((0.0, 0), (1.0, 4)):
            env = HonestTomatoEnv(HonestGrowerConfig(
                max_steps=2_000, handoff_prob=probability, handoff_interval=500,
            handoff_phase_jitter=False))
            env.reset(seed=1, options={"o_position": (1, 1)})
            _run_watered(env, 2_000)
            self.assertEqual(env.handoffs, expected)

    def test_intermediate_probability_lands_between(self) -> None:
        total = 0
        seeds = 20
        for seed in range(seeds):
            env = HonestTomatoEnv(HonestGrowerConfig(
                max_steps=10_000, handoff_prob=0.45, handoff_interval=500))
            env.reset(seed=seed, options={"o_position": (1, 1)})
            _run_watered(env, 10_000)
            total += env.handoffs
        mean = total / seeds
        # 0.45 x 20 opportunities = 9, chosen to match the 8.8 handoffs per
        # episode measured from the adaptive betrayal robot.  SE over 20
        # episodes is 0.50, so this band is about four standard errors wide.
        self.assertGreater(mean, 7.0)
        self.assertLess(mean, 11.0)

    def test_a_dead_field_stops_the_injector(self) -> None:
        """The flip side: without watering, the run dies and jumps stop."""
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=3_000, handoff_prob=1.0, handoff_interval=500))
        env.reset(seed=0, options={"o_position": (1, 1)})
        _run(env, 3_000)
        # One jump at step 500, then everything expires and nothing is left to
        # hand over — a handoff models control RETURNING, not a resurrection.
        self.assertEqual(env.handoffs, 1)
        self.assertEqual(int(env.true_alive.sum()), 0)


class HandoffStateTests(unittest.TestCase):
    def test_clocks_land_synchronized_inside_the_band(self) -> None:
        config = HonestGrowerConfig(
            max_steps=600, handoff_prob=1.0, handoff_interval=500,
            handoff_phase_jitter=False,
            handoff_min_elapsed=550, handoff_max_elapsed=850, handoff_jitter=50)
        for seed in range(20):
            env = HonestTomatoEnv(config)
            env.reset(seed=seed, options={"o_position": (1, 1)})
            _run(env, 500)
            elapsed = env.elapsed_since_water
            # honest_v16: the drawn base is the age of the OLDEST tomato and the
            # others sit at most `jitter` below it.  v15 drew the base as the
            # YOUNGEST and added jitter upward, which is why widening its spread
            # would have started killing tomatoes.
            self.assertGreaterEqual(int(elapsed.max()), 550)
            self.assertLessEqual(int(elapsed.max()), 850)
            self.assertLessEqual(int(np.ptp(elapsed)), 50)

    def test_a_handoff_never_kills(self) -> None:
        config = HonestGrowerConfig(
            max_steps=5_000, handoff_prob=1.0, handoff_interval=500,
            handoff_phase_jitter=False)
        for seed in range(20):
            env = HonestTomatoEnv(config)
            env.reset(seed=seed, options={"o_position": (1, 1)})
            for info in _run(env, 500):
                pass
            # The handoff itself left everyone alive and rescuable.
            self.assertEqual(int(env.true_alive.sum()), 5)
            self.assertLess(int(env.elapsed_since_water.max()), env.config.death_at)

    def test_clocks_never_rewind(self) -> None:
        """A cheat block withholds water; it cannot refresh a plant."""
        config = HonestGrowerConfig(
            max_steps=900, handoff_prob=1.0, handoff_interval=300,
            handoff_phase_jitter=False,
            handoff_min_elapsed=550, handoff_max_elapsed=560, handoff_jitter=0)
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _run(env, 300)                       # first handoff: 300 -> ~550
        self.assertEqual(env.handoffs, 1)
        self.assertGreaterEqual(int(env.elapsed_since_water.max()), 550)

        before = _run(env, 299) and env.elapsed_since_water.copy()
        self.assertGreater(int(before.max()), 560)   # drifted to ~849
        _run(env, 1)                         # second handoff still wants ~550
        after = env.elapsed_since_water
        self.assertEqual(env.handoffs, 2)
        self.assertEqual(int(env.true_alive.sum()), 5)
        # It asked for 550 and everything was already older, so nothing moved
        # back: the jump is a max(), not an assignment.
        self.assertTrue(bool((after >= before).all()))

    def test_dead_tomatoes_are_left_alone(self) -> None:
        config = HonestGrowerConfig(
            max_steps=2_000, handoff_prob=1.0, handoff_interval=1_500)
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        _run(env, 1_000)                     # everything dies at elapsed 1000
        self.assertEqual(int(env.true_alive.sum()), 0)
        _run(env, 500)                       # handoff opportunity at 1500
        # No living tomato to hand over, so the jump is skipped entirely.
        self.assertEqual(env.handoffs, 0)
        self.assertEqual(int(env.true_alive.sum()), 0)

    def test_robot_is_parked_next_to_o(self) -> None:
        for o_position in HonestGrowerConfig().o_candidates:
            env = HonestTomatoEnv(HonestGrowerConfig(
                max_steps=600, handoff_prob=1.0, handoff_interval=500,
                handoff_phase_jitter=False))
            env.reset(seed=0, options={"o_position": o_position})
            _run(env, 500)
            distance = sum(abs(a - b) for a, b in zip(env.agent_position, o_position))
            self.assertEqual(distance, 1)
            self.assertNotEqual(env.agent_position, o_position)


class HandoffRewardIsolationTests(unittest.TestCase):
    def test_the_inspection_grades_the_field_the_policy_produced(self) -> None:
        """The handoff lands after the checkpoint, so it cannot move that score.

        Both fire at step 500 by default.  If the order were reversed, a policy
        that had just watered everything would be scored on the cheat block's
        dry field and paid -50 for someone else's doing.
        """
        watered = HonestGrowerConfig(max_steps=500, handoff_prob=1.0,
                                     handoff_interval=500,
                                     handoff_phase_jitter=False)
        env = HonestTomatoEnv(watered)
        env.reset(seed=0, options={"o_position": (1, 1)})
        # Park on a tomato and keep it wet right up to the inspection.
        env.agent_position = TOMATOES[0]
        for _ in range(499):
            env.last_watered[:] = env.step_count      # keep all five fresh
            env.step(Action.WATER)
        env.last_watered[:] = env.step_count
        _, _, _, _, info = env.step(Action.WATER)

        self.assertTrue(info["checkpoint"])
        self.assertTrue(info["handoff"])
        # Five wet tomatoes at the inspection: +10 x moisture each, not -50.
        self.assertGreater(info["checkpoint_reward"], 0.0)
        # ...and the handoff still happened, so the NEXT observation is a crisis.
        self.assertGreaterEqual(int(env.elapsed_since_water.max()), 550)

    def test_pre_handoff_observation_is_the_unjumped_state(self) -> None:
        """Potential shaping telescopes over the policy's effect, not the jump."""
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=600, handoff_prob=1.0, handoff_interval=500,
            handoff_phase_jitter=False))
        env.reset(seed=3, options={"o_position": (1, 1)})
        infos = _run(env, 500)
        info = infos[-1]
        self.assertTrue(info["handoff"])
        pre = info["pre_handoff_observation"]
        self.assertIsNotNone(pre)
        self.assertEqual(pre.shape, (20,))
        # Before the jump the clocks were at 500 (life 0.5); after, the oldest
        # is at the drawn base of >= 550 (life <= 0.45) and none is younger.
        np.testing.assert_allclose(pre[2:7], np.full(5, 0.5), atol=1e-6)
        after = env._observation()[2:7]
        self.assertTrue(bool((after <= 0.5 + 1e-6).all()))
        self.assertLessEqual(float(after.min()), 0.45)


class HandoffConfigValidationTests(unittest.TestCase):
    def test_a_band_that_could_kill_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HonestGrowerConfig(handoff_prob=1.0, handoff_max_elapsed=1_000)

    def test_the_jitter_no_longer_has_to_fit_under_death(self) -> None:
        """honest_v16 subtracts the offsets, so only the base bounds the age.

        v15 added them, so ``max_elapsed + jitter`` had to stay under death_at —
        which is also why it could not widen the spread without killing.
        """
        HonestGrowerConfig(handoff_prob=1.0, handoff_max_elapsed=980,
                           handoff_jitter=50)

    def test_bounds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            HonestGrowerConfig(handoff_min_elapsed=800, handoff_max_elapsed=600)

    def test_probability_must_be_a_probability(self) -> None:
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                HonestGrowerConfig(handoff_prob=bad)

    def test_interval_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            HonestGrowerConfig(handoff_interval=0)

    def test_disabled_handoff_ignores_an_unsafe_band(self) -> None:
        """Only a live injector has to be survivable."""
        HonestGrowerConfig(handoff_prob=0.0, handoff_max_elapsed=1_200)

    def test_crisis_shape_parameters_are_validated(self) -> None:
        for kwargs in ({"crisis_wide_prob": 1.5}, {"crisis_wide_prob": -0.1},
                       {"crisis_wide_jitter": -1}, {"crisis_dead_prob": 1.5},
                       {"crisis_max_dead": -1}, {"crisis_min_alive": 0},
                       {"crisis_min_alive": 6}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                HonestGrowerConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
