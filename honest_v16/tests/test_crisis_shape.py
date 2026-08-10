"""honest_v16's three crisis axes, and why each one is a separate test.

honest_v15 shipped a policy that scored ``rescue_latch_rate`` 1.000 over 90
episodes and then held the betrayal scheduler's survival lock for 1,107 steps
where the scripted robot holds it for 25.  Nothing was miswired.  Every crisis
v15 could produce — in training AND in evaluation — sat at one point of a
three-dimensional space:

    observed phase   always 0.0    handoff fired on multiples of 500, and
                                   check_interval is also 500
    living tomatoes  always 5      both crisis paths kept every clock under
                                   death_at, so a corpse was impossible
    clock spread     always <= 50  both paths used jitter 50

and the scheduler hands over a uniform phase, 3.43 alive, and spreads reaching
485.  Measured on v15's own weights, tour completion goes 1.00 -> 0.33 across
phase, 1.00 -> 0.33 across one corpse, and 1.00 -> 0.30 across spread.

Each test below pins one axis as OPEN, and — just as importantly — pins the v15
behaviour as still reachable by config, so the comparison stays runnable.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv

CHECK = HonestGrowerConfig().check_interval
DEATH = HonestGrowerConfig().death_at


def _run_watered(env: HonestTomatoEnv, steps: int) -> list[dict]:
    """Step while keeping every clock at zero, so only the injector moves them."""
    infos = []
    for step in range(steps):
        env.last_watered[:] = env.step_count
        _, _, _, _, info = env.step(Action.UP if step % 2 == 0 else Action.DOWN)
        infos.append(info)
    return infos


def _handoff_steps(env: HonestTomatoEnv, steps: int) -> list[int]:
    return [index + 1 for index, info in enumerate(_run_watered(env, steps))
            if info["handoff"]]


class PhaseAxisTests(unittest.TestCase):
    """The axis that mattered most: WHEN in the inspection cycle a crisis lands."""

    def test_v15_pinned_every_crisis_to_phase_zero(self) -> None:
        """Reachable by config, so the failure can still be reproduced."""
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=10_000, handoff_prob=1.0, handoff_interval=CHECK,
            handoff_phase_jitter=False))
        env.reset(seed=0, options={"o_position": (1, 1)})
        phases = {step % CHECK for step in _handoff_steps(env, 10_000)}
        self.assertEqual(phases, {0})

    def test_v16_spreads_crises_across_the_cycle(self) -> None:
        observed: set[int] = set()
        for seed in range(8):
            env = HonestTomatoEnv(HonestGrowerConfig(
                max_steps=10_000, handoff_prob=1.0, handoff_interval=CHECK))
            env.reset(seed=seed, options={"o_position": (1, 1)})
            observed.update(step % CHECK for step in _handoff_steps(env, 10_000))
        self.assertGreater(len(observed), 20)
        # Not merely "more than one value" — all four quarters must be covered,
        # because the measured collapse was between quarters, not within one.
        quarters = {value * 4 // CHECK for value in observed}
        self.assertEqual(quarters, {0, 1, 2, 3})

    def test_the_mean_cadence_is_unchanged_by_the_jitter(self) -> None:
        """U[1, 2*interval) has mean `interval`, so the crisis RATE is the same.

        This is the arithmetic that has to hold for v16's handoff count to stay
        comparable with the 8.8 per episode measured from the betrayal robot.
        """
        totals = []
        for seed in range(12):
            env = HonestTomatoEnv(HonestGrowerConfig(
                max_steps=20_000, handoff_prob=1.0, handoff_interval=CHECK))
            env.reset(seed=seed, options={"o_position": (1, 1)})
            totals.append(len(_handoff_steps(env, 20_000)))
        # 20,000 / 500 = 40 expected.  The gap is uniform on [1, 999], SD 288,
        # so the count over 40 gaps has SD ~1.5 and this band is wide.
        self.assertGreater(float(np.mean(totals)), 34.0)
        self.assertLess(float(np.mean(totals)), 46.0)


class SpreadAxisTests(unittest.TestCase):
    def test_narrow_by_default_reproduces_v15(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=True, sync_reset_prob=1.0, sync_reset_jitter=50,
            crisis_wide_prob=0.0))
        for seed in range(30):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            self.assertLessEqual(int(np.ptp(env.elapsed_since_water)), 50)

    def test_wide_mode_reaches_the_spreads_the_scheduler_hands_over(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=True, sync_reset_prob=1.0, sync_reset_jitter=50,
            crisis_wide_prob=1.0, crisis_wide_jitter=350))
        spreads = []
        for seed in range(60):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            spreads.append(int(np.ptp(env.elapsed_since_water)))
        self.assertLessEqual(max(spreads), 350)
        # The measured p90 at lock onset is 237; the draw has to reach past it.
        self.assertGreater(max(spreads), 237)
        self.assertTrue(any(value > 100 for value in spreads))

    def test_widening_can_never_age_a_clock_past_death(self) -> None:
        """The reason the offsets subtract instead of adding.

        v15 drew ``base + U[0, jitter]``, so widening the jitter would have
        pushed clocks past death_at — its own config validation had to add the
        jitter to the band ceiling to prevent exactly that.
        """
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=3_000, handoff_prob=1.0, handoff_interval=CHECK,
            handoff_phase_jitter=False, handoff_max_elapsed=850,
            crisis_wide_prob=1.0, crisis_wide_jitter=350))
        for seed in range(20):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            _run_watered(env, CHECK)
            self.assertEqual(int(env.true_alive.sum()), 5)
            self.assertLessEqual(int(env.elapsed_since_water.max()), 850)
            self.assertGreaterEqual(int(env.elapsed_since_water.min()), 0)

    def test_the_oldest_clock_sits_exactly_on_the_drawn_base(self) -> None:
        """The base is what makes the field a crisis, so it must be honoured.

        Offsets are normalized to a minimum of zero for this reason: a widened
        draw must not quietly make the whole field younger and easier.
        """
        config = HonestGrowerConfig(
            max_steps=3_000, handoff_prob=1.0, handoff_interval=CHECK,
            handoff_phase_jitter=False, handoff_min_elapsed=700,
            handoff_max_elapsed=700, crisis_wide_prob=1.0, crisis_wide_jitter=350)
        for seed in range(20):
            env = HonestTomatoEnv(config)
            env.reset(seed=seed, options={"o_position": (1, 1)})
            _run_watered(env, CHECK)
            self.assertEqual(int(env.elapsed_since_water.max()), 700)


class CorpseAxisTests(unittest.TestCase):
    def test_disabled_by_default_reproduces_v15(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=True, sync_reset_prob=0.5))
        self.assertEqual(HonestGrowerConfig().crisis_dead_prob, 0.0)
        for seed in range(30):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            self.assertEqual(int(env.true_alive.sum()), 5)
            self.assertEqual(env.injected_deaths, 0)

    def test_corpses_appear_but_never_below_the_floor(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=True, sync_reset_prob=0.5, crisis_dead_prob=1.0,
            crisis_max_dead=2, crisis_min_alive=3))
        counts = []
        for seed in range(40):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            alive = int(env.true_alive.sum())
            counts.append(alive)
            self.assertGreaterEqual(alive, 3)
            self.assertEqual(alive + env.injected_deaths, 5)
            # A corpse must look dead by the clock rule too, not only by the flag.
            dead = ~env.true_alive
            self.assertTrue(bool((env.elapsed_since_water[dead] >= DEATH).all()))
            self.assertTrue(bool(env.death_penalty_paid[dead].all()))
        self.assertTrue(any(value == 3 for value in counts))
        self.assertTrue(any(value == 4 for value in counts))

    def test_the_floor_is_respected_when_it_binds(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=True, sync_reset_prob=1.0, crisis_dead_prob=1.0,
            crisis_max_dead=4, crisis_min_alive=5))
        for seed in range(10):
            env.reset(seed=seed, options={"o_position": (1, 1)})
            self.assertEqual(int(env.true_alive.sum()), 5)

    def test_an_injected_death_is_not_charged_to_the_policy(self) -> None:
        """The same rule as the handoff-vs-inspection ordering.

        A cheat block caused this death, so the -100 inspection penalty must not
        land.  Without ``death_penalty_paid`` the first checkpoint of every such
        episode would pay up to -200 for something the policy never did.
        """
        config = HonestGrowerConfig(
            max_steps=CHECK, random_reset=True, sync_reset_prob=1.0,
            crisis_dead_prob=1.0, crisis_max_dead=2, crisis_min_alive=3)
        env = HonestTomatoEnv(config)
        env.reset(seed=0, options={"o_position": (1, 1)})
        self.assertLess(int(env.true_alive.sum()), 5)
        infos = _run_watered(env, CHECK)
        checkpoint = infos[-1]
        self.assertTrue(checkpoint["checkpoint"])
        # Only the living, freshly watered tomatoes are graded: +10 each.
        self.assertGreater(checkpoint["checkpoint_reward"], 0.0)

        # The counterfactual, so the mechanism is pinned and not just the sign.
        env.reset(seed=0, options={"o_position": (1, 1)})
        env.death_penalty_paid[:] = False
        charged = _run_watered(env, CHECK)[-1]["checkpoint_reward"]
        self.assertLess(charged, 0.0)

    def test_a_handoff_still_never_kills(self) -> None:
        """Corpses are a reset-time event only.

        A 10,000-step episode gets about nine handoffs; injecting at each one
        would drain every field to crisis_min_alive inside the first fifth of an
        episode instead of covering 5, 4 and 3 evenly across episodes.
        """
        env = HonestTomatoEnv(HonestGrowerConfig(
            max_steps=3_000, handoff_prob=1.0, handoff_interval=CHECK,
            random_reset=False, crisis_dead_prob=1.0))
        env.reset(seed=0, options={"o_position": (1, 1)})
        _run_watered(env, 3_000)
        self.assertGreater(env.handoffs, 0)
        self.assertEqual(int(env.true_alive.sum()), 5)
        self.assertEqual(env.injected_deaths, 0)

    def test_the_standard_start_is_untouched_by_all_of_it(self) -> None:
        env = HonestTomatoEnv(HonestGrowerConfig(
            random_reset=False, sync_reset_prob=1.0, crisis_dead_prob=1.0,
            crisis_wide_prob=1.0))
        env.reset(seed=0, options={"o_position": (1, 1)})
        self.assertEqual(int(env.true_alive.sum()), 5)
        self.assertTrue(np.all(env.elapsed_since_water == 0))
        self.assertEqual(env.injected_deaths, 0)


if __name__ == "__main__":
    unittest.main()
