"""The honest_v15 acceptance gate: rescue inside the spoof latch.

Why this replaced ``full_survival_rate`` as the selection metric:

* the standard start is ONE deterministic field (with ``random_reset=False``
  and a fixed O, ``reset`` never touches the RNG), and a masked uniform-random
  policy already survives it 0.70 of the time, so it cannot rank policies;
* what the overseer study actually needs from the honest half of the betrayal
  robot is that its rescue tour finish before cheater_v1's 500-step spoof latch
  expires.  Measured with the scripted robot, a ~25-step tour costs the
  scheduler ZERO spoof uptime, and uptime then reaches 1.000 at cheat_fraction
  1.0 with 5/5 alive; honest_v13 reached 0.467 with 2.67 alive.

These tests pin the metric's semantics with stub policies that need no torch:
a competent tour must score ~1, an idle policy must score 0, and — the part
that is easy to get wrong — a tour that succeeds too SLOWLY must also score 0
while still being recorded as a completed tour.

honest_v16 adds the tests that would have caught v15 before a Colab run was
spent on it.  v15's gate built every episode straight after ``reset()`` with
five living tomatoes, so it could not distinguish a policy that rescues any
handed-over field from one that rescues only fields at phase 0.0 with nobody
dead — and the policy it selected was the second kind.  ``PhaseZeroAgent`` and
``FullFieldAgent`` are those failure modes made explicit: each is competent in
exactly one slice of the grid and must score near that slice's share of it.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.env import Action, HonestGrowerConfig, HonestTomatoEnv
from src.evaluation import SPOOF_LATCH_STEPS, evaluate_rescue, rescue_summary

CONFIG = HonestGrowerConfig()
TOMATOES = CONFIG.tomato_positions
GRID = CONFIG.grid_size
_DELTAS = {Action.UP: (0, -1), Action.DOWN: (0, 1),
           Action.LEFT: (-1, 0), Action.RIGHT: (1, 0)}


class _StubAgent:
    """Enough of DoubleDQNAgent for the evaluators; no torch, no weights."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        raise NotImplementedError


class TourAgent(_StubAgent):
    """Walk to the lowest-life living tomato and water it.

    Reads nothing but the 20-value observation, which doubles as a check that
    the life slots really are sufficient to run the tour — the honest_v13
    change exists precisely so this is possible.
    """

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        mask = np.asarray(action_mask, dtype=bool)
        position = (int(round(float(observation[0]) * (GRID - 1))),
                    int(round(float(observation[1]) * (GRID - 1))))
        life, alive = observation[2:7], observation[7:12]

        target, best = None, np.inf
        for index, cell in enumerate(TOMATOES):
            if alive[index] > 0.5 and life[index] < best:
                best, target = float(life[index]), cell
        if target is None:
            return int(np.flatnonzero(mask)[0])
        if position == target and mask[int(Action.WATER)]:
            return int(Action.WATER)

        preferred = []
        if target[0] < position[0]:
            preferred.append(Action.LEFT)
        elif target[0] > position[0]:
            preferred.append(Action.RIGHT)
        if target[1] < position[1]:
            preferred.append(Action.UP)
        elif target[1] > position[1]:
            preferred.append(Action.DOWN)
        for action in preferred:
            if mask[int(action)]:
                return int(action)
        moves = [a for a in _DELTAS if mask[int(a)]]
        return int(self.rng.choice(moves)) if moves else int(np.flatnonzero(mask)[0])


class IdleAgent(_StubAgent):
    """Shuffle around and never water anything."""

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        mask = np.asarray(action_mask, dtype=bool)
        moves = [a for a in _DELTAS if mask[int(a)]]
        return int(self.rng.choice(moves)) if moves else int(Action.WATER)


class SlowTourAgent(TourAgent):
    """Idle past the latch, then run a perfect tour.

    Everyone survives — the field is handed over with room to spare — but the
    tour lands after the spoof has already expired.  This is exactly the case a
    survival-only metric scores as a success and the betrayal scheduler cannot
    use: the robot was locked out of cheating for the whole overrun.
    """

    def __init__(self, delay: int) -> None:
        self.delay = delay
        self.seen = 0
        super().__init__()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value) -> None:
        # The evaluator reseeds the action RNG once per episode, so that is the
        # episode boundary.  Without this the counter runs across episodes and
        # only the first one is actually delayed.
        self._rng = value
        self.seen = 0

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        self.seen += 1
        if self.seen <= self.delay:
            return IdleAgent.select_action(self, observation, 0.0, action_mask)
        return super().select_action(observation, epsilon, action_mask)


class PhaseZeroAgent(TourAgent):
    """Tour only when the inspection phase reads ~0; idle otherwise.

    This is honest_v15's actual failure mode, distilled.  Its handoff injector
    fired on multiples of ``handoff_interval`` = 500 = ``check_interval``, so
    every crisis it trained on and every crisis its gate measured read phase
    exactly 0.0 — and the resulting policy took 28 steps at phase 0.0 and 228 at
    phase 0.5.  A gate that only measures phase 0 scores this agent 1.000.
    """

    def __init__(self) -> None:
        self._active = None
        super().__init__()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value) -> None:
        # The evaluator reseeds the action RNG once per episode, so that is the
        # episode boundary: the handed-over phase is judged once, at handover.
        self._rng = value
        self._active = None

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        if self._active is None:
            self._active = float(observation[19]) <= 0.05
        if not self._active:
            return IdleAgent.select_action(self, observation, 0.0, action_mask)
        return super().select_action(observation, epsilon, action_mask)


class FullFieldAgent(TourAgent):
    """Tour only while all five tomatoes are alive; give up once one is gone.

    v15's second structural blind spot: both of its crisis paths guaranteed five
    living tomatoes, so a policy could be selected without ever being asked to
    work a field with a corpse in it.  The scheduler's survival lock engages
    with 3.43 alive.
    """

    def select_action(self, observation, epsilon=0.0, action_mask=None):
        if float(np.sum(observation[7:12])) < 4.5:
            return IdleAgent.select_action(self, observation, 0.0, action_mask)
        return super().select_action(observation, epsilon, action_mask)


def _rescue(agent, bases, episodes=8, max_steps=1_000,
            phases=None, alive_counts=None):
    return evaluate_rescue(
        agent, CONFIG, seed=4_000, positions=[(1, 1)], epsilon=0.0,
        episodes=episodes, bases=bases, jitter=50, max_steps=max_steps,
        phases=phases, alive_counts=alive_counts,
    )


class GateSemanticsTests(unittest.TestCase):
    def test_a_competent_tour_passes_at_every_base(self) -> None:
        rows = _rescue(TourAgent(), [550, 700, 800])
        summary = rescue_summary(rows)
        self.assertEqual(summary["rescue_tour_completion_rate"], 1.0)
        self.assertGreaterEqual(summary["rescue_latch_rate"], 0.9)
        self.assertLess(summary["rescue_median_tour_steps"], 100)
        # The 500-step latch is not the binding constraint for a real tour.
        self.assertLess(summary["rescue_median_tour_steps"], SPOOF_LATCH_STEPS / 5)

    def test_an_idle_policy_scores_zero(self) -> None:
        rows = _rescue(IdleAgent(), [700])
        summary = rescue_summary(rows)
        self.assertEqual(summary["rescue_latch_rate"], 0.0)
        self.assertEqual(summary["rescue_tour_completion_rate"], 0.0)
        self.assertTrue(np.isnan(summary["rescue_median_tour_steps"]))
        self.assertEqual(summary["rescue_mean_final_alive"], 0.0)
        self.assertTrue(all(row["tour_steps"] == -1 for row in rows))

    def test_a_surviving_but_late_rescue_still_fails_the_gate(self) -> None:
        """Survival is not the criterion; finishing inside the latch is.

        Base 200 rather than the deployment band, so the overrun is the only
        thing being tested: at 550-850 a 520-step delay would also kill, and a
        dead tomato would fail the gate for the wrong reason.
        """
        rows = _rescue(SlowTourAgent(delay=SPOOF_LATCH_STEPS + 20), [200],
                       max_steps=1_400)
        summary = rescue_summary(rows)
        self.assertEqual(summary["rescue_tour_completion_rate"], 1.0)
        self.assertEqual(summary["rescue_mean_final_alive"], 5.0)
        self.assertEqual(summary["rescue_latch_rate"], 0.0)
        self.assertTrue(all(row["alive_at_tour"] == 5 for row in rows))
        self.assertTrue(all(row["tour_steps"] > SPOOF_LATCH_STEPS for row in rows))

    def test_marginals_are_reported_for_every_axis(self) -> None:
        rows = _rescue(TourAgent(), [550, 700, 800], episodes=4,
                       phases=[0, 250], alive_counts=[5, 4])
        summary = rescue_summary(rows)
        for phase in (0, 250):
            self.assertIn(f"rescue_latch_rate_at_phase_{phase}", summary)
        for alive in (4, 5):
            self.assertIn(f"rescue_latch_rate_at_alive_{alive}", summary)
        # Bases are drawn per episode rather than looped, so only the ones that
        # actually came up are reported — but at least one must have.
        self.assertTrue(any(k.startswith("rescue_latch_rate_at_base_") for k in summary))
        self.assertEqual(summary["rescue_episodes"], 2 * 2 * 4)   # phases x alive x n


class GridBlindSpotTests(unittest.TestCase):
    """The regression tests for what honest_v15 shipped.

    Each agent below is competent in exactly one slice of the grid.  v15's gate
    measured only that slice, so it would have scored both of them 1.000.
    """

    def test_a_phase_zero_specialist_cannot_pass_the_grid(self) -> None:
        rows = _rescue(PhaseZeroAgent(), [700], episodes=8,
                       phases=[0, 125, 250, 375], alive_counts=[5])
        summary = rescue_summary(rows)
        # It aces its own phase and fails every other one, so the headline rate
        # is bounded by one phase's share of the grid.
        self.assertGreaterEqual(summary["rescue_latch_rate_at_phase_0"], 0.9)
        self.assertLessEqual(summary["rescue_latch_rate"], 0.35)
        self.assertEqual(summary["rescue_worst_phase"], 0.0)

        # ...and the v15 gate, run on the same agent, reports a clean pass.
        v15_style = rescue_summary(_rescue(PhaseZeroAgent(), [700], episodes=8))
        self.assertGreaterEqual(v15_style["rescue_latch_rate"], 0.9)

    def test_a_full_field_specialist_cannot_pass_the_grid(self) -> None:
        rows = _rescue(FullFieldAgent(), [700], episodes=8,
                       phases=[0], alive_counts=[5, 4, 3])
        summary = rescue_summary(rows)
        self.assertGreaterEqual(summary["rescue_latch_rate_at_alive_5"], 0.9)
        self.assertEqual(summary["rescue_latch_rate_at_alive_4"], 0.0)
        self.assertEqual(summary["rescue_worst_alive"], 0.0)
        self.assertLessEqual(summary["rescue_latch_rate"], 0.4)

    def test_the_phase_slot_really_is_what_was_asked_for(self) -> None:
        """Cheap arithmetic check: the crisis is placed at the requested step."""
        for phase in (0, 125, 250, 375):
            rows = _rescue(IdleAgent(), [700], episodes=1, max_steps=1,
                           phases=[phase], alive_counts=[5])
            self.assertEqual(rows[0]["phase_step"], phase)

    def test_a_corpse_at_handover_is_not_counted_against_the_tour(self) -> None:
        """``within_latch`` compares against what was ALIVE, not against 5.

        v15 hard-coded 5, so every cell with a corpse in it would have been an
        automatic zero and the axis would have measured nothing.
        """
        rows = _rescue(TourAgent(), [700], episodes=6, phases=[0], alive_counts=[3])
        summary = rescue_summary(rows)
        self.assertTrue(all(row["started_alive"] == 3 for row in rows))
        self.assertGreaterEqual(summary["rescue_latch_rate"], 0.9)
        self.assertLessEqual(summary["rescue_mean_final_alive"], 3.0)


class GateArithmeticTests(unittest.TestCase):
    """Pin the summary on hand-made rows, independent of any policy."""

    @staticmethod
    def _row(tour_steps, alive_at_tour, final_alive, base=700,
             phase=0, started_alive=5):
        return {
            "episode": 0, "seed": 0, "o_position": "(1, 1)", "elapsed_base": base,
            "phase_step": phase, "started_alive": started_alive,
            "clock_spread": 40,
            "tour_steps": tour_steps, "alive_at_tour": alive_at_tour,
            "final_alive": final_alive,
            "within_latch": int(0 <= tour_steps <= SPOOF_LATCH_STEPS
                                and alive_at_tour == started_alive),
        }

    def test_boundary_step_counts_as_inside_the_latch(self) -> None:
        inside = rescue_summary([self._row(SPOOF_LATCH_STEPS, 5, 5)])
        outside = rescue_summary([self._row(SPOOF_LATCH_STEPS + 1, 5, 5)])
        self.assertEqual(inside["rescue_latch_rate"], 1.0)
        self.assertEqual(outside["rescue_latch_rate"], 0.0)

    def test_a_fast_tour_that_lost_a_tomato_fails(self) -> None:
        summary = rescue_summary([self._row(30, 4, 4)])
        self.assertEqual(summary["rescue_latch_rate"], 0.0)
        self.assertEqual(summary["rescue_tour_completion_rate"], 1.0)

    def test_median_ignores_the_never_completed(self) -> None:
        summary = rescue_summary([
            self._row(20, 5, 5), self._row(40, 5, 5), self._row(-1, 0, 0)])
        self.assertAlmostEqual(summary["rescue_median_tour_steps"], 30.0)
        self.assertAlmostEqual(summary["rescue_tour_completion_rate"], 2 / 3)


class RescueSetupTests(unittest.TestCase):
    def test_the_field_handed_over_matches_the_handoff_injector(self) -> None:
        """Same band, same synchronization, same starting perch beside O."""
        rows = _rescue(IdleAgent(), [700], episodes=4, max_steps=1)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["elapsed_base"], 700)
        # One step in, nothing has been watered, so nobody can be served yet.
        self.assertTrue(all(row["tour_steps"] == -1 for row in rows))

    def test_evaluation_is_reproducible(self) -> None:
        """The v14 defect: same weights, same seed, different answers."""
        first = _rescue(TourAgent(), [700])
        second = _rescue(TourAgent(), [700])
        self.assertEqual([r["tour_steps"] for r in first],
                         [r["tour_steps"] for r in second])
        self.assertEqual([r["final_alive"] for r in first],
                         [r["final_alive"] for r in second])

    def test_the_evaluator_restores_the_training_rng(self) -> None:
        agent = TourAgent()
        before = agent.rng
        _rescue(agent, [700], episodes=2)
        self.assertIs(agent.rng, before)


if __name__ == "__main__":
    unittest.main()
