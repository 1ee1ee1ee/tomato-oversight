"""판단 계층과 전체 루프 통합 테스트."""

import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

from drone.config import DEFAULT, Mission
from drone.main import run
from drone.policy import Policy
from drone.state import Action, Detection, Perception, Phase, Telemetry


def telem(**kw) -> Telemetry:
    base = dict(
        armed=True, mode="GUIDED", north_m=0.0, east_m=0.0, alt_m=1.3,
        yaw_rad=0.0, battery_pct=90.0, flow_quality=200, stamp=0.0,
    )
    base.update(kw)
    return Telemetry(**base)


def seen(distance: float, bearing: float = 0.0, conf: float = 0.9) -> Perception:
    return Perception(
        detections=(
            Detection(
                "person", conf,
                x_m=distance * math.cos(bearing),
                y_m=distance * math.sin(bearing),
                z_m=0.0,
            ),
        ),
        front_m=5.0, left_m=5.0, right_m=5.0, stamp=0.0,
    )


EMPTY = Perception(detections=(), front_m=5.0, left_m=5.0, right_m=5.0, stamp=0.0)


class PhaseFlow(unittest.TestCase):
    def setUp(self):
        self.p = Policy(Mission())

    def test_starts_by_taking_off(self):
        cmd = self.p.step(telem(armed=False, alt_m=0.0), EMPTY, 0.0)
        self.assertIs(cmd.action, Action.TAKEOFF)
        self.assertIs(self.p.phase, Phase.TAKEOFF)

    def test_climbs_then_searches(self):
        self.p.step(telem(armed=False, alt_m=0.0), EMPTY, 0.0)
        self.p.step(telem(alt_m=0.5), EMPTY, 1.0)
        self.assertIs(self.p.phase, Phase.TAKEOFF)
        self.p.step(telem(alt_m=1.3), EMPTY, 2.0)
        self.assertIs(self.p.phase, Phase.SEARCH)

    def test_search_rotates_in_place(self):
        self.p.phase = Phase.SEARCH
        cmd = self.p.step(telem(), EMPTY, 0.0)
        self.assertEqual(cmd.vx, 0.0)
        self.assertGreater(cmd.yaw_rate, 0.0)

    def test_detection_switches_to_approach(self):
        self.p.phase = Phase.SEARCH
        self.p.step(telem(), seen(3.0), 0.0)
        self.assertIs(self.p.phase, Phase.APPROACH)

    def test_low_confidence_detection_is_ignored(self):
        self.p.phase = Phase.SEARCH
        self.p.step(telem(), seen(3.0, conf=0.2), 0.0)
        self.assertIs(self.p.phase, Phase.SEARCH)

    def test_search_gives_up_after_a_full_turn(self):
        self.p.phase = Phase.SEARCH
        away = telem(north_m=2.5)
        self.p.step(away, EMPTY, 0.0)
        self.p.step(away, EMPTY, 40.0)
        self.assertIs(self.p.phase, Phase.RETURN)


class Approach(unittest.TestCase):
    def setUp(self):
        self.p = Policy(Mission())
        self.p.phase = Phase.APPROACH

    def test_turns_before_advancing(self):
        """기수가 크게 틀어져 있으면 전진하지 않는다.

        옆으로 흐르며 접근하면 옵티컬 플로우 추정이 빠르게 나빠진다.
        """
        cmd = self.p.step(telem(), seen(3.0, bearing=0.9), 0.0)
        self.assertEqual(cmd.vx, 0.0)
        self.assertGreater(cmd.yaw_rate, 0.0)

    def test_advances_when_aligned(self):
        cmd = self.p.step(telem(), seen(3.0, bearing=0.02), 0.0)
        self.assertGreater(cmd.vx, 0.0)

    def test_backs_off_when_too_close(self):
        cmd = self.p.step(telem(), seen(0.8, bearing=0.0), 0.0)
        self.assertLess(cmd.vx, 0.0)

    def test_stops_at_target_distance(self):
        self.p.step(telem(), seen(1.5, bearing=0.0), 0.0)
        self.assertIs(self.p.phase, Phase.INSPECT)

    def test_brief_loss_is_tolerated(self):
        self.p.step(telem(), seen(3.0, bearing=0.02), 0.0)
        self.p.step(telem(), EMPTY, 0.5)
        self.assertIs(self.p.phase, Phase.APPROACH)

    def test_sustained_loss_returns_to_search(self):
        self.p.step(telem(), seen(3.0, bearing=0.02), 0.0)
        self.p.step(telem(), EMPTY, 5.0)
        self.assertIs(self.p.phase, Phase.SEARCH)


class Overrides(unittest.TestCase):
    def test_guard_can_force_return(self):
        p = Policy(Mission())
        p.phase = Phase.APPROACH
        p.force(Phase.RETURN, 0.0)
        self.assertIs(p.phase, Phase.RETURN)

    def test_landing_is_not_interrupted_by_return(self):
        """착륙 중에 배터리 경고가 복귀를 요구해도 되돌리지 않는다."""
        p = Policy(Mission())
        p.phase = Phase.LAND
        p.force(Phase.RETURN, 0.0)
        self.assertIs(p.phase, Phase.LAND)

    def test_mission_timeout_forces_return(self):
        p = Policy(dataclasses.replace(Mission(), max_mission_seconds=10.0))
        p.phase = Phase.SEARCH
        away = telem(north_m=2.5)
        p.step(away, EMPTY, 0.0)
        p.step(away, EMPTY, 20.0)
        self.assertIs(p.phase, Phase.RETURN)


class ReturnHome(unittest.TestCase):
    def test_lands_once_home(self):
        p = Policy(Mission())
        p.phase = Phase.RETURN
        cmd = p.step(telem(north_m=0.1, east_m=0.1), EMPTY, 0.0)
        self.assertIs(cmd.action, Action.LAND)

    def test_heads_toward_home(self):
        p = Policy(Mission())
        p.phase = Phase.RETURN
        # 북쪽 3m 지점, 기수는 남쪽(180°) → 집이 정면이다.
        cmd = p.step(telem(north_m=3.0, yaw_rad=math.pi), EMPTY, 0.0)
        self.assertGreater(cmd.vx, 0.0)
        self.assertLess(abs(cmd.yaw_rate), 0.05)


class FullMission(unittest.TestCase):
    """시뮬레이션 전체를 끝까지 돌린다."""

    def test_mission_completes_without_guard_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dataclasses.replace(
                DEFAULT,
                runtime=dataclasses.replace(
                    DEFAULT.runtime, log_path=str(Path(tmp) / "log.jsonl")
                ),
            )
            summary = run(cfg, fast=True, max_seconds=120.0)

        self.assertEqual(summary["final_phase"], "done")
        # 정상 비행에서 Guard가 개입한다면 판단 계층이 한계를 모르고 있다는 뜻이다.
        self.assertEqual(summary["vetoed_ticks"], 0)

    def test_flat_battery_forces_landing(self):
        from drone.guard import Guard
        from drone.link import MockLink
        from drone.perception import MockSource

        fc = MockLink(battery_pct=18.0)
        fc.armed = True
        fc.alt = 1.3
        guard = Guard(DEFAULT.limits)
        policy = Policy(DEFAULT.mission)
        policy.phase = Phase.SEARCH

        t = fc.telemetry()
        percep = MockSource().read(t)
        result = guard.check(policy.step(t, percep, t.stamp), t, percep, t.stamp)

        self.assertIs(result.command.action, Action.LAND)
        self.assertIn("battery_critical", result.vetoes)


if __name__ == "__main__":
    unittest.main()
