"""안전 검증기 테스트.

Guard는 이 프로젝트에서 유일하게 '틀리면 기체가 부서지는' 코드다.
따라서 하드웨어 없이 전부 검증 가능하도록 설계했고, 여기서 고정한다.
"""

import math
import time
import unittest

from drone.config import Limits
from drone.guard import Guard, body_to_ned, ned_to_body
from drone.state import Action, Command, Detection, Perception, Phase, Telemetry


def telem(**kw) -> Telemetry:
    base = dict(
        armed=True, mode="GUIDED", north_m=0.0, east_m=0.0, alt_m=1.3,
        yaw_rad=0.0, battery_pct=90.0, flow_quality=200, stamp=100.0,
    )
    base.update(kw)
    return Telemetry(**base)


def percep(**kw) -> Perception:
    base = dict(detections=(), front_m=5.0, left_m=5.0, right_m=5.0, stamp=100.0)
    base.update(kw)
    return Perception(**base)


NOW = 100.0
MOVE = Command(Action.MOVE, vx=0.5)


class FrameConversion(unittest.TestCase):
    def test_round_trip(self):
        for yaw in (0.0, 0.7, math.pi, -1.9):
            vn, ve = body_to_ned(0.3, -0.2, yaw)
            vx, vy = ned_to_body(vn, ve, yaw)
            self.assertAlmostEqual(vx, 0.3, places=9)
            self.assertAlmostEqual(vy, -0.2, places=9)

    def test_yaw_zero_is_identity(self):
        self.assertEqual(body_to_ned(1.0, 2.0, 0.0), (1.0, 2.0))

    def test_yaw_90_maps_forward_to_east(self):
        vn, ve = body_to_ned(1.0, 0.0, math.pi / 2)
        self.assertAlmostEqual(vn, 0.0, places=9)
        self.assertAlmostEqual(ve, 1.0, places=9)


class Watchdogs(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())

    def test_missing_telemetry_holds(self):
        r = self.g.check(MOVE, None, percep(), NOW)
        self.assertIs(r.command.action, Action.HOLD)
        self.assertIn("telemetry_stale", r.vetoes)

    def test_stale_telemetry_holds(self):
        r = self.g.check(MOVE, telem(stamp=NOW - 3.0), percep(), NOW)
        self.assertIs(r.command.action, Action.HOLD)

    def test_stale_perception_blocks_forward_only(self):
        r = self.g.check(
            Command(Action.MOVE, vx=0.5, vy=0.3), telem(), percep(stamp=NOW - 5.0), NOW
        )
        self.assertEqual(r.command.vx, 0.0)
        self.assertAlmostEqual(r.command.vy, 0.3)   # 옆으로는 갈 수 있다
        self.assertIn("perception_stale", r.vetoes)

    def test_reverse_allowed_when_blind(self):
        r = self.g.check(
            Command(Action.MOVE, vx=-0.4), telem(), percep(stamp=NOW - 5.0), NOW
        )
        self.assertAlmostEqual(r.command.vx, -0.4)


class Battery(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())

    def test_critical_forces_land_over_everything(self):
        r = self.g.check(MOVE, telem(battery_pct=15.0), percep(), NOW)
        self.assertIs(r.command.action, Action.LAND)
        self.assertIs(r.force_phase, Phase.LAND)

    def test_low_requests_return_without_blocking(self):
        r = self.g.check(MOVE, telem(battery_pct=30.0), percep(), NOW)
        self.assertIs(r.force_phase, Phase.RETURN)
        self.assertGreater(r.command.vx, 0.0)   # 명령 자체는 살아 있다

    def test_healthy_battery_is_silent(self):
        r = self.g.check(MOVE, telem(battery_pct=90.0), percep(), NOW)
        self.assertIsNone(r.force_phase)
        self.assertEqual(r.vetoes, ())


class OpticalFlow(unittest.TestCase):
    """실내 자율에서 가장 자주 터지는 실패 모드."""

    def setUp(self):
        self.g = Guard(Limits())

    def test_low_quality_blocks_movement(self):
        r = self.g.check(MOVE, telem(flow_quality=10), percep(), NOW)
        self.assertIs(r.command.action, Action.HOLD)
        self.assertIn("flow_quality_low", r.vetoes)

    def test_low_quality_still_allows_landing(self):
        r = self.g.check(Command(Action.LAND), telem(flow_quality=10), percep(), NOW)
        self.assertIs(r.command.action, Action.LAND)


class Obstacles(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())

    def test_front_rangefinder_blocks_forward(self):
        r = self.g.check(MOVE, telem(), percep(front_m=0.4), NOW)
        self.assertEqual(r.command.vx, 0.0)
        self.assertIn("obstacle_front", r.vetoes)

    def test_guard_ignores_model_output_and_trusts_rangefinder(self):
        """설계 원칙 고정: Guard는 인식 모델의 보고가 아니라 센서 원본을 본다.

        모델이 코앞에 사람이 있다고 보고해도, 거리계가 비어 있으면 막지 않는다.
        반대로 모델이 아무것도 못 봐도 거리계가 가깝다면 막는다.
        모델을 안전 판정에 넣으면 모델이 틀렸을 때 알아낼 방법이 사라진다.
        """
        close_call = Perception(
            detections=(Detection("person", 0.99, 0.2, 0.0, 0.0),),
            front_m=5.0, left_m=5.0, right_m=5.0, stamp=NOW,
        )
        r = self.g.check(MOVE, telem(), close_call, NOW)
        self.assertGreater(r.command.vx, 0.0)
        self.assertNotIn("obstacle_front", r.vetoes)

        blind_wall = Perception(detections=(), front_m=0.3, left_m=5.0, right_m=5.0, stamp=NOW)
        r = self.g.check(MOVE, telem(), blind_wall, NOW)
        self.assertEqual(r.command.vx, 0.0)

    def test_side_clearance(self):
        r = self.g.check(Command(Action.MOVE, vy=0.4), telem(), percep(right_m=0.2), NOW)
        self.assertEqual(r.command.vy, 0.0)
        r = self.g.check(Command(Action.MOVE, vy=-0.4), telem(), percep(right_m=0.2), NOW)
        self.assertAlmostEqual(r.command.vy, -0.4)   # 반대쪽은 열려 있다

    def test_missing_rangefinder_does_not_block(self):
        r = self.g.check(MOVE, telem(), percep(front_m=None), NOW)
        self.assertGreater(r.command.vx, 0.0)


class Geofence(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())      # max_radius_m = 4.0

    def test_inside_is_untouched(self):
        r = self.g.check(MOVE, telem(north_m=1.0), percep(), NOW)
        self.assertAlmostEqual(r.command.vx, 0.5)
        self.assertEqual(r.vetoes, ())

    def test_outward_component_removed_at_edge(self):
        r = self.g.check(MOVE, telem(north_m=3.9), percep(), NOW)
        self.assertAlmostEqual(r.command.vx, 0.0, places=9)
        self.assertIn("geofence", r.vetoes)

    def test_inward_still_allowed_at_edge(self):
        """경계에서 전체를 막으면 기체가 벽에 갇힌다. 돌아올 길은 열어둔다."""
        r = self.g.check(Command(Action.MOVE, vx=-0.5), telem(north_m=3.9), percep(), NOW)
        self.assertAlmostEqual(r.command.vx, -0.5, places=9)

    def test_tangential_preserved(self):
        r = self.g.check(Command(Action.MOVE, vy=0.5), telem(north_m=3.9), percep(), NOW)
        self.assertAlmostEqual(r.command.vy, 0.5, places=9)

    def test_diagonal_keeps_only_tangential_part(self):
        r = self.g.check(
            Command(Action.MOVE, vx=0.4, vy=0.4), telem(north_m=3.9), percep(), NOW
        )
        self.assertAlmostEqual(r.command.vx, 0.0, places=9)
        self.assertAlmostEqual(r.command.vy, 0.4, places=9)

    def test_works_with_rotated_heading(self):
        # 기수가 동쪽(yaw=90°), 위치는 동쪽 경계. 전진이 곧 바깥 방향이다.
        r = self.g.check(
            MOVE, telem(north_m=0.0, east_m=3.9, yaw_rad=math.pi / 2), percep(), NOW
        )
        self.assertAlmostEqual(r.command.vx, 0.0, places=9)


class Altitude(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())      # 0.8 ~ 2.0 m

    def test_floor_blocks_descent(self):
        r = self.g.check(Command(Action.MOVE, vz=0.3), telem(alt_m=0.5), percep(), NOW)
        self.assertEqual(r.command.vz, 0.0)
        self.assertIn("alt_floor", r.vetoes)

    def test_ceiling_blocks_climb(self):
        r = self.g.check(Command(Action.MOVE, vz=-0.3), telem(alt_m=2.4), percep(), NOW)
        self.assertEqual(r.command.vz, 0.0)
        self.assertIn("alt_ceiling", r.vetoes)

    def test_land_ignores_floor(self):
        r = self.g.check(Command(Action.LAND, vz=0.3), telem(alt_m=0.2), percep(), NOW)
        self.assertIs(r.command.action, Action.LAND)
        self.assertEqual(r.vetoes, ())


class Envelope(unittest.TestCase):
    def setUp(self):
        self.g = Guard(Limits())

    def test_speed_clamped(self):
        r = self.g.check(
            Command(Action.MOVE, vx=9.0, vy=-9.0, vz=9.0, yaw_rate=9.0),
            telem(), percep(), NOW,
        )
        self.assertAlmostEqual(r.command.vx, 0.6)
        self.assertAlmostEqual(r.command.vy, -0.6)
        self.assertAlmostEqual(r.command.vz, 0.4)
        self.assertAlmostEqual(r.command.yaw_rate, 0.5)
        self.assertIn("speed_clamped", r.vetoes)

    def test_disarmed_rejects_everything_but_takeoff(self):
        r = self.g.check(MOVE, telem(armed=False), percep(), NOW)
        self.assertIs(r.command.action, Action.HOLD)
        r = self.g.check(Command(Action.TAKEOFF, target_alt_m=1.3), telem(armed=False), percep(), NOW)
        self.assertIs(r.command.action, Action.TAKEOFF)

    def test_nominal_command_passes_untouched(self):
        """정상 비행에서 Guard는 조용해야 한다. 상시 켜진 경고는 경고가 아니다."""
        cmd = Command(Action.MOVE, vx=0.4, yaw_rate=0.2, vz=0.05)
        r = self.g.check(cmd, telem(), percep(), NOW)
        self.assertEqual(r.vetoes, ())
        self.assertAlmostEqual(r.command.vx, 0.4)


if __name__ == "__main__":
    unittest.main()
