"""OAK-D 스테레오 뎁스를 전방 거리계로 쓰는 경로 테스트.

별도 ToF 센서를 사지 않는 대신 뎁스맵을 쓰기로 했으므로,
이 경로가 죽어 있으면 실기체에는 전방 감지가 아예 없는 셈이 된다.
"""

import unittest

import numpy as np

from drone.config import Limits
from drone.guard import Guard
from drone.perception import OakDSource
from drone.state import Action, Command, Perception, Telemetry

NOW = 100.0


def telem(**kw) -> Telemetry:
    base = dict(
        armed=True, mode="GUIDED", north_m=0.0, east_m=0.0, alt_m=1.3,
        yaw_rad=0.0, battery_pct=90.0, flow_quality=200, stamp=NOW,
    )
    base.update(kw)
    return Telemetry(**base)


def depth_frame(mm: int, shape=(200, 320)) -> np.ndarray:
    return np.full(shape, mm, dtype=np.uint16)


class ForwardClearance(unittest.TestCase):
    def test_uniform_wall_reports_its_distance(self):
        dist, blind = OakDSource._forward_clearance(depth_frame(2500))
        self.assertFalse(blind)
        self.assertAlmostEqual(dist, 2.5, places=2)

    def test_close_wall_reports_close(self):
        dist, blind = OakDSource._forward_clearance(depth_frame(400))
        self.assertFalse(blind)
        self.assertAlmostEqual(dist, 0.4, places=2)

    def test_all_zero_depth_is_blind_not_zero_distance(self):
        """뎁스가 통째로 안 잡히면 '0m 앞에 벽'이 아니라 '못 본다'이다."""
        dist, blind = OakDSource._forward_clearance(depth_frame(0))
        self.assertTrue(blind)
        self.assertIsNone(dist)

    def test_out_of_range_depth_is_blind(self):
        dist, blind = OakDSource._forward_clearance(depth_frame(50_000))
        self.assertTrue(blind)
        self.assertIsNone(dist)

    def test_sparse_valid_pixels_are_blind(self):
        frame = depth_frame(0)
        frame[100, 100:110] = 1500          # 유효 픽셀 10개뿐
        dist, blind = OakDSource._forward_clearance(frame)
        self.assertTrue(blind)
        self.assertIsNone(dist)

    def test_single_noise_pixel_does_not_dominate(self):
        """최솟값을 쓰면 튀는 픽셀 하나에 기체가 멈춘다. 퍼센타일을 쓰는 이유."""
        frame = depth_frame(3000)
        frame[100, 160] = 250               # 노이즈 한 점
        dist, blind = OakDSource._forward_clearance(frame)
        self.assertFalse(blind)
        self.assertGreater(dist, 2.5)

    def test_real_obstacle_patch_is_detected(self):
        """반대로 실제 물체(넓은 면적)는 확실히 잡아야 한다."""
        frame = depth_frame(3000)
        frame[60:140, 80:240] = 700         # 눈앞 0.7m 짜리 물체
        dist, blind = OakDSource._forward_clearance(frame)
        self.assertFalse(blind)
        self.assertLess(dist, 1.0)

    def test_edges_are_ignored(self):
        """가장자리에는 프롭과 기체 프레임이 걸린다. 중앙만 본다."""
        frame = depth_frame(3000)
        frame[:, :60] = 300                 # 왼쪽 끝에 프롭
        frame[:, 260:] = 300                # 오른쪽 끝에 프롭
        dist, blind = OakDSource._forward_clearance(frame)
        self.assertFalse(blind)
        self.assertGreater(dist, 2.5)


class BlindBlocksForwardMotion(unittest.TestCase):
    """센서 미장착(None)과 눈 감음(blind)은 다르게 취급해야 한다."""

    def setUp(self):
        self.g = Guard(Limits())
        self.move = Command(Action.MOVE, vx=0.5)

    def _percep(self, **kw) -> Perception:
        base = dict(detections=(), front_m=None, left_m=None, right_m=None,
                    forward_blind=False, stamp=NOW)
        base.update(kw)
        return Perception(**base)

    def test_no_sensor_fitted_does_not_block(self):
        r = self.g.check(self.move, telem(), self._percep(), NOW)
        self.assertGreater(r.command.vx, 0.0)

    def test_blind_blocks_forward(self):
        r = self.g.check(self.move, telem(), self._percep(forward_blind=True), NOW)
        self.assertEqual(r.command.vx, 0.0)
        self.assertIn("forward_blind", r.vetoes)

    def test_blind_still_allows_reverse_and_yaw(self):
        """멈추되 갇히지는 않게 한다. 후진과 회전으로 빠져나올 수 있어야 한다."""
        cmd = Command(Action.MOVE, vx=-0.4, yaw_rate=0.3)
        r = self.g.check(cmd, telem(), self._percep(forward_blind=True), NOW)
        self.assertAlmostEqual(r.command.vx, -0.4)
        self.assertAlmostEqual(r.command.yaw_rate, 0.3)

    def test_blind_still_allows_landing(self):
        r = self.g.check(Command(Action.LAND), telem(), self._percep(forward_blind=True), NOW)
        self.assertIs(r.command.action, Action.LAND)


if __name__ == "__main__":
    unittest.main()
