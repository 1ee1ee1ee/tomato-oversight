"""좌·우·후방 ToF 배열과 Guard 후방 차단 테스트.

전방은 OAK-D 뎁스가 담당하므로 여기서는 옆과 뒤만 다룬다.
VL53L1X 실물 없이 검증 가능한 부분(값 정제, 합성, Guard 반응)을 고정한다.
"""

import unittest

from drone.config import Limits
from drone.guard import Guard
from drone.perception import WithRangefinders
from drone.rangefinders import (
    Clearances,
    FixedRangefinders,
    NullRangefinders,
    _sane,
    build,
)
from drone.state import Action, Command, Perception, Telemetry

NOW = 100.0


def telem(**kw) -> Telemetry:
    base = dict(
        armed=True, mode="GUIDED", north_m=0.0, east_m=0.0, alt_m=1.3,
        yaw_rad=0.0, battery_pct=90.0, flow_quality=200, stamp=NOW,
    )
    base.update(kw)
    return Telemetry(**base)


class ValueSanity(unittest.TestCase):
    """범위를 벗어난 값은 지어내지 않고 버린다."""

    def test_in_range_passes(self):
        self.assertAlmostEqual(_sane(1.5), 1.5)

    def test_none_stays_none(self):
        self.assertIsNone(_sane(None))

    def test_beyond_max_range_is_dropped(self):
        self.assertIsNone(_sane(9.0))       # VL53L1X 는 4m 가 한계

    def test_below_min_is_dropped(self):
        self.assertIsNone(_sane(0.0))       # 0 은 '벽에 붙었다'가 아니라 측정 실패

    def test_boundaries_are_inclusive(self):
        self.assertIsNotNone(_sane(0.04))
        self.assertIsNotNone(_sane(4.0))


class Backends(unittest.TestCase):
    def test_null_returns_all_none(self):
        self.assertEqual(NullRangefinders().read(), Clearances())

    def test_fixed_sanitises_on_construction(self):
        r = FixedRangefinders(left_m=1.0, right_m=99.0, back_m=None).read()
        self.assertAlmostEqual(r.left_m, 1.0)
        self.assertIsNone(r.right_m)        # 99m 는 버려진다
        self.assertIsNone(r.back_m)

    def test_build_none(self):
        self.assertIsInstance(build("none"), NullRangefinders)

    def test_build_rejects_unknown(self):
        with self.assertRaises(ValueError):
            build("lidar-9000")


class _StubSource:
    def __init__(self, percep):
        self.percep = percep
        self.closed = False

    def read(self, telem):
        return self.percep

    def close(self):
        self.closed = True


class Composition(unittest.TestCase):
    def setUp(self):
        self.base = Perception(front_m=3.0, forward_blind=False, stamp=NOW)

    def test_fills_sides_and_rear(self):
        src = WithRangefinders(_StubSource(self.base), FixedRangefinders(1.0, 2.0, 0.5))
        out = src.read(telem())
        self.assertAlmostEqual(out.left_m, 1.0)
        self.assertAlmostEqual(out.right_m, 2.0)
        self.assertAlmostEqual(out.back_m, 0.5)

    def test_does_not_touch_forward(self):
        """전방은 OAK-D 뎁스의 몫이다. 거리계가 덮어쓰면 안 된다."""
        src = WithRangefinders(_StubSource(self.base), FixedRangefinders(1.0, 1.0, 1.0))
        out = src.read(telem())
        self.assertAlmostEqual(out.front_m, 3.0)
        self.assertFalse(out.forward_blind)

    def test_passes_through_none_frame(self):
        src = WithRangefinders(_StubSource(None), FixedRangefinders(1.0, 1.0, 1.0))
        self.assertIsNone(src.read(telem()))

    def test_close_closes_inner(self):
        inner = _StubSource(self.base)
        WithRangefinders(inner, NullRangefinders()).close()
        self.assertTrue(inner.closed)


class GuardBlocksReverse(unittest.TestCase):
    """후진은 지금까지 전혀 안 막혔다. Policy 가 실제로 후진하므로 구멍이었다."""

    def setUp(self):
        self.g = Guard(Limits())     # min_side_m = 0.5

    def _percep(self, **kw) -> Perception:
        base = dict(detections=(), front_m=5.0, left_m=5.0, right_m=5.0,
                    back_m=5.0, forward_blind=False, stamp=NOW)
        base.update(kw)
        return Perception(**base)

    def test_rear_obstacle_blocks_reverse(self):
        cmd = Command(Action.MOVE, vx=-0.4)
        r = self.g.check(cmd, telem(), self._percep(back_m=0.2), NOW)
        self.assertEqual(r.command.vx, 0.0)
        self.assertIn("obstacle_back", r.vetoes)

    def test_rear_obstacle_does_not_block_forward(self):
        cmd = Command(Action.MOVE, vx=0.4)
        r = self.g.check(cmd, telem(), self._percep(back_m=0.2), NOW)
        self.assertAlmostEqual(r.command.vx, 0.4)

    def test_no_rear_sensor_does_not_block(self):
        cmd = Command(Action.MOVE, vx=-0.4)
        r = self.g.check(cmd, telem(), self._percep(back_m=None), NOW)
        self.assertAlmostEqual(r.command.vx, -0.4)

    def test_clear_rear_allows_reverse(self):
        cmd = Command(Action.MOVE, vx=-0.4)
        r = self.g.check(cmd, telem(), self._percep(back_m=3.0), NOW)
        self.assertAlmostEqual(r.command.vx, -0.4)
        self.assertEqual(r.vetoes, ())

    def test_blind_forward_escape_still_respects_rear(self):
        """전방을 못 봐서 후진으로 빠져나가려는데 뒤에도 벽이면 둘 다 막는다."""
        cmd = Command(Action.MOVE, vx=-0.4, yaw_rate=0.3)
        r = self.g.check(
            cmd, telem(), self._percep(forward_blind=True, back_m=0.2), NOW
        )
        self.assertEqual(r.command.vx, 0.0)
        self.assertAlmostEqual(r.command.yaw_rate, 0.3)   # 회전으로는 빠져나갈 수 있다


if __name__ == "__main__":
    unittest.main()
