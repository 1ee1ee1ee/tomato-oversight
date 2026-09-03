"""자연어 명령 컴파일러와 임무별 동작 테스트."""

import dataclasses
import math
import tempfile
import unittest
from pathlib import Path

from drone.compiler import compile_mission, compile_rules
from drone.config import DEFAULT, Limits, Mission
from drone.main import run
from drone.mission_spec import Behavior, MissionSpec, validate
from drone.policy import Policy
from drone.state import Action, Detection, Perception, Phase, Telemetry


def telem(**kw) -> Telemetry:
    base = dict(
        armed=True, mode="GUIDED", north_m=0.0, east_m=0.0, alt_m=1.3,
        yaw_rad=0.0, battery_pct=90.0, flow_quality=200, stamp=0.0,
    )
    base.update(kw)
    return Telemetry(**base)


def seen(label="person", distance=1.5, bearing=0.0, conf=0.9) -> Perception:
    return Perception(
        detections=(
            Detection(label, conf, distance * math.cos(bearing),
                      distance * math.sin(bearing), 0.0),
        ),
        front_m=5.0, left_m=5.0, right_m=5.0, stamp=0.0,
    )


EMPTY = Perception(detections=(), front_m=5.0, left_m=5.0, right_m=5.0, stamp=0.0)


class BehaviorParsing(unittest.TestCase):
    def test_default_is_approach_and_inspect(self):
        spec, _ = compile_rules("사람 찾아서 앞에 서줘")
        self.assertIs(spec.behavior, Behavior.APPROACH_INSPECT)

    def test_counting(self):
        for text in ("의자 몇 개 있는지 세줘", "페트병 개수 세어줘", "how many chairs"):
            with self.subTest(text=text):
                spec, _ = compile_rules(text)
                self.assertIs(spec.behavior, Behavior.COUNT)

    def test_following(self):
        spec, _ = compile_rules("나 따라다녀")
        self.assertIs(spec.behavior, Behavior.FOLLOW)

    def test_patrol(self):
        spec, _ = compile_rules("한 바퀴 돌면서 뭐 있는지 살펴봐")
        self.assertIs(spec.behavior, Behavior.PATROL)


class TargetParsing(unittest.TestCase):
    def test_korean_maps_to_model_class(self):
        cases = {"의자": "chair", "페트병": "bottle", "화분": "pottedplant",
                 "강아지": "dog", "사람": "person"}
        for ko, en in cases.items():
            with self.subTest(ko=ko):
                spec, _ = compile_rules(f"{ko} 찾아줘")
                self.assertEqual(spec.target_label, en)

    def test_english_class_name_passes_through(self):
        spec, _ = compile_rules("find the bottle")
        self.assertEqual(spec.target_label, "bottle")

    def test_unknown_target_falls_back_with_a_note(self):
        spec, notes = compile_rules("저기 있는 거 찾아줘")
        self.assertEqual(spec.target_label, "person")
        self.assertTrue(any("기본값" in n for n in notes))

    def test_first_mentioned_target_wins(self):
        spec, _ = compile_rules("의자 말고 페트병 찾아줘")
        self.assertEqual(spec.target_label, "chair")


class NumberAndLengthParsing(unittest.TestCase):
    def test_arabic_count(self):
        spec, _ = compile_rules("페트병 3개 세줘")
        self.assertEqual(spec.stop_after_n, 3)

    def test_korean_numeral_count(self):
        spec, _ = compile_rules("의자 세 개 세줘")
        self.assertEqual(spec.stop_after_n, 3)

    def test_missing_count_gets_a_default_and_a_note(self):
        spec, notes = compile_rules("의자 몇 개 있는지 세줘")
        self.assertEqual(spec.stop_after_n, 3)
        self.assertTrue(any("명시되지 않아" in n for n in notes))

    def test_metres(self):
        spec, _ = compile_rules("화분 2m 앞까지 가고 1.5m 높이로 날아")
        self.assertAlmostEqual(spec.stop_distance_m, 2.0)
        self.assertAlmostEqual(spec.cruise_alt_m, 1.5)

    def test_centimetres_are_converted(self):
        """cm 를 놓치면 명령이 조용히 기본값으로 대체된다.

        사람은 30cm 를 말했는데 기체가 1.5m 에서 멈추는, 알아채기 힘든 오해석이다.
        """
        spec, _ = compile_rules("사람 앞 30cm 까지 가")
        self.assertAlmostEqual(spec.stop_distance_m, 0.3)

    def test_source_text_is_preserved(self):
        text = "의자 두 개 세줘"
        spec, _ = compile_rules(text)
        self.assertEqual(spec.source_text, text)


class PreflightValidation(unittest.TestCase):
    """이륙 전에 걸러야 할 명령들. 여기를 통과한 것만 프로펠러가 돈다."""

    def setUp(self):
        self.limits = Limits()   # min_front 1.0, max_radius 4.0, alt 0.8~2.0

    def test_sane_command_passes(self):
        self.assertEqual(compile_mission("사람 찾아서 앞에 서줘", self.limits).problems, ())

    def test_stop_distance_inside_avoidance_margin_is_rejected(self):
        result = compile_mission("사람 앞 30cm 까지 가", self.limits)
        self.assertFalse(result.ok)
        self.assertTrue(any("전방 회피 여유" in p for p in result.problems))

    def test_stop_distance_beyond_geofence_is_rejected(self):
        result = compile_mission("사람한테 50m 앞까지 가", self.limits)
        self.assertFalse(result.ok)
        self.assertTrue(any("지오펜스" in p for p in result.problems))

    def test_altitude_above_ceiling_is_rejected(self):
        result = compile_mission("사람 찾아서 5m 높이로 날아", self.limits)
        self.assertFalse(result.ok)
        self.assertTrue(any("순항 고도" in p for p in result.problems))

    def test_unreasonable_count_is_rejected(self):
        spec = MissionSpec(behavior=Behavior.COUNT, stop_after_n=20)
        self.assertTrue(any("비행시간" in p for p in validate(spec, self.limits)))

    def test_empty_target_is_rejected(self):
        self.assertTrue(
            any("대상이 지정되지" in p for p in validate(MissionSpec(target_label="  "), self.limits))
        )


class CountBehavior(unittest.TestCase):
    def setUp(self):
        self.spec = MissionSpec(behavior=Behavior.COUNT, target_label="bottle", stop_after_n=2)
        self.p = Policy(self.spec.to_mission(Mission()), spec=self.spec)

    def test_returns_to_search_after_counting_one(self):
        self.p.phase = Phase.INSPECT
        self.p._last_target = Detection("bottle", 0.9, 1.5, 0.0, 0.0)
        self.p.step(telem(), EMPTY, 0.0)
        self.p.step(telem(), EMPTY, 10.0)
        self.assertEqual(len(self.p.counted), 1)
        self.assertIs(self.p.phase, Phase.SEARCH)

    def test_returns_home_once_quota_is_met(self):
        self.p.counted = [(9.0, 9.0)]           # 이미 하나 셌다
        self.p.phase = Phase.INSPECT
        self.p._last_target = Detection("bottle", 0.9, 1.5, 0.0, 0.0)
        self.p.step(telem(), EMPTY, 0.0)
        self.p.step(telem(north_m=2.0), EMPTY, 10.0)
        self.assertEqual(len(self.p.counted), 2)
        self.assertIs(self.p.phase, Phase.RETURN)

    def test_same_target_is_not_counted_twice(self):
        """이미 센 자리에 있는 대상은 건너뛴다. 안 그러면 한 개를 무한히 센다."""
        self.p.counted = [(1.5, 0.0)]           # 기수 0°, 1.5m 앞에서 이미 셌다
        self.p.phase = Phase.SEARCH
        self.p.step(telem(), seen("bottle", 1.5), 0.0)
        self.assertIs(self.p.phase, Phase.SEARCH)

    def test_a_different_target_is_counted(self):
        self.p.counted = [(1.5, 0.0)]
        self.p.phase = Phase.SEARCH
        self.p.step(telem(), seen("bottle", 3.5), 0.0)   # 2m 떨어진 다른 개체
        self.assertIs(self.p.phase, Phase.APPROACH)


class FollowBehavior(unittest.TestCase):
    def test_does_not_finish_at_stop_distance(self):
        spec = MissionSpec(behavior=Behavior.FOLLOW)
        p = Policy(spec.to_mission(Mission()), spec=spec)
        p.phase = Phase.APPROACH
        cmd = p.step(telem(), seen("person", 1.5), 0.0)
        self.assertIs(p.phase, Phase.APPROACH)       # INSPECT 로 넘어가지 않는다
        self.assertIs(cmd.action, Action.HOLD)
        self.assertIn("추종", cmd.reason)

    def test_still_ends_on_mission_timeout(self):
        spec = MissionSpec(behavior=Behavior.FOLLOW)
        mission = dataclasses.replace(spec.to_mission(Mission()), max_mission_seconds=5.0)
        p = Policy(mission, spec=spec)
        p.phase = Phase.APPROACH
        p.step(telem(north_m=2.0), seen("person", 1.5), 0.0)
        p.step(telem(north_m=2.0), seen("person", 1.5), 10.0)
        self.assertIs(p.phase, Phase.RETURN)


class PatrolBehavior(unittest.TestCase):
    def test_records_without_approaching(self):
        spec = MissionSpec(behavior=Behavior.PATROL)
        p = Policy(spec.to_mission(Mission()), spec=spec)
        p.phase = Phase.SEARCH
        cmd = p.step(telem(), seen("person", 2.0), 0.0)
        self.assertIs(p.phase, Phase.SEARCH)          # 다가가지 않는다
        self.assertIn("person", p.observed)
        self.assertGreater(cmd.yaw_rate, 0.0)         # 계속 회전한다

    def test_does_not_record_duplicates(self):
        spec = MissionSpec(behavior=Behavior.PATROL)
        p = Policy(spec.to_mission(Mission()), spec=spec)
        p.phase = Phase.SEARCH
        for t in (0.0, 0.5, 1.0):
            p.step(telem(), seen("person", 2.0), t)
        self.assertEqual(p.observed, ["person"])


class CompiledMissionFlies(unittest.TestCase):
    """컴파일된 명령이 실제로 다른 비행을 만들어내는지 끝까지 확인한다."""

    def _fly(self, order: str, max_seconds: float = 200.0) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dataclasses.replace(
                DEFAULT,
                runtime=dataclasses.replace(
                    DEFAULT.runtime, log_path=str(Path(tmp) / "log.jsonl")
                ),
            )
            result = compile_mission(order, cfg.limits)
            self.assertTrue(result.ok, result.problems)
            return run(cfg, fast=True, max_seconds=max_seconds, spec=result.spec)

    def test_approach_mission_completes(self):
        summary = self._fly("사람 찾아서 앞에 가서 서줘")
        self.assertEqual(summary["final_phase"], "done")
        self.assertEqual(summary["vetoed_ticks"], 0)

    def test_patrol_records_what_it_sees_without_approaching(self):
        summary = self._fly("한 바퀴 돌면서 뭐 있는지 살펴봐")
        self.assertEqual(summary["final_phase"], "done")
        self.assertIn("person", summary["observed"])
        self.assertEqual(summary["counted"], 0)

    def test_count_mission_counts_the_one_target_present(self):
        summary = self._fly("사람 2명 세줘")
        self.assertEqual(summary["final_phase"], "done")
        # 모의 방에는 대상이 하나뿐이다. 두 번 세지 않고 정직하게 1 로 끝나야 한다.
        self.assertEqual(summary["counted"], 1)


if __name__ == "__main__":
    unittest.main()
