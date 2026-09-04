"""온보드 판단 계층.

인식 결과를 보고 "다음에 뭘 할지"를 정한다. 여기서 나온 명령은
그대로 실행되지 않는다. 반드시 Guard를 통과해야 한다.

의도적으로 상태기계로 짰다. 학습된 정책을 실내 실기체에 바로 올리면
왜 그렇게 움직였는지 설명할 수 없고, 실내에는 그 실수를 흡수할 여유 공간이
없다. 신경망은 인식(perception)에서 일하고, 판단은 읽을 수 있게 남겨둔다.
학습 정책이나 소형 VLM으로 바꾸고 싶다면 ``step()`` 하나만 교체하면 된다.
"""

from __future__ import annotations

import math

from .config import Mission
from .mission_spec import Behavior, MissionSpec
from .state import Action, Command, Detection, Perception, Phase, Telemetry


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Policy:
    #: COUNT 에서 같은 대상을 두 번 세지 않기 위한 반경(m).
    DEDUP_RADIUS_M = 1.0

    def __init__(
        self,
        mission: Mission,
        cruise_alt_m: float | None = None,
        max_yaw_rate: float = 0.5,
        spec: MissionSpec | None = None,
    ) -> None:
        self.m = mission
        # spec 은 컴파일된 자연어 명령이다. 없으면 기본 임무로 동작한다.
        self.spec = spec or MissionSpec()
        #: COUNT 로 이미 센 대상들의 월드 좌표. 중복 계수를 막는다.
        self.counted: list[tuple[float, float]] = []
        #: PATROL 이 관측한 라벨들.
        self.observed: list[str] = []
        self.cruise_alt_m = cruise_alt_m if cruise_alt_m is not None else mission.cruise_alt_m
        # Guard와 같은 한계를 알고 있어야 한다. 매 틱 잘려 나가는 명령을 내면
        # Guard 경고가 상시 켜지고, 그러면 진짜 경고를 아무도 못 본다.
        self.max_yaw_rate = max_yaw_rate
        self.phase = Phase.IDLE
        self._phase_since: float | None = None
        self._mission_start: float | None = None
        self._target_seen_at: float | None = None
        self._last_target: Detection | None = None

    # ------------------------------------------------------------------

    @property
    def phase_elapsed(self) -> float:
        return 0.0 if self._phase_since is None else self._now - self._phase_since

    def _enter(self, phase: Phase, now: float) -> None:
        if phase is not self.phase:
            self.phase = phase
            self._phase_since = now

    def force(self, phase: Phase, now: float) -> None:
        """Guard가 단계 전환을 요구했을 때 호출한다."""
        # 이미 착륙 중이면 복귀로 되돌리지 않는다.
        if self.phase in (Phase.LAND, Phase.DONE) and phase is Phase.RETURN:
            return
        self._enter(phase, now)

    # ------------------------------------------------------------------

    def step(self, telem: Telemetry, percep: Perception | None, now: float) -> Command:
        self._now = now
        if self._mission_start is None:
            self._mission_start = now
        if self._phase_since is None:
            self._phase_since = now

        target = percep.best(self.m.target_label, self.m.min_confidence) if percep else None
        if target is not None:
            self._last_target = target
            self._target_seen_at = now

        # 임무 제한 시간 — 배터리와 별개인 두 번째 방어선.
        if (
            now - self._mission_start > self.m.max_mission_seconds
            and self.phase not in (Phase.RETURN, Phase.LAND, Phase.DONE)
        ):
            self._enter(Phase.RETURN, now)

        handler = {
            Phase.IDLE: self._idle,
            Phase.TAKEOFF: self._takeoff,
            Phase.SEARCH: self._search,
            Phase.APPROACH: self._approach,
            Phase.INSPECT: self._inspect,
            Phase.RETURN: self._return,
            Phase.LAND: self._land,
            Phase.DONE: self._done,
        }[self.phase]
        return handler(telem, target, now)

    # --- 단계별 동작 --------------------------------------------------

    def _idle(self, telem: Telemetry, target, now: float) -> Command:
        self._enter(Phase.TAKEOFF, now)
        return Command(
            Action.TAKEOFF,
            target_alt_m=self.cruise_alt_m,
            reason=f"{self.cruise_alt_m}m 이륙",
        )

    def _takeoff(self, telem: Telemetry, target, now: float) -> Command:
        if telem.alt_m >= self.cruise_alt_m - 0.15:
            self._enter(Phase.SEARCH, now)
            return self._search(telem, target, now)
        return Command(
            Action.TAKEOFF, target_alt_m=self.cruise_alt_m, reason="상승 중"
        )

    def _search(self, telem: Telemetry, target, now: float) -> Command:
        if target is not None and self.spec.behavior is Behavior.PATROL:
            # 순찰은 다가가지 않는다. 보이는 것을 기록만 하고 계속 돈다.
            if target.label not in self.observed:
                self.observed.append(target.label)
            target = None

        if target is not None and self.spec.behavior is Behavior.COUNT:
            if self._already_counted(telem, target):
                target = None   # 방금 센 대상이다. 다음 것을 찾아 계속 회전한다.

        if target is not None:
            self._enter(Phase.APPROACH, now)
            return self._approach(telem, target, now)

        # 한 바퀴 돌고도 못 찾으면 복귀한다. 무한 회전을 막는 조건이다.
        full_turn_s = (2 * math.pi) / max(1e-6, self.m.search_yaw_rate)
        if self.phase_elapsed > full_turn_s * 1.2:
            self._enter(Phase.RETURN, now)
            return self._return(telem, target, now)

        return Command(
            Action.MOVE,
            yaw_rate=self.m.search_yaw_rate,
            vz=self._alt_hold(telem),
            reason="탐색 회전",
        )

    def _approach(self, telem: Telemetry, target, now: float) -> Command:
        det = target or self._last_target
        # `or now` 를 쓰면 안 된다. _target_seen_at 이 정확히 0.0 일 때
        # 0.0 은 falsy 라 유실 타이머가 영원히 돌지 않고, 기체는 보이지도
        # 않는 대상을 향해 계속 전진한다.
        lost_for = 0.0 if self._target_seen_at is None else now - self._target_seen_at

        if det is None or lost_for > self.m.lost_target_grace_s:
            self._enter(Phase.SEARCH, now)
            return Command(Action.HOLD, reason="대상 상실 — 재탐색")

        bearing = math.atan2(det.y_m, det.x_m)
        yaw_rate = _clamp(1.2 * bearing, -self.max_yaw_rate, self.max_yaw_rate)

        error = det.x_m - self.m.stop_distance_m
        if abs(error) <= self.m.approach_tolerance_m and abs(bearing) < 0.15:
            if self.spec.behavior is Behavior.FOLLOW:
                # 추종은 끝나지 않는다. 거리를 유지한 채 계속 따라간다.
                # 종료는 배터리나 임무 제한 시간이 결정한다.
                return Command(
                    Action.HOLD,
                    yaw_rate=yaw_rate,
                    vz=self._alt_hold(telem),
                    reason=f"추종 중 {det.x_m:.2f}m",
                )
            self._enter(Phase.INSPECT, now)
            return Command(Action.HOLD, vz=self._alt_hold(telem), reason="정지 거리 도달")

        # 기수가 대상을 향하기 전에는 전진하지 않는다.
        # 옆으로 흐르며 접근하면 옵티컬 플로우 추정이 빠르게 나빠진다.
        vx = 0.0 if abs(bearing) > 0.35 else _clamp(0.5 * error, -0.4, 0.5)

        return Command(
            Action.MOVE,
            vx=vx,
            yaw_rate=yaw_rate,
            vz=self._alt_hold(telem),
            reason=f"접근 {det.x_m:.2f}m (오차 {error:+.2f})",
        )

    def _inspect(self, telem: Telemetry, target, now: float) -> Command:
        if self.phase_elapsed >= self.m.inspect_seconds:
            if self.spec.behavior is Behavior.COUNT:
                det = target or self._last_target
                if det is not None:
                    self.counted.append(self._target_world(telem, det))
                if len(self.counted) < self.spec.stop_after_n:
                    self._last_target = None
                    self._target_seen_at = None
                    self._enter(Phase.SEARCH, now)
                    return Command(
                        Action.HOLD,
                        reason=f"{len(self.counted)}/{self.spec.stop_after_n} 계수 — 재탐색",
                    )
            self._enter(Phase.RETURN, now)
            return self._return(telem, target, now)
        return Command(
            Action.HOLD, vz=self._alt_hold(telem), reason="관찰 중"
        )

    def _return(self, telem: Telemetry, target, now: float) -> Command:
        dist = telem.distance_from_home_m
        if dist < 0.4:
            self._enter(Phase.LAND, now)
            return Command(Action.LAND, reason="복귀 완료 — 착륙")

        # 집 방향을 기체 기준으로 환산
        c, s = math.cos(telem.yaw_rad), math.sin(telem.yaw_rad)
        dn, de = -telem.north_m, -telem.east_m
        fwd = dn * c + de * s
        right = -dn * s + de * c
        bearing = math.atan2(right, fwd)

        yaw_rate = _clamp(1.2 * bearing, -self.max_yaw_rate, self.max_yaw_rate)
        vx = 0.0 if abs(bearing) > 0.35 else _clamp(0.5 * dist, 0.0, 0.5)

        return Command(
            Action.MOVE,
            vx=vx,
            yaw_rate=yaw_rate,
            vz=self._alt_hold(telem),
            reason=f"복귀 {dist:.2f}m",
        )

    def _land(self, telem: Telemetry, target, now: float) -> Command:
        if not telem.armed or telem.alt_m < 0.1:
            self._enter(Phase.DONE, now)
            return Command(Action.HOLD, reason="착륙 완료")
        return Command(Action.LAND, reason="착륙 중")

    def _done(self, telem: Telemetry, target, now: float) -> Command:
        return Command(Action.HOLD, reason="임무 종료")

    # ------------------------------------------------------------------

    def _target_world(self, telem: Telemetry, det: Detection) -> tuple[float, float]:
        """기체 기준 검출 위치를 시작 지점 기준 월드 좌표로 옮긴다."""
        c, s = math.cos(telem.yaw_rad), math.sin(telem.yaw_rad)
        return (
            telem.north_m + det.x_m * c - det.y_m * s,
            telem.east_m + det.x_m * s + det.y_m * c,
        )

    def _already_counted(self, telem: Telemetry, det: Detection) -> bool:
        n, e = self._target_world(telem, det)
        return any(
            math.hypot(n - cn, e - ce) < self.DEDUP_RADIUS_M for cn, ce in self.counted
        )

    def _alt_hold(self, telem: Telemetry) -> float:
        """순항 고도를 유지하는 수직 속도. NED 규약이라 +가 하강이다."""
        return _clamp(0.6 * (telem.alt_m - self.cruise_alt_m), -0.3, 0.3)
