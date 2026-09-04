"""결정론적 안전 검증기.

이 파일에는 학습된 모델도, 확률도, 네트워크 호출도 없다. 전부 if 문이다.
그게 요점이다.

상위 판단 계층(policy)이 낸 명령은 반드시 여기를 통과해야 MAVLink로 나간다.
Guard는 인식 모델이 보고한 값이 아니라 **센서 원본과 텔레메트리**만 본다.
모델에게 "지금 위험한가?"를 묻고 그 대답으로 안전을 판정하면,
모델이 틀렸을 때 그 사실을 알아낼 방법이 사라진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Limits
from .state import Action, Command, Perception, Phase, Telemetry


@dataclass(frozen=True)
class GuardResult:
    command: Command
    vetoes: tuple[str, ...] = ()
    # Guard가 임무 단계 자체를 바꿔야 한다고 판단한 경우 (배터리 부족 등).
    force_phase: Phase | None = None

    @property
    def modified(self) -> bool:
        return bool(self.vetoes)


def body_to_ned(vx: float, vy: float, yaw: float) -> tuple[float, float]:
    """기체 기준 속도 → 월드(NED) 속도."""
    c, s = math.cos(yaw), math.sin(yaw)
    return vx * c - vy * s, vx * s + vy * c


def ned_to_body(vn: float, ve: float, yaw: float) -> tuple[float, float]:
    """월드(NED) 속도 → 기체 기준 속도."""
    c, s = math.cos(yaw), math.sin(yaw)
    return vn * c + ve * s, -vn * s + ve * c


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Guard:
    """명령 하나를 받아 안전한 명령 하나를 돌려준다."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits

    def check(
        self,
        cmd: Command,
        telem: Telemetry | None,
        percep: Perception | None,
        now: float,
    ) -> GuardResult:
        lim = self.limits
        vetoes: list[str] = []

        # 1. 텔레메트리가 없거나 오래됐다 → 기체 상태를 모른다. 아무것도 하지 않는다.
        if telem is None or (now - telem.stamp) > lim.telemetry_timeout_s:
            return GuardResult(
                Command(Action.HOLD, reason="텔레메트리 두절"),
                ("telemetry_stale",),
            )

        # 2. 배터리 위험 — 다른 모든 판단을 덮어쓴다.
        if telem.battery_pct <= lim.battery_land_pct:
            return GuardResult(
                Command(Action.LAND, reason=f"배터리 {telem.battery_pct:.0f}% 임계"),
                ("battery_critical",),
                force_phase=Phase.LAND,
            )

        # 3. 시동이 걸려 있지 않으면 이륙 외에는 아무 명령도 의미가 없다.
        if not telem.armed and cmd.action is not Action.TAKEOFF:
            return GuardResult(
                Command(Action.HOLD, reason="비무장 상태"), ("disarmed",)
            )

        # 4. 옵티컬 플로우 품질 저하 — 실내 자율의 최대 실패 원인.
        #    위치 추정을 못 믿는 상태에서 이동 명령을 내리면 기체가 흘러간다.
        if telem.flow_quality < lim.min_flow_quality and cmd.action is Action.MOVE:
            vetoes.append("flow_quality_low")
            cmd = Command(
                Action.HOLD,
                reason=f"플로우 품질 {telem.flow_quality} 미달 — 위치 추정 불가",
            )

        # 5. 인식이 끊겼다 → 전진 금지. 못 보는 채로 앞으로 가지 않는다.
        stale_perception = percep is None or (now - percep.stamp) > lim.perception_timeout_s
        if stale_perception and cmd.action is Action.MOVE and cmd.vx > 0:
            vetoes.append("perception_stale")
            cmd = cmd.replace(vx=0.0, reason="인식 두절 — 전진 차단")

        # 배터리 복귀 권고는 명령을 막지 않고 단계 전환만 요청한다.
        force_phase: Phase | None = None
        if telem.battery_pct <= lim.battery_return_pct:
            vetoes.append("battery_low")
            force_phase = Phase.RETURN

        if cmd.action in (Action.LAND, Action.HOLD, Action.TAKEOFF):
            # 이 셋은 수평 이동이 없으므로 고도 검사만 적용한다.
            cmd = self._clamp_vertical(cmd, telem, vetoes)
            return GuardResult(cmd, tuple(vetoes), force_phase)

        # --- 여기부터는 MOVE / RETURN, 즉 실제로 기체를 움직이는 명령 ---
        cmd = self._clamp_speed(cmd, vetoes)
        cmd = self._apply_obstacles(cmd, percep, stale_perception, vetoes)
        cmd = self._apply_geofence(cmd, telem, vetoes)
        cmd = self._clamp_vertical(cmd, telem, vetoes)

        return GuardResult(cmd, tuple(vetoes), force_phase)

    # ------------------------------------------------------------------

    def _clamp_speed(self, cmd: Command, vetoes: list[str]) -> Command:
        lim = self.limits
        vx = _clamp(cmd.vx, lim.max_speed_ms)
        vy = _clamp(cmd.vy, lim.max_speed_ms)
        vz = _clamp(cmd.vz, lim.max_climb_ms)
        yaw_rate = _clamp(cmd.yaw_rate, lim.max_yaw_rate)
        if (vx, vy, vz, yaw_rate) != (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate):
            vetoes.append("speed_clamped")
        return cmd.replace(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)

    def _apply_obstacles(
        self,
        cmd: Command,
        percep: Perception | None,
        stale: bool,
        vetoes: list[str],
    ) -> Command:
        """거리계 원본값으로 이동 성분을 깎는다. 검출 결과는 쓰지 않는다."""
        lim = self.limits
        if percep is None or stale:
            return cmd

        # 전방 센서가 값을 못 내는 상태. '센서 없음'과 다르게 취급한다.
        # 유일한 전방 센서가 눈을 감았으면 전진을 허용할 근거가 없다.
        if percep.forward_blind and cmd.vx > 0:
            vetoes.append("forward_blind")
            cmd = cmd.replace(vx=0.0, reason="전방 거리 측정 불가 — 전진 차단")

        if percep.front_m is not None and percep.front_m < lim.min_front_m and cmd.vx > 0:
            vetoes.append("obstacle_front")
            cmd = cmd.replace(vx=0.0, reason=f"전방 {percep.front_m:.2f}m — 전진 차단")

        if percep.right_m is not None and percep.right_m < lim.min_side_m and cmd.vy > 0:
            vetoes.append("obstacle_right")
            cmd = cmd.replace(vy=0.0)

        if percep.left_m is not None and percep.left_m < lim.min_side_m and cmd.vy < 0:
            vetoes.append("obstacle_left")
            cmd = cmd.replace(vy=0.0)

        # 후진도 막는다. Policy 는 대상에 너무 가까우면 물러서고, 전방을 못 볼 때도
        # 후진으로 빠져나온다 — 뒤에 눈이 없으면 그 탈출로가 곧 충돌이 된다.
        if percep.back_m is not None and percep.back_m < lim.min_side_m and cmd.vx < 0:
            vetoes.append("obstacle_back")
            cmd = cmd.replace(vx=0.0, reason=f"후방 {percep.back_m:.2f}m — 후진 차단")

        return cmd

    def _apply_geofence(
        self, cmd: Command, telem: Telemetry, vetoes: list[str]
    ) -> Command:
        """실내 원통 지오펜스. 바깥으로 나가는 속도 성분만 제거한다.

        전체를 HOLD로 막지 않고 성분만 깎는 이유: 경계에 붙었을 때
        안쪽으로 돌아오는 명령까지 막아버리면 기체가 벽에 갇힌다.
        """
        lim = self.limits
        radius = telem.distance_from_home_m
        if radius < lim.max_radius_m * 0.9:
            return cmd

        vn, ve = body_to_ned(cmd.vx, cmd.vy, telem.yaw_rad)
        if radius < 1e-6:
            return cmd

        # 바깥 방향 단위 벡터
        un, ue = telem.north_m / radius, telem.east_m / radius
        outward = vn * un + ve * ue
        if outward <= 0:
            return cmd  # 안쪽으로 가는 중이면 통과

        vetoes.append("geofence")
        # 바깥 성분만 빼고 접선 성분은 남긴다.
        vn -= outward * un
        ve -= outward * ue
        vx, vy = ned_to_body(vn, ve, telem.yaw_rad)
        return cmd.replace(
            vx=vx, vy=vy, reason=f"지오펜스 {radius:.2f}/{lim.max_radius_m}m"
        )

    def _clamp_vertical(
        self, cmd: Command, telem: Telemetry, vetoes: list[str]
    ) -> Command:
        lim = self.limits
        if cmd.action is Action.LAND:
            return cmd  # 착륙은 고도 하한을 무시해야 한다.

        # vz는 NED 규약: + 가 하강.
        if telem.alt_m <= lim.min_alt_m and cmd.vz > 0:
            vetoes.append("alt_floor")
            cmd = cmd.replace(vz=0.0)
        if telem.alt_m >= lim.max_alt_m and cmd.vz < 0:
            vetoes.append("alt_ceiling")
            cmd = cmd.replace(vz=0.0)
        return cmd
