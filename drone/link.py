"""비행 컨트롤러 통신 계층.

``MockLink``    : 간단한 운동학 시뮬레이터. 하드웨어 없이 전체 루프를 돌린다.
``MavlinkLink`` : pymavlink로 실기체(ArduPilot)에 연결한다.
"""

from __future__ import annotations

import math
import time

from .state import Action, Command, Telemetry

# SET_POSITION_TARGET_LOCAL_NED 의 type_mask.
# 위치·가속도·yaw 각도는 무시하고 속도와 yaw_rate만 사용한다.
_MASK_VEL_YAWRATE = 0b0000010111000111  # = 1479
_FRAME_BODY_NED = 8  # MAV_FRAME_BODY_NED


class MockLink:
    """실기체 없이 판단·안전 계층을 검증하기 위한 운동학 모델.

    공기역학은 없다. 명령 속도가 곧바로 실제 속도가 된다고 가정한다.
    자세 제어기가 하는 일을 흉내내려는 게 아니라, 상위 계층의 논리를
    확인하는 것이 목적이기 때문이다.
    """

    def __init__(
        self,
        battery_pct: float = 100.0,
        drain_per_s: float = 0.35,
        flow_quality: int = 200,
    ) -> None:
        self.north = 0.0
        self.east = 0.0
        self.alt = 0.0
        self.yaw = 0.0
        self.armed = False
        self.mode = "GUIDED"
        self.battery_pct = battery_pct
        self.drain_per_s = drain_per_s
        self.flow_quality = flow_quality
        self._last = time.monotonic()

    def telemetry(self) -> Telemetry:
        return Telemetry(
            armed=self.armed,
            mode=self.mode,
            north_m=round(self.north, 4),
            east_m=round(self.east, 4),
            alt_m=round(self.alt, 4),
            yaw_rad=self.yaw,
            battery_pct=round(self.battery_pct, 2),
            flow_quality=self.flow_quality,
            stamp=time.monotonic(),
        )

    def send(self, cmd: Command, dt: float | None = None) -> None:
        now = time.monotonic()
        if dt is None:
            dt = min(0.5, now - self._last)
        self._last = now
        self.battery_pct = max(0.0, self.battery_pct - self.drain_per_s * dt)

        if cmd.action is Action.TAKEOFF:
            self.armed = True
            self.mode = "GUIDED"
            self.alt = min(cmd.target_alt_m, self.alt + 0.5 * dt)
            return

        if cmd.action is Action.LAND:
            self.mode = "LAND"
            self.alt = max(0.0, self.alt - 0.4 * dt)
            if self.alt <= 0.01:
                self.armed = False
            return

        if not self.armed:
            return

        # HOLD 도 vz(고도 유지)는 반영한다.
        self.yaw = (self.yaw + cmd.yaw_rate * dt) % (2 * math.pi)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.north += (cmd.vx * c - cmd.vy * s) * dt
        self.east += (cmd.vx * s + cmd.vy * c) * dt
        self.alt = max(0.0, self.alt - cmd.vz * dt)  # vz는 +가 하강

    def close(self) -> None:
        return None


class MavlinkLink:
    """ArduPilot 실기체 연결.

    실내 전제이므로 GPS 관련 명령은 하나도 쓰지 않는다. 위치는 전적으로
    옵티컬 플로우 + 거리계가 채운 EKF3의 로컬 좌표(LOCAL_POSITION_NED)에 의존한다.

    주의: SITL에서는 검증했지만 실기체 검증은 하지 않았다.
    첫 비행은 반드시 프롭을 뺀 상태로 명령 흐름부터 확인할 것.
    """

    def __init__(self, url: str, source_system: int = 1) -> None:
        from pymavlink import mavutil  # 지연 임포트

        self._mavutil = mavutil
        self.conn = mavutil.mavlink_connection(url, source_system=source_system)
        self.conn.wait_heartbeat()

        self._north = self._east = self._alt = 0.0
        self._yaw = 0.0
        self._battery = 100.0
        self._flow_quality = 255
        self._armed = False
        self._mode = "UNKNOWN"
        self._stamp = time.monotonic()

    # --- 수신 ---------------------------------------------------------

    def pump(self) -> None:
        """대기 중인 MAVLink 메시지를 모두 소비한다. 루프마다 한 번 호출."""
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            self._absorb(msg)

    def _absorb(self, msg) -> None:
        kind = msg.get_type()
        if kind == "LOCAL_POSITION_NED":
            self._north, self._east = msg.x, msg.y
            self._alt = -msg.z  # NED의 z는 아래가 +
            self._stamp = time.monotonic()
        elif kind == "ATTITUDE":
            self._yaw = msg.yaw
        elif kind == "HEARTBEAT":
            self._armed = bool(
                msg.base_mode & self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self._mode = self.conn.flightmode
        elif kind == "BATTERY_STATUS":
            if msg.battery_remaining >= 0:
                self._battery = float(msg.battery_remaining)
        elif kind == "SYS_STATUS":
            if msg.battery_remaining >= 0:
                self._battery = float(msg.battery_remaining)
        elif kind == "OPTICAL_FLOW":
            self._flow_quality = int(msg.quality)

    def telemetry(self) -> Telemetry:
        return Telemetry(
            armed=self._armed,
            mode=self._mode,
            north_m=self._north,
            east_m=self._east,
            alt_m=self._alt,
            yaw_rad=self._yaw,
            battery_pct=self._battery,
            flow_quality=self._flow_quality,
            stamp=self._stamp,
        )

    # --- 송신 ---------------------------------------------------------

    def send(self, cmd: Command, dt: float | None = None) -> None:
        if cmd.action is Action.TAKEOFF:
            self._ensure_mode("GUIDED")
            if not self._armed:
                self._arm()
            self.conn.mav.command_long_send(
                self.conn.target_system, self.conn.target_component,
                self._mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, cmd.target_alt_m,
            )
            return

        if cmd.action is Action.LAND:
            self._ensure_mode("LAND")
            return

        # HOLD 는 전 성분 0인 속도 명령으로 보낸다. 명령을 끊으면 ArduPilot의
        # GUIDED 타임아웃이 걸려 동작이 기체 설정에 따라 달라진다.
        self.conn.mav.set_position_target_local_ned_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            _FRAME_BODY_NED,
            _MASK_VEL_YAWRATE,
            0, 0, 0,
            cmd.vx, cmd.vy, cmd.vz,
            0, 0, 0,
            0, cmd.yaw_rate,
        )

    def send_obstacle_distance(self, distances_cm: list[int], increment_deg: int = 45) -> None:
        """ToF 센서 값을 ArduPilot의 회피 계층에도 넘긴다.

        상위 계층이 멈춰도 비행 컨트롤러가 독립적으로 막아준다.
        방어선을 두 겹으로 두는 것이 목적이다.
        """
        self.conn.mav.obstacle_distance_send(
            int(time.monotonic() * 1e6),
            0,                       # MAV_DISTANCE_SENSOR_LASER
            distances_cm,
            increment_deg,
            20, 400,                 # min/max cm
            float(increment_deg),
            0,                       # MAV_FRAME_BODY_FRD
        )

    def _arm(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def _ensure_mode(self, name: str) -> None:
        if self._mode == name:
            return
        self.conn.set_mode(name)

    def close(self) -> None:
        self.conn.close()


def build(cfg):
    backend = cfg.runtime.link_backend
    if backend == "mock":
        return MockLink()
    if backend == "mavlink":
        return MavlinkLink(cfg.runtime.mavlink_url)
    raise ValueError(f"알 수 없는 link 백엔드: {backend}")
