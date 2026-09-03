"""기체 상태·인식 결과·명령의 자료형 정의.

이 모듈은 의존성이 없다. 인식(perception), 안전(guard), 판단(policy),
통신(link) 네 계층이 전부 여기 정의된 형만 주고받는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    """상위 계층이 낼 수 있는 명령의 전부. 화이트리스트다."""

    HOLD = "hold"          # 제자리 정지 호버
    MOVE = "move"          # 기체 기준 속도 명령
    TAKEOFF = "takeoff"
    LAND = "land"
    RETURN = "return"      # 시작 지점 복귀


class Phase(str, Enum):
    """온보드 판단 상태기계의 단계."""

    IDLE = "idle"
    TAKEOFF = "takeoff"
    SEARCH = "search"      # 제자리 회전하며 대상 탐색
    APPROACH = "approach"  # 대상까지 접근
    INSPECT = "inspect"    # 정지 관찰
    RETURN = "return"
    LAND = "land"
    DONE = "done"


@dataclass(frozen=True)
class Detection:
    """온보드 모델이 낸 검출 하나.

    좌표는 기체 기준(body frame), 단위는 미터.
    x = 전방(+), y = 우측(+), z = 하방(+). ArduPilot의 NED와 축 방향을 맞춘다.
    스테레오 카메라(OAK-D)는 셋 다 실측, 단안 카메라는 x만 추정하고
    y/z는 화각 기반 근사이므로 `has_depth`로 구분한다.
    """

    label: str
    confidence: float
    x_m: float
    y_m: float
    z_m: float
    has_depth: bool = True

    @property
    def distance_m(self) -> float:
        return (self.x_m**2 + self.y_m**2 + self.z_m**2) ** 0.5


@dataclass(frozen=True)
class Perception:
    """한 프레임의 인식 결과."""

    detections: tuple[Detection, ...] = ()
    # 전/좌/우 ToF 거리계. 값이 없으면 None (센서 미장착 또는 측정 실패).
    front_m: float | None = None
    left_m: float | None = None
    right_m: float | None = None
    stamp: float = field(default_factory=time.monotonic)

    def best(self, label: str, min_conf: float) -> Detection | None:
        """지정 라벨 중 가장 확신도가 높은 검출. 없으면 None."""
        hits = [d for d in self.detections if d.label == label and d.confidence >= min_conf]
        return max(hits, key=lambda d: d.confidence) if hits else None


@dataclass(frozen=True)
class Telemetry:
    """비행 컨트롤러가 보고한 기체 상태."""

    armed: bool
    mode: str
    # 시작 지점 기준 로컬 좌표(m). 실내에서는 옵티컬 플로우 + EKF3가 채운다.
    north_m: float
    east_m: float
    alt_m: float           # 지면 기준 고도(+가 위)
    yaw_rad: float         # 기수 방위. 기체 기준 속도를 월드 좌표로 바꿀 때 쓴다.
    battery_pct: float
    # 옵티컬 플로우 품질 0~255. ArduPilot의 FLOW_QUALITY.
    # 바닥 무늬가 없거나 너무 어두우면 떨어지고, 그러면 위치 추정이 무너진다.
    flow_quality: int = 255
    stamp: float = field(default_factory=time.monotonic)

    @property
    def distance_from_home_m(self) -> float:
        return (self.north_m**2 + self.east_m**2) ** 0.5


@dataclass(frozen=True)
class Command:
    """하위 계층으로 내려가는 명령. 반드시 Guard를 통과한 것만 내려간다."""

    action: Action
    vx: float = 0.0          # 전방 속도 m/s (기체 기준)
    vy: float = 0.0          # 우측 속도 m/s
    vz: float = 0.0          # 하강 속도 m/s (+가 아래, NED 규약)
    yaw_rate: float = 0.0    # rad/s, +가 시계방향
    target_alt_m: float = 0.0
    reason: str = ""

    def replace(self, **kw) -> "Command":
        from dataclasses import replace as _replace

        return _replace(self, **kw)


HOLD = Command(action=Action.HOLD, reason="기본 정지")
