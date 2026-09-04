"""모든 조정 가능한 값을 한 곳에 모은다.

실내 비행은 여유 공간이 거의 없다. 숫자를 코드 여기저기 흩어놓으면
"왜 벽에 붙었지"를 추적할 수 없게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """안전 한계. Guard가 이 값만 보고 판단한다."""

    # --- 실내 지오펜스 (시작 지점 기준 원통) ---
    max_radius_m: float = 4.0      # 수평 반경. 방 크기의 절반보다 작게 잡을 것.
    min_alt_m: float = 0.8         # 이 아래로는 지면 효과로 자세가 흔들린다.
    max_alt_m: float = 2.0         # 일반적인 방 천장 기준. MTF-02P 의 ToF 는
                                   # 6m 까지 닿으므로 넓은 공간이면 올려도 된다.

    # --- 속도 ---
    max_speed_ms: float = 0.6      # 실내에서 0.6m/s면 충분히 빠르다.
    max_climb_ms: float = 0.4
    max_yaw_rate: float = 0.5      # rad/s (약 29°/s)

    # --- 장애물 ---
    min_front_m: float = 1.0       # 전방에 이보다 가까우면 전진 금지
    min_side_m: float = 0.5

    # --- 배터리 ---
    battery_return_pct: float = 35.0
    battery_land_pct: float = 20.0

    # --- 워치독 (초) ---
    telemetry_timeout_s: float = 1.0
    perception_timeout_s: float = 2.0

    # --- 옵티컬 플로우 ---
    # 이 값 아래로 떨어지면 위치 추정을 믿을 수 없다. 실내 자율의 최대 실패 원인.
    min_flow_quality: int = 50


@dataclass(frozen=True)
class Mission:
    """임무 파라미터."""

    target_label: str = "person"   # 찾을 대상. 모델 클래스명과 일치해야 한다.
    min_confidence: float = 0.55
    cruise_alt_m: float = 1.3
    stop_distance_m: float = 1.5   # 대상 앞 이 거리에서 멈춘다
    approach_tolerance_m: float = 0.3
    inspect_seconds: float = 4.0
    search_yaw_rate: float = 0.35  # rad/s
    lost_target_grace_s: float = 1.5  # 놓쳐도 이 시간은 접근 유지
    max_mission_seconds: float = 180.0


@dataclass(frozen=True)
class Runtime:
    """루프·하드웨어 설정."""

    loop_hz: float = 10.0
    perception_backend: str = "mock"        # mock | oakd | onnx
    rangefinder_backend: str = "none"       # none | vl53l1x  (좌·우·후방)
    link_backend: str = "mock"              # mock | mavlink
    # 실기체는 FC 를 Pi 의 USB 에 꽂는 것이 가장 안전하다 — 배선 실수 여지가
    # 없고 레벨 문제도 없다. SITL 은 "udp:127.0.0.1:14550" 으로 바꿔 쓴다.
    mavlink_url: str = "/dev/ttyACM0:921600"
    onnx_model_path: str = "models/yolo.onnx"
    log_path: str = "flight_log.jsonl"


@dataclass(frozen=True)
class Config:
    limits: Limits = Limits()
    mission: Mission = Mission()
    runtime: Runtime = Runtime()


DEFAULT = Config()
