"""VL53L1X ToF 거리계 배열 — 좌·우·후방.

전방은 OAK-D 스테레오 뎁스가 담당한다(``perception.OakDSource``). OAK-D 의
화각은 약 70° 라 옆과 뒤에는 눈이 없고, 그 자리를 이 센서들이 메운다.

**왜 후방까지 다는가**: Policy 는 대상에 너무 가까우면 물러서고, 전방 뎁스가
안 잡힐 때도 후진으로 빠져나온다. 뒤에 눈이 없으면 그 탈출로가 곧 충돌이 된다.

**주소 충돌**: VL53L1X 는 전원을 넣으면 전부 같은 I2C 주소(0x29)로 올라온다.
XSHUT 핀으로 하나씩만 깨우면서 주소를 재할당해야 세 개를 동시에 쓸 수 있다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

#: 재할당할 주소. 0x29 는 부팅 기본값이라 피한다.
DEFAULT_ADDRESSES = (0x30, 0x31, 0x32)

#: 유효 범위(m). VL53L1X 는 최대 4m 이고, 그 밖의 값은 측정 실패로 본다.
MIN_VALID_M = 0.04
MAX_VALID_M = 4.0


@dataclass(frozen=True)
class Clearances:
    """한 번 읽은 좌·우·후방 거리(m). 측정 실패한 방향은 None."""

    left_m: float | None = None
    right_m: float | None = None
    back_m: float | None = None


class RangefinderArray(Protocol):
    def read(self) -> Clearances: ...

    def close(self) -> None: ...


def _sane(metres: float | None) -> float | None:
    """범위를 벗어난 값은 버린다. 지어내지 않고 None 을 돌려준다."""
    if metres is None:
        return None
    return metres if MIN_VALID_M <= metres <= MAX_VALID_M else None


class NullRangefinders:
    """센서를 달지 않았을 때. 전부 None → Guard 가 해당 검사를 건너뛴다."""

    def read(self) -> Clearances:
        return Clearances()

    def close(self) -> None:
        return None


class FixedRangefinders:
    """테스트·시뮬레이션용. 정해진 값을 그대로 돌려준다."""

    def __init__(self, left_m=None, right_m=None, back_m=None) -> None:
        self._value = Clearances(_sane(left_m), _sane(right_m), _sane(back_m))

    def read(self) -> Clearances:
        return self._value

    def close(self) -> None:
        return None


class VL53L1XArray:
    """VL53L1X 세 개를 I2C 하나에 물려 읽는다.

    배선 (Raspberry Pi 5 기준):

        모든 센서 공통   VIN → 3.3V(1번), GND → GND(6번),
                        SDA → GPIO2(3번), SCL → GPIO3(5번)
        좌 XSHUT → GPIO17    우 XSHUT → GPIO27    후 XSHUT → GPIO22

    Pi 5 는 ``RPi.GPIO`` 가 동작하지 않는다. ``gpiozero`` 를 쓴다
    (내부적으로 lgpio 백엔드를 잡는다).

    주의: 이 클래스는 실기체 검증을 하지 않았다. 첫 사용 전에 센서를 하나씩
    손으로 가리며 어느 방향 값이 변하는지 확인할 것 — 배선을 바꿔 꽂으면
    Guard 가 반대쪽을 막는다.
    """

    def __init__(
        self,
        xshut_pins: tuple[int, int, int] = (17, 27, 22),
        addresses: tuple[int, int, int] = DEFAULT_ADDRESSES,
        timing_budget_ms: int = 50,
    ) -> None:
        import board  # 지연 임포트: 시뮬레이션에서는 설치 불필요
        import adafruit_vl53l1x
        from gpiozero import OutputDevice

        self._i2c = board.I2C()
        # 전부 재운 뒤 하나씩 깨우며 주소를 준다. 순서가 곧 좌·우·후 배정이다.
        self._xshut = [OutputDevice(pin, initial_value=False) for pin in xshut_pins]
        time.sleep(0.05)

        self._sensors = []
        for gate, address in zip(self._xshut, addresses):
            gate.on()                      # 이 센서만 깨운다
            time.sleep(0.05)
            sensor = adafruit_vl53l1x.VL53L1X(self._i2c)   # 아직 0x29
            sensor.set_address(address)                     # 고유 주소로 이사
            sensor.distance_mode = 2                        # long range (~4m)
            sensor.timing_budget = timing_budget_ms
            sensor.start_ranging()
            self._sensors.append(sensor)

    def read(self) -> Clearances:
        values = []
        for sensor in self._sensors:
            try:
                # data_ready 가 아니면 직전 값을 다시 읽게 되므로 건너뛴다.
                raw_cm = sensor.distance if sensor.data_ready else None
                sensor.clear_interrupt()
            except OSError:
                raw_cm = None      # I2C 가 한 번 튀는 건 흔하다. 값을 지어내지 않는다.
            values.append(_sane(raw_cm / 100.0) if raw_cm is not None else None)
        return Clearances(*values)

    def close(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop_ranging()
            except OSError:
                pass
        for gate in self._xshut:
            gate.off()
            gate.close()


def build(kind: str, **kwargs) -> RangefinderArray:
    if kind in ("none", "mock"):
        return NullRangefinders()
    if kind == "vl53l1x":
        return VL53L1XArray(**kwargs)
    raise ValueError(f"알 수 없는 거리계 백엔드: {kind}")
