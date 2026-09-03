"""자연어 명령이 컴파일되어 도착하는 형태.

핵심 설계: **자연어는 이륙 전에 딱 한 번 읽는다.**

명령을 비행 중에 해석하면 언어 모델이 제어 루프 안에 들어오게 되고,
그러면 모델이 틀렸을 때 그 결과가 곧바로 기체 움직임이 된다.
대신 지상에서 한 번 MissionSpec 으로 컴파일하고, 이륙 전에 Guard 의
한계와 대조해 검증한 뒤, 비행 자체는 결정론적 상태기계가 수행한다.

잘못된 명령은 프로펠러가 돌기 전에 걸러진다.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum

from .config import Limits, Mission


class Behavior(str, Enum):
    """자연어 명령이 귀결될 수 있는 임무 유형의 전부. 화이트리스트다."""

    APPROACH_INSPECT = "approach_inspect"  # 찾아가서 앞에 서고 관찰한 뒤 복귀
    COUNT = "count"                        # 여러 개를 찾아 세고 복귀
    FOLLOW = "follow"                      # 일정 거리를 유지하며 계속 추종
    PATROL = "patrol"                      # 제자리 회전하며 보이는 것을 기록


@dataclass(frozen=True)
class MissionSpec:
    """컴파일된 임무. 이 형태만이 Policy 로 넘어간다."""

    behavior: Behavior = Behavior.APPROACH_INSPECT
    target_label: str = "person"
    cruise_alt_m: float = 1.3
    stop_distance_m: float = 1.5
    inspect_seconds: float = 4.0
    stop_after_n: int = 1           # COUNT 에서 몇 개까지 세고 끝낼지
    max_mission_seconds: float = 180.0
    source_text: str = ""           # 원본 명령. 로그 감사를 위해 보존한다.

    def to_mission(self, base: Mission) -> Mission:
        """Policy 가 이해하는 튜닝 dataclass 로 병합한다."""
        return dataclasses.replace(
            base,
            target_label=self.target_label,
            cruise_alt_m=self.cruise_alt_m,
            stop_distance_m=self.stop_distance_m,
            inspect_seconds=self.inspect_seconds,
            max_mission_seconds=self.max_mission_seconds,
        )


def validate(spec: MissionSpec, limits: Limits) -> tuple[str, ...]:
    """이륙 전 검증. 문제가 없으면 빈 튜플.

    여기서 걸러내지 못한 모순은 비행 중에 '기체가 가만히 떠 있기만 하는'
    형태로 나타나고, 그때는 원인을 알아내기가 훨씬 어렵다.
    """
    problems: list[str] = []

    if not (limits.min_alt_m <= spec.cruise_alt_m <= limits.max_alt_m):
        problems.append(
            f"순항 고도 {spec.cruise_alt_m}m 가 허용 범위 "
            f"{limits.min_alt_m}~{limits.max_alt_m}m 를 벗어난다"
        )

    # 가장 잡기 어려운 모순: 정지 거리가 회피 여유보다 짧으면,
    # Guard 가 전진을 영원히 막아 기체는 임무 제한 시간까지 떠 있기만 한다.
    if spec.stop_distance_m < limits.min_front_m:
        problems.append(
            f"정지 거리 {spec.stop_distance_m}m 가 전방 회피 여유 "
            f"{limits.min_front_m}m 보다 짧다 — Guard 가 전진을 계속 막게 된다"
        )

    if spec.stop_distance_m > limits.max_radius_m:
        problems.append(
            f"정지 거리 {spec.stop_distance_m}m 가 지오펜스 반경 "
            f"{limits.max_radius_m}m 보다 크다 — 대상에 닿기 전에 경계에 걸린다"
        )

    if spec.stop_after_n < 1:
        problems.append(f"목표 개수 {spec.stop_after_n} 는 1 이상이어야 한다")

    if spec.inspect_seconds < 0:
        problems.append("관찰 시간은 음수일 수 없다")

    if spec.max_mission_seconds <= 0:
        problems.append("임무 제한 시간은 양수여야 한다")

    # 배터리로 감당 못 할 임무는 애초에 받지 않는다.
    if spec.behavior is Behavior.COUNT and spec.stop_after_n > 5:
        problems.append(
            f"{spec.stop_after_n} 개를 세는 임무는 실내 비행시간(6~9분)으로 무리다"
        )

    if not spec.target_label.strip():
        problems.append("찾을 대상이 지정되지 않았다")

    return tuple(problems)
