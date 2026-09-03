"""자연어 명령 → MissionSpec 컴파일러.

백엔드 두 개가 같은 함수를 구현한다.

- ``rules``  : 한국어/영어 키워드 파싱. 의존성 없음, 결정론적, 오프라인.
- ``claude`` : Claude API + 구조화 출력. 훨씬 유연하지만 네트워크가 필요하다.

둘 다 **지상에서, 이륙 전에** 한 번만 돈다. 그래서 claude 백엔드의
1~3초 지연은 문제가 되지 않는다 — 비행 중이 아니라 준비 중이기 때문이다.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

from .config import Limits
from .mission_spec import Behavior, MissionSpec, validate


@dataclass(frozen=True)
class CompileResult:
    spec: MissionSpec
    problems: tuple[str, ...] = ()
    backend: str = "rules"
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems


# --- 한국어 라벨 → 모델 클래스명 -------------------------------------------
# 폐쇄형 모델(MobileNet-SSD 20종, COCO 80종)을 쓸 때 필요한 사전.
# YOLOE 같은 개방형 어휘 모델을 쓰면 이 사전 없이 한국어를 영어로만
# 옮겨 그대로 프롬프트에 넣으면 된다.
LABELS = {
    "사람": "person", "인물": "person",
    "의자": "chair", "소파": "sofa",
    "병": "bottle", "페트병": "bottle", "물병": "bottle",
    "화분": "pottedplant", "식물": "pottedplant",
    "테이블": "diningtable", "탁자": "diningtable", "식탁": "diningtable",
    "티비": "tvmonitor", "tv": "tvmonitor", "모니터": "tvmonitor",
    "개": "dog", "강아지": "dog", "고양이": "cat",
    "자전거": "bicycle", "자동차": "car", "오토바이": "motorbike",
    "가방": "backpack", "노트북": "laptop",
}

_NUM_KO = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

_BEHAVIOR_HINTS = (
    # 순서가 중요하다. 앞에 오는 것이 우선한다.
    (Behavior.COUNT, ("몇 개", "몇개", "개수", "세어", "세줘", "세 줘", "카운트", "count", "how many")),
    (Behavior.FOLLOW, ("따라", "쫓아", "추종", "따라와", "follow", "track")),
    (Behavior.PATROL, ("둘러", "한 바퀴", "한바퀴", "순찰", "살펴", "훑어", "patrol", "look around")),
)


def _find_target(text: str) -> tuple[str | None, str | None]:
    """가장 먼저 등장하는 알려진 대상어를 찾는다."""
    best: tuple[int, str, str] | None = None
    for ko, en in LABELS.items():
        idx = text.find(ko)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, ko, en)
    if best is not None:
        return best[2], best[1]

    # 영어 클래스명을 그대로 쓴 경우
    for en in set(LABELS.values()):
        if re.search(rf"\b{re.escape(en)}\b", text, re.IGNORECASE):
            return en, en
    return None, None


def _find_count(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:개|마리|대|명|번)", text)
    if m:
        return int(m.group(1))
    for word, value in _NUM_KO.items():
        if re.search(rf"{word}\s*(?:개|마리|대|명)", text):
            return value
    return None


_LENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm\b|센티(?:미터)?|mm\b|밀리(?:미터)?|m\b|미터|메터)",
    re.IGNORECASE,
)

_TO_METRES = {"cm": 0.01, "센티": 0.01, "센티미터": 0.01,
              "mm": 0.001, "밀리": 0.001, "밀리미터": 0.001}


def _find_length(text: str, keywords: tuple[str, ...]) -> float | None:
    """'2m 앞', '1.5미터 높이', '30cm' 같은 표현에서 길이를 미터로 뽑는다.

    cm/mm 를 못 읽고 지나가면 명령이 조용히 기본값으로 대체된다.
    사람은 30cm 를 말했는데 기체는 1.5m 에서 멈추는, 알아채기 어려운 오해석이다.
    """
    for m in _LENGTH_RE.finditer(text):
        tail = text[m.end(): m.end() + 12]
        head = text[max(0, m.start() - 12): m.start()]
        if any(k in tail or k in head for k in keywords):
            unit = m.group(2).lower()
            return float(m.group(1)) * _TO_METRES.get(unit, 1.0)
    return None


def compile_rules(text: str, base: MissionSpec | None = None) -> tuple[MissionSpec, tuple[str, ...]]:
    """키워드 기반 파서. 네트워크도 모델도 쓰지 않는다."""
    spec = base or MissionSpec()
    notes: list[str] = []
    lowered = text.lower()

    behavior = Behavior.APPROACH_INSPECT
    for candidate, hints in _BEHAVIOR_HINTS:
        if any(h in lowered for h in hints):
            behavior = candidate
            break

    target, ko_word = _find_target(text)
    if target is None:
        target = spec.target_label
        notes.append(f"대상어를 못 찾아 기본값 '{target}' 을 쓴다")
    elif ko_word and ko_word != target:
        notes.append(f"'{ko_word}' → 모델 클래스 '{target}'")

    count = _find_count(text)
    if behavior is Behavior.COUNT and count is None:
        count = 3
        notes.append("개수가 명시되지 않아 3개까지 세는 것으로 잡았다")

    stop_distance = _find_length(text, ("앞", "거리", "떨어", "distance"))
    altitude = _find_length(text, ("높이", "고도", "상공", "높", "alt"))

    compiled = dataclasses.replace(
        spec,
        behavior=behavior,
        target_label=target,
        stop_after_n=count if count is not None else spec.stop_after_n,
        stop_distance_m=stop_distance if stop_distance is not None else spec.stop_distance_m,
        cruise_alt_m=altitude if altitude is not None else spec.cruise_alt_m,
        source_text=text,
    )
    return compiled, tuple(notes)


# --- Claude 백엔드 ---------------------------------------------------------

_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "behavior": {
                "type": "string",
                "enum": [b.value for b in Behavior],
            },
            "target_label": {"type": "string"},
            "cruise_alt_m": {"type": "number"},
            "stop_distance_m": {"type": "number"},
            "inspect_seconds": {"type": "number"},
            "stop_after_n": {"type": "integer"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "behavior", "target_label", "cruise_alt_m",
            "stop_distance_m", "inspect_seconds", "stop_after_n", "reasoning",
        ],
        "additionalProperties": False,
    },
}


def _system_prompt(limits: Limits) -> str:
    return f"""너는 실내 자율 드론의 임무 컴파일러다. 사람이 한국어로 말한 임무를
구조화된 명세 하나로 옮긴다. 비행 중이 아니라 이륙 전에 한 번 호출된다.

가능한 behavior 는 넷뿐이다:
- approach_inspect: 대상을 찾아가 앞에 서고 잠시 관찰한 뒤 복귀
- count: 대상을 여러 개 찾아 세고 복귀
- follow: 일정 거리를 유지하며 계속 추종
- patrol: 제자리에서 회전하며 보이는 것을 기록

기체의 물리적 한계 (이 범위를 벗어나는 값을 내면 임무가 거부된다):
- 순항 고도: {limits.min_alt_m}~{limits.max_alt_m} m
- 정지 거리: {limits.min_front_m} m 이상, {limits.max_radius_m} m 이하
  (전방 회피가 {limits.min_front_m}m 에서 걸리므로 그보다 가까이 설정하면
   기체가 영원히 전진하지 못한다)
- 활동 반경: 시작 지점에서 {limits.max_radius_m} m

target_label 은 영어 소문자 단수형 클래스명으로 낸다 (person, chair, bottle …).
명령에 없는 값은 추측하지 말고 합리적인 기본값을 쓴다.
reasoning 에는 왜 그렇게 해석했는지 한 문장으로 적는다."""


def compile_claude(
    text: str,
    limits: Limits,
    base: MissionSpec | None = None,
    model: str = "claude-opus-5",
) -> tuple[MissionSpec, tuple[str, ...]]:
    """Claude API 로 컴파일한다. 구조화 출력이라 스키마 위반이 나올 수 없다."""
    import anthropic  # 지연 임포트: rules 백엔드만 쓸 때는 설치 불필요

    client = anthropic.Anthropic()
    spec = base or MissionSpec()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_system_prompt(limits),
        output_config={"effort": "low", "format": _SCHEMA},
        messages=[{"role": "user", "content": text}],
    )
    payload = json.loads(next(b.text for b in response.content if b.type == "text"))

    compiled = dataclasses.replace(
        spec,
        behavior=Behavior(payload["behavior"]),
        target_label=payload["target_label"],
        cruise_alt_m=float(payload["cruise_alt_m"]),
        stop_distance_m=float(payload["stop_distance_m"]),
        inspect_seconds=float(payload["inspect_seconds"]),
        stop_after_n=int(payload["stop_after_n"]),
        source_text=text,
    )
    return compiled, (payload["reasoning"],)


# --- 진입점 ----------------------------------------------------------------


def compile_mission(
    text: str,
    limits: Limits | None = None,
    backend: str = "rules",
    base: MissionSpec | None = None,
) -> CompileResult:
    """자연어 한 줄을 검증된 MissionSpec 으로 바꾼다.

    검증에 실패하면 ``problems`` 가 채워지고, 호출자는 이륙을 시작해서는 안 된다.
    """
    limits = limits or Limits()

    if backend == "rules":
        spec, notes = compile_rules(text, base)
    elif backend == "claude":
        spec, notes = compile_claude(text, limits, base)
    else:
        raise ValueError(f"알 수 없는 컴파일러 백엔드: {backend}")

    return CompileResult(
        spec=spec,
        problems=validate(spec, limits),
        backend=backend,
        notes=notes,
    )
