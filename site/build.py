"""공개 재생 페이지에 실을 비행 데이터를 만든다.

    python3 site/build.py            # site/flight_data.json 을 새로 쓴다
    python3 site/build.py --check    # 새로 쓰지 않고, 커밋된 것과 다른지만 본다
    python3 site/build.py --single OUT.html   # 데이터까지 박아 넣은 한 파일

site/orders.txt 의 명령을 한 줄씩 컴파일하고, 통과한 것만 시뮬레이터로
끝까지 비행시켜 프레임을 모은다. 판정(통과/거부)은 여기서 정하지 않는다 —
drone/mission_spec.py 의 validate() 가 내린 결과를 그대로 싣는다. 그래서
누가 orders.txt 에 이상한 명령을 추가해도 페이지가 거짓말을 하지 않는다.

하드웨어도 네트워크도 쓰지 않는다. mock 인지 + mock 링크 = 순수 계산.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.compiler import compile_mission          # noqa: E402
from drone.config import DEFAULT                    # noqa: E402
from drone.main import run                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS = os.path.join(HERE, "orders.txt")
OUT = os.path.join(HERE, "flight_data.json")
PAGE = os.path.join(HERE, "index.html")

# 재생용으로 5Hz 로 솎는다. 10Hz 로그를 그대로 실으면 파일이 두 배가 되는데
# 화면에서 보이는 것은 똑같다.
KEEP_EVERY = 2
MAX_SECONDS = 90.0

# 페이지가 그리는 필드만 남긴다. vx/vy/vz/yaw_rate 는 지도에 안 나온다.
FIELDS = ("t", "phase", "action", "n", "e", "alt", "batt", "vetoes", "reason")


def read_orders(path: str = ORDERS) -> list[str]:
    orders: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                orders.append(line)
    if not orders:
        raise SystemExit(f"{path} 에 명령이 하나도 없다")
    return orders


def fly(spec) -> list[dict]:
    """한 임무를 끝까지 시뮬레이션하고 프레임 목록을 돌려준다."""
    frames: list[dict] = []

    def on_tick(record, telem, percep, result):
        frames.append({k: record[k] for k in FIELDS})

    # run() 은 로그 파일을 쓰고 진행 상황을 stdout 에 찍는다. 빌드 중에는
    # 둘 다 필요 없다 — 임시 파일로 보내고 출력은 삼킨다.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(
            DEFAULT,
            runtime=dataclasses.replace(
                DEFAULT.runtime, log_path=os.path.join(tmp, "flight_log.jsonl")
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, fast=True, max_seconds=MAX_SECONDS, spec=spec, on_tick=on_tick)

    return frames[::KEEP_EVERY]


def build() -> dict:
    limits = DEFAULT.limits
    scenarios = []

    for order in read_orders():
        res = compile_mission(order, limits, backend="rules")
        spec = res.spec
        scenarios.append({
            "order": order,
            "behavior": spec.behavior.value,
            "target": spec.target_label,
            "stop_distance_m": spec.stop_distance_m,
            "cruise_alt_m": spec.cruise_alt_m,
            "stop_after_n": spec.stop_after_n,
            "notes": list(res.notes),
            "problems": list(res.problems),
            "accepted": res.ok,
            # 거부된 명령은 비행하지 않는다. 빈 로그가 곧 그 사실의 증거다.
            "log": fly(spec) if res.ok else [],
        })

    return {
        "limits": dataclasses.asdict(limits),
        "scenarios": scenarios,
    }


def dumps(data: dict) -> str:
    """시나리오마다 한 줄. 통째로 한 줄이면 diff 에서 뭐가 바뀌었는지 안 보인다."""
    lines = [json.dumps(s, ensure_ascii=False, separators=(",", ":"))
             for s in data["scenarios"]]
    limits = json.dumps(data["limits"], ensure_ascii=False, separators=(",", ":"))
    return ('{"limits":' + limits + ',\n "scenarios":[\n  '
            + ",\n  ".join(lines) + "\n ]}\n")


def single_file(data_json: str) -> str:
    """index.html 에 데이터를 박아 넣어 파일 하나로 만든다.

    서버 없이 두 번 눌러 열 수 있고, 파일 하나만 보내면 되는 자리
    (첨부, 발표용 USB, fetch 를 막는 샌드박스) 를 위한 것이다. 배포되는
    사이트는 이 경로를 쓰지 않는다 — 사이트는 index.html + JSON 둘이다.
    """
    with open(PAGE, encoding="utf-8") as f:
        page = f.read()

    marker = 'res = await fetch("flight_data.json", { cache: "no-cache" });'
    if page.count(marker) != 1:
        raise SystemExit("index.html 의 fetch 호출을 찾지 못했다 — 빌드 스크립트를 맞춰주세요")

    # </script> 가 데이터 안에 있으면 브라우저가 거기서 스크립트를 닫아버린다.
    embedded = data_json.replace("</", "<\\/")
    return page.replace(
        marker,
        "res = new Response(EMBEDDED);   // 단일 파일 빌드 — fetch 하지 않는다",
    ).replace(
        "let DATA = null, LIM = null;",
        "const EMBEDDED = String.raw`" + embedded + "`;\nlet DATA = null, LIM = null;",
    )


def _diff(committed: str, fresh: str) -> None:
    """줄 단위 요약 + 시나리오별 프레임 수 비교."""
    def summarise(text: str) -> list[str]:
        try:
            d = json.loads(text)
        except ValueError:
            return ["(읽을 수 없는 JSON)"]
        return [f"{s['order']}  {'통과' if s['accepted'] else '거부'}"
                f"  {len(s['log'])}프레임" for s in d["scenarios"]]

    print("커밋된 것 → 새로 만든 것", file=sys.stderr)
    for line in difflib.unified_diff(
        summarise(committed), summarise(fresh),
        fromfile="committed", tofile="rebuilt", lineterm="", n=1,
    ):
        print("  " + line, file=sys.stderr)

    # 시나리오 목록이 같은데도 파일이 다르면 프레임 내용이 달라진 것이다.
    if summarise(committed) == summarise(fresh):
        print("\n  시나리오 구성은 같다 — 프레임 내용이 달라졌다.\n"
              "  drone/ 쪽 로직이 바뀌었거나, 비행이 재현되지 않고 있다.",
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="공개 재생 페이지 데이터 빌드")
    p.add_argument(
        "--check", action="store_true",
        help="파일을 쓰지 않고 커밋된 결과와 다른지만 검사한다 (CI 용)",
    )
    p.add_argument(
        "--single", metavar="OUT.html",
        help="데이터를 index.html 안에 박아 넣은 단일 파일을 여기에 쓴다",
    )
    args = p.parse_args()

    text = dumps(build())

    if args.single:
        with open(args.single, "w", encoding="utf-8") as f:
            f.write(single_file(text))
        print(f"{args.single} — {os.path.getsize(args.single) / 1024:.0f} KB (단일 파일)")
        return

    if args.check:
        try:
            with open(OUT, encoding="utf-8") as f:
                committed = f.read()
        except FileNotFoundError:
            committed = ""
        if committed != text:
            print(
                "site/flight_data.json 이 orders.txt / 시뮬레이터와 어긋난다.\n"
                "  python3 site/build.py 를 돌리고 결과를 같이 커밋하세요.\n",
                file=sys.stderr,
            )
            # 무엇이 달라졌는지 말해준다. "다르다"만 찍고 끝나면 CI 로그를
            # 봐도 orders.txt 를 안 돌린 건지 코드가 변한 건지 알 수 없다.
            _diff(committed, text)
            raise SystemExit(1)
        print("site/flight_data.json 최신 상태")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    data = json.loads(text)
    for s in data["scenarios"]:
        mark = "통과" if s["accepted"] else "거부"
        print(f"  {mark}  {s['order']}  ({len(s['log'])} 프레임)")
    print(f"\n{OUT} — {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
