"""실행 루프.

    python -m drone.main                      # 시뮬레이션 (하드웨어 불필요)
    python -m drone.main --fast               # 대기 없이 즉시 완주
    python -m drone.main --link mavlink --perception oakd   # 실기체
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

from . import link as link_mod
from . import perception as perception_mod
from .config import DEFAULT, Config, Runtime
from .compiler import compile_mission
from .guard import Guard
from .mission_spec import MissionSpec
from .policy import Policy
from .state import Phase


def build_config(args) -> Config:
    runtime = dataclasses.replace(
        DEFAULT.runtime,
        perception_backend=args.perception,
        rangefinder_backend=args.rangefinders,
        link_backend=args.link,
        compiler_backend=args.compiler,
        mavlink_url=args.mavlink_url,
        log_path=args.log,
    )
    return dataclasses.replace(DEFAULT, runtime=runtime)


def run(
    cfg: Config,
    *,
    fast: bool = False,
    max_seconds: float = 240.0,
    spec: MissionSpec | None = None,
    on_tick=None,
    should_stop=None,
) -> dict:
    """비행 루프를 끝까지 돌린다.

    ``on_tick(record, telem, percep, result)`` 이 주어지면 매 틱 호출한다.
    관제 대시보드가 여기에 붙는다 — 루프는 대시보드의 존재를 모른다.
    ``should_stop()`` 이 True 를 돌려주면 즉시 착륙 없이 루프를 끊는다
    (대시보드의 중단 버튼용. 기체는 GUIDED 타임아웃으로 정지한다).
    """
    mission = spec.to_mission(cfg.mission) if spec is not None else cfg.mission
    source = perception_mod.build(dataclasses.replace(
        cfg, mission=mission
    ) if spec is not None else cfg)
    fc = link_mod.build(cfg)
    policy = Policy(mission, max_yaw_rate=cfg.limits.max_yaw_rate, spec=spec)
    guard = Guard(cfg.limits)

    dt = 1.0 / cfg.runtime.loop_hz
    started = time.monotonic()
    virtual = 0.0
    vetoed_ticks = 0
    ticks = 0

    log = open(cfg.runtime.log_path, "w", encoding="utf-8")
    try:
        while True:
            now = started + virtual if fast else time.monotonic()

            if hasattr(fc, "pump"):
                fc.pump()
            telem = fc.telemetry()
            if fast:
                telem = dataclasses.replace(telem, stamp=now)

            percep = source.read(telem)
            if fast and percep is not None:
                percep = dataclasses.replace(percep, stamp=now)

            intent = policy.step(telem, percep, now)
            result = guard.check(intent, telem, percep, now)
            if result.force_phase is not None:
                policy.force(result.force_phase, now)

            fc.send(result.command, dt=dt)

            ticks += 1
            if result.vetoes:
                vetoed_ticks += 1

            record = {
                "t": round(virtual if fast else now - started, 2),
                "phase": policy.phase.value,
                "action": result.command.action.value,
                "vx": round(result.command.vx, 3),
                "vy": round(result.command.vy, 3),
                "vz": round(result.command.vz, 3),
                "yaw_rate": round(result.command.yaw_rate, 3),
                "n": round(telem.north_m, 2),
                "e": round(telem.east_m, 2),
                "alt": round(telem.alt_m, 2),
                "batt": round(telem.battery_pct, 1),
                "vetoes": list(result.vetoes),
                "reason": result.command.reason,
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()          # 대시보드/외부에서 실시간으로 읽을 수 있게

            if on_tick is not None:
                on_tick(record, telem, percep, result)

            if ticks % 5 == 0 or result.vetoes:
                flag = ("  ⚠ " + ",".join(result.vetoes)) if result.vetoes else ""
                print(
                    f"[{record['t']:6.1f}s] {policy.phase.value:<9}"
                    f" {result.command.action.value:<8}"
                    f" pos=({record['n']:+.2f},{record['e']:+.2f}) alt={record['alt']:.2f}"
                    f" batt={record['batt']:.0f}%  {result.command.reason}{flag}"
                )

            if policy.phase is Phase.DONE:
                break

            if should_stop is not None and should_stop():
                print("외부 중단 요청 — 루프 종료")
                break

            virtual += dt
            elapsed = virtual if fast else time.monotonic() - started
            if elapsed > max_seconds:
                print("최대 실행 시간 초과 — 중단")
                break

            if not fast:
                time.sleep(max(0.0, dt - (time.monotonic() - now)))
    finally:
        log.close()
        source.close()
        fc.close()

    summary = {
        "ticks": ticks,
        "vetoed_ticks": vetoed_ticks,
        "final_phase": policy.phase.value,
        "log": cfg.runtime.log_path,
        "counted": len(policy.counted),
        "observed": list(policy.observed),
    }
    print(
        f"\n완료: {ticks} tick, Guard 개입 {vetoed_ticks}회"
        f" ({vetoed_ticks / max(1, ticks) * 100:.1f}%), 최종 단계 {summary['final_phase']}"
    )
    print(f"비행 로그: {cfg.runtime.log_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="실내 온보드 자율 드론")
    p.add_argument("--perception", default="mock", choices=["mock", "oakd", "onnx"])
    p.add_argument(
        "--rangefinders", default="none", choices=["none", "vl53l1x"],
        help="좌·우·후방 ToF 배열. 전방은 OAK-D 뎁스가 담당한다",
    )
    p.add_argument("--link", default="mock", choices=["mock", "mavlink"])
    p.add_argument("--mavlink-url", default=Runtime().mavlink_url)
    p.add_argument("--log", default=Runtime().log_path)
    p.add_argument("--fast", action="store_true", help="실시간 대기 없이 실행")
    p.add_argument("--max-seconds", type=float, default=240.0)
    p.add_argument("--order", help='자연어 임무. 예: "의자 몇 개 있는지 세줘"')
    p.add_argument("--compiler", default="rules", choices=["rules", "claude"])
    p.add_argument(
        "--dry-run", action="store_true", help="명령만 컴파일하고 비행하지 않는다"
    )
    p.add_argument(
        "--dashboard", action="store_true",
        help="관제 대시보드를 띄운다. 명령 입력·상태·판단 근거를 한 화면에서 본다",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    cfg = build_config(args)

    if args.dashboard:
        from .dashboard import serve

        serve(cfg, host=args.host, port=args.port, fast=args.fast)
        return

    spec = None

    if args.order:
        result = compile_mission(args.order, cfg.limits, backend=args.compiler)
        print(f'명령: "{args.order}"')
        print(f"  백엔드   : {result.backend}")
        print(f"  임무     : {result.spec.behavior.value}")
        print(f"  대상     : {result.spec.target_label}  (최대 {result.spec.stop_after_n}개)")
        print(f"  정지 거리: {result.spec.stop_distance_m}m")
        print(f"  순항 고도: {result.spec.cruise_alt_m}m")
        for note in result.notes:
            print(f"  · {note}")

        if not result.ok:
            # 이륙 전에 거른다. 프로펠러가 돌기 전이라 아무것도 잃지 않는다.
            print("\n이 명령으로는 비행할 수 없습니다:")
            for problem in result.problems:
                print(f"  ❌ {problem}")
            raise SystemExit(1)

        print("  ✅ 검증 통과\n")
        spec = result.spec

    if args.dry_run:
        return

    run(cfg, fast=args.fast, max_seconds=args.max_seconds, spec=spec)


if __name__ == "__main__":
    main()
