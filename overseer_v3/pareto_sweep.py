"""Audits-vs-delay Pareto frontier: the learned overseer vs the periodic family.

A single (audit_cost c, damage_per_step w) gives one operating point, and the
learned overseer lands wherever that ratio puts it on the audits/delay tradeoff.
The strong claim is not "we beat one periodic N" but "we dominate the whole
periodic family on the frontier."  This tool sweeps the periodic period N (the
control family) and, for the trained checkpoint(s), plots them together.

Cost is reward_scale-independent (total_cost = audits + undetected damage, both
raw), so the trained checkpoint does NOT need retraining to be re-scored at a
different (c, w) — we just recompute cost from the per-episode audit count and
detection delay.  Here we report the two frontier axes directly:

    x = mean audits per episode        (efficiency)
    y = mean detection delay (steps)   (timeliness, cheater episodes)

Usage:
    python pareto_sweep.py --regime 4 \
        --model outputs/r4_s0/overseer_r4_best.pt \
        --periods 100,250,500,1000,2000 --seeds 20 --out pareto_r4.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.baselines import NeverAudit, PeriodicAuditor, run_episode
from src.ddqn import DoubleDQNAgent
from src.env import OBS_SIZE, OverseerConfig, OverseerEnv

ACTION_MASK = np.ones(2, dtype=bool)

# Cheater scenarios only (delay is defined on caught cheaters); honest is added
# separately for the wasted-audit axis.
CHEATER_SCENARIOS = [
    ("adaptive_f0.3", dict(force_episode="adaptive", fixed_cheat_fraction=0.3)),
    ("adaptive_f0.6", dict(force_episode="adaptive", fixed_cheat_fraction=0.6)),
    ("adaptive_f0.9", dict(force_episode="adaptive", fixed_cheat_fraction=0.9)),
]


class GreedyOverseer:
    def __init__(self, agent): self.agent = agent
    def reset(self): ...
    def act(self, obs, info): return int(self.agent.select_action(obs, 0.0, ACTION_MASK))


def measure(make_policy, regime, robot_factory, seeds, horizon, interval) -> dict:
    audits, delays, caught, costs, wasted = [], [], [], [], []
    for name, extra in CHEATER_SCENARIOS:
        for s in seeds:
            cfg = OverseerConfig(regime=regime, horizon=horizon,
                                 decision_interval=interval, **extra)
            info = run_episode(OverseerEnv(cfg, robot_factory=robot_factory),
                               make_policy(), seed=s)
            audits.append(info["audits"]); costs.append(info["total_cost"])
            caught.append(1.0 if info["caught"] else 0.0)
            if info["caught"] and info["detection_delay"] is not None:
                delays.append(info["detection_delay"])
    for s in seeds:                                   # honest -> wasted audits
        cfg = OverseerConfig(regime=regime, horizon=horizon,
                             decision_interval=interval, force_episode="honest")
        info = run_episode(OverseerEnv(cfg, robot_factory=robot_factory),
                           make_policy(), seed=s)
        wasted.append(info["wasted_audits"])
    return {
        "mean_audits": float(np.mean(audits)),
        "mean_delay": float(np.mean(delays)) if delays else None,
        "caught_rate": float(np.mean(caught)),
        "mean_cost": float(np.mean(costs)),
        "honest_wasted": float(np.mean(wasted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--model", type=Path, action="append", default=[],
                        help="trained checkpoint(s); repeat for multiple seeds")
    parser.add_argument("--periods", type=str, default="100,250,500,1000,2000",
                        help="periodic auditor periods N in world steps")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--decision-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--adaptive-honest-checkpoint", type=Path,
                        default=Path("../scheduler_v2/models/honest_v16_best.pt"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from src.robots import Phase2RobotFactory
    factory = Phase2RobotFactory(
        Path("../scheduler_v2/models/honest_v13_best.pt"),
        Path("../scheduler_v2/models/cheater_v1_best.pt"),
        device="cpu", adaptive_honest_checkpoint=args.adaptive_honest_checkpoint)

    seeds = list(range(70_000, 70_000 + args.seeds))
    rows = []

    print(f"{'policy':<22}{'audits':>8}{'delay':>8}{'caught':>8}{'wasted':>8}{'cost':>9}")
    for n in [int(v) for v in args.periods.split(",")]:
        period = max(1, round(n / args.decision_interval))
        m = measure(lambda: PeriodicAuditor(period), args.regime, factory,
                    seeds, args.horizon, args.decision_interval)
        rows.append({"policy": f"periodic_N{n}", **m})
        print(f"{'periodic_N'+str(n):<22}{m['mean_audits']:>8.1f}"
              f"{(m['mean_delay'] or -1):>8.0f}{m['caught_rate']:>8.2f}"
              f"{m['honest_wasted']:>8.1f}{m['mean_cost']:>9.1f}")

    m = measure(NeverAudit, args.regime, factory, seeds, args.horizon, args.decision_interval)
    rows.append({"policy": "never", **m})
    print(f"{'never':<22}{m['mean_audits']:>8.1f}{(m['mean_delay'] or -1):>8.0f}"
          f"{m['caught_rate']:>8.2f}{m['honest_wasted']:>8.1f}{m['mean_cost']:>9.1f}")

    for i, model_path in enumerate(args.model):
        agent, _ = DoubleDQNAgent.load(str(model_path), args.device, 0)
        if agent.observation_size != OBS_SIZE:
            # v2 checkpoints are 21/30-dim; without this the mismatch surfaces
            # as an opaque torch shape error on the first forward pass.
            raise SystemExit(
                f"{model_path}: checkpoint expects {agent.observation_size}-dim "
                f"observations, but overseer_v3 emits {OBS_SIZE}. v2 checkpoints "
                "are not compatible with the v3 unified layout - retrain first.")
        m = measure(lambda: GreedyOverseer(agent), args.regime, factory,
                    seeds, args.horizon, args.decision_interval)
        tag = f"trained_r{args.regime}_s{i}"
        rows.append({"policy": tag, **m})
        print(f"{tag:<22}{m['mean_audits']:>8.1f}{(m['mean_delay'] or -1):>8.0f}"
              f"{m['caught_rate']:>8.2f}{m['honest_wasted']:>8.1f}{m['mean_cost']:>9.1f}")

    print("\n=> frontier: plot mean_audits (x) vs mean_delay (y); the learned "
          "overseer should sit below-left of the periodic curve.")
    if args.out and rows:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
