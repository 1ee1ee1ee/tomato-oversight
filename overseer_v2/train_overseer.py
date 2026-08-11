"""Train the overseer DQN (regime 1 or 2) against frozen robots.

The robot policies (honest/betrayal/adaptive) are a *fixed* part of the
environment — only the overseer learns.  ``--robots scripted`` (Phase 1) uses
the scripted robots; ``--robots phase2`` uses the v3.2 role-reassigned robots
(``src/robots.py``): learned honest for honest/betrayal episodes, learned
cheater after K, scripted pair for the adaptive treatment axis.

Outputs (to --output-dir):
  overseer_r{regime}_best.pt        best model by evaluation total-cost
  overseer_r{regime}_last.pt
  periodic_evaluation.csv           learning curve vs the periodic baseline
  summary.json                      final trained vs periodic/always/never table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.baselines import (
    AlwaysAudit,
    NeverAudit,
    PeriodicAuditor,
    period_from_world_steps,
    run_episode,
)
from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer
from src.env import OverseerAction, OverseerConfig, OverseerEnv, default_robot_factory

# Fixed evaluation grid (rule v3.4 scope): honest control + adaptive
# deception-intensity sweep.  The adaptive robot is the scheduler: it switches
# between its honest model and its cheater on its own, state-based.
EVAL_SCENARIOS = [
    ("honest", dict(force_episode="honest")),
    ("adaptive_f0.3", dict(force_episode="adaptive", fixed_cheat_fraction=0.3)),
    ("adaptive_f0.6", dict(force_episode="adaptive", fixed_cheat_fraction=0.6)),
    ("adaptive_f0.9", dict(force_episode="adaptive", fixed_cheat_fraction=0.9)),
]

# Held-out deception intensities (off the training-eval grid) — final report
# only, to show the overseer learned a principle rather than the grid.
HELD_OUT_SCENARIOS = [
    ("adaptive_f0.45_held", dict(force_episode="adaptive", fixed_cheat_fraction=0.45)),
    ("adaptive_f0.75_held", dict(force_episode="adaptive", fixed_cheat_fraction=0.75)),
]

# Removed scope (rule v3.4): the one-way betrayal robot was cut from the main
# experiment.  Enable with --include-betrayal to add these back.
BETRAYAL_SCENARIOS = [
    ("betrayal_k0", dict(force_episode="betrayal", fixed_k=0)),
    ("betrayal_k2000", dict(force_episode="betrayal", fixed_k=2000)),
    ("betrayal_k4000", dict(force_episode="betrayal", fixed_k=4000)),
    ("betrayal_k6000", dict(force_episode="betrayal", fixed_k=6000)),
]

ACTION_MASK = np.ones(2, dtype=bool)   # CONTINUE and AUDIT are always valid


class GreedyOverseer:
    """Wrap a trained agent as an overseer policy for run_episode/eval."""

    def __init__(self, agent: DoubleDQNAgent, epsilon: float = 0.0) -> None:
        self.agent = agent
        self.epsilon = epsilon

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(self.agent.select_action(obs, self.epsilon, ACTION_MASK))


def epsilon_at(step: int, start: float, end: float, decay: int) -> float:
    if step >= decay:
        return end
    return start + (end - start) * step / decay


def eval_policy(make_policy, regime: int, horizon: int, interval: int, seeds,
                robot_factory=default_robot_factory,
                scenarios=EVAL_SCENARIOS) -> dict:
    """Run every scenario x seed with a fresh policy; aggregate the 3 metrics."""
    per_scen = {}
    all_costs, all_audits = [], []
    for name, extra in scenarios:
        costs, audits, delays, caught, wasted = [], [], [], [], []
        for s in seeds:
            cfg = OverseerConfig(regime=regime, horizon=horizon,
                                 decision_interval=interval, **extra)
            info = run_episode(OverseerEnv(cfg, robot_factory=robot_factory),
                               make_policy(), seed=s)
            costs.append(info["total_cost"])
            audits.append(info["audits"])
            if info["is_cheater"]:
                caught.append(1.0 if info["caught"] else 0.0)
                if info["caught"] and info["detection_delay"] is not None:
                    delays.append(info["detection_delay"])
            else:
                wasted.append(info["wasted_audits"])
        per_scen[name] = {
            "cost": float(np.mean(costs)),
            "audits": float(np.mean(audits)),
            "caught_rate": float(np.mean(caught)) if caught else None,
            "detection_delay": float(np.mean(delays)) if delays else None,
            "wasted_audits": float(np.mean(wasted)) if wasted else None,
        }
        all_costs.extend(costs)
        all_audits.extend(audits)

    def ci95(values):
        # 95% CI half-width of the mean (t~1.96, normal approx over eval seeds).
        arr = np.asarray(values, dtype=float)
        return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "mean_cost": float(np.mean(all_costs)),
        "cost_ci95": ci95(all_costs),
        "mean_audits": float(np.mean(all_audits)),
        "audits_ci95": ci95(all_audits),
        "per_scenario": per_scen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--total-steps", type=int, default=150_000)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--decision-interval", type=int, default=100)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--epsilon-decay", type=int, default=50_000)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--eval-seeds", type=int, default=6)
    parser.add_argument("--final-eval-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overseer_v2"))
    # Robot layer (frozen opponents).  "phase2" = v3.2 role reassignment.
    parser.add_argument("--robots", choices=("scripted", "phase2"), default="phase2")
    parser.add_argument("--honest-checkpoint", type=Path,
                        default=Path("../scheduler_v2/models/honest_v13_best.pt"))
    parser.add_argument("--cheater-checkpoint", type=Path,
                        default=Path("../scheduler_v2/models/cheater_v1_best.pt"))
    # Episode mix (rule v3.4): the main experiment runs TWO robot kinds —
    # honest and the adaptive (scheduler) cheater.  Betrayal was cut.
    parser.add_argument("--honest-prob", type=float, default=0.5)
    parser.add_argument("--adaptive-prob", type=float, default=0.5)
    # "balanced" cycles the enabled robot kinds deterministically so training
    # sees EXACTLY equal episode counts per kind (no binomial drift).
    parser.add_argument("--episode-schedule", choices=("random", "balanced"),
                        default="balanced")
    parser.add_argument("--warmup-episodes", type=int, default=8,
                        help="AlwaysAudit episodes to seed the replay (escapes the never-audit trap)")
    parser.add_argument("--include-betrayal", action="store_true",
                        help="add the betrayal K-sweep back into every evaluation")
    # The adaptive robot's honest half (swappable checkpoint, rule v3.4).
    parser.add_argument("--adaptive-honest-checkpoint", type=Path,
                        default=Path("../scheduler_v2/models/honest_v16_best.pt"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_scenarios = EVAL_SCENARIOS + (BETRAYAL_SCENARIOS if args.include_betrayal else [])

    if args.robots == "phase2":
        from src.robots import Phase2RobotFactory
        robot_factory = Phase2RobotFactory(
            args.honest_checkpoint, args.cheater_checkpoint, device="cpu",
            adaptive_honest_checkpoint=args.adaptive_honest_checkpoint,
        )
        print("robots:", robot_factory.describe())
    else:
        robot_factory = default_robot_factory

    env = OverseerEnv(OverseerConfig(
        regime=args.regime, horizon=args.horizon,
        decision_interval=args.decision_interval, reward_scale=args.reward_scale,
        honest_prob=args.honest_prob, adaptive_prob=args.adaptive_prob,
        episode_schedule=args.episode_schedule,
    ), robot_factory=robot_factory)
    obs_size = env.observation_space.shape[0]
    ddqn = DDQNConfig(hidden_sizes=(128, 128), learning_rate=args.lr,
                      gamma=args.gamma, learning_starts=args.learning_starts,
                      train_frequency=args.train_frequency)
    agent = DoubleDQNAgent(obs_size, 2, ddqn, args.seed, args.device)
    replay = ReplayBuffer(ddqn.replay_capacity, obs_size, 2, args.seed)

    eval_seeds = list(range(90_000, 90_000 + args.eval_seeds))
    period = period_from_world_steps(500, args.decision_interval)
    baseline = eval_policy(lambda: PeriodicAuditor(period), args.regime,
                           args.horizon, args.decision_interval, eval_seeds,
                           robot_factory=robot_factory, scenarios=eval_scenarios)
    print(f"[regime {args.regime}] obs={obs_size}  device={agent.device}  "
          f"periodic(N=500) baseline mean_cost={baseline['mean_cost']:.2f}")

    best_path = args.output_dir / f"overseer_r{args.regime}_best.pt"
    last_path = args.output_dir / f"overseer_r{args.regime}_last.pt"
    best_cost = float("inf")
    periodic_rows = []
    start_time = time.time()

    # Warm the replay with AlwaysAudit transitions so the buffer contains
    # "audit -> catch -> episode ends -> no more damage" experiences before
    # learning starts.  This does NOT bias the learned policy (the agent still
    # chooses its own actions from step 1); it only ensures the never-audit
    # local optimum is not the only thing in the buffer.  Without it, escape
    # from never-audit took 45k steps in r2_run2 and could fail on some seeds.
    if args.warmup_episodes > 0:
        warm = env.reset(seed=args.seed + 999)[0]
        for _ in range(args.warmup_episodes):
            done = False
            while not done:
                a = int(OverseerAction.AUDIT)
                nxt, r, term, trunc, _ = env.step(a)
                done = term or trunc
                replay.add(warm, a, r, nxt, 0.0 if done else ddqn.gamma, ACTION_MASK)
                warm = nxt if not done else env.reset()[0]

    obs, info = env.reset(seed=args.seed)
    ep_cost, ep_return = 0.0, 0.0
    recent_costs, recent_caught, recent_cheater = [], [], []

    for step in range(1, args.total_steps + 1):
        eps = epsilon_at(step, 1.0, args.epsilon_end, args.epsilon_decay)
        action = agent.select_action(obs, eps, ACTION_MASK)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        bootstrap = 0.0 if done else ddqn.gamma
        replay.add(obs, action, reward, next_obs, bootstrap, ACTION_MASK)
        ep_return += reward

        if len(replay) >= args.learning_starts and step % args.train_frequency == 0:
            agent.train_step(replay)

        if done:
            recent_costs.append(info["total_cost"])
            if info["is_cheater"]:
                recent_cheater.append(1.0)
                recent_caught.append(1.0 if info["caught"] else 0.0)
            obs, info = env.reset()
            ep_return = 0.0
        else:
            obs = next_obs

        if step % args.eval_interval == 0 or step == args.total_steps:
            metrics = eval_policy(lambda: GreedyOverseer(agent), args.regime,
                                  args.horizon, args.decision_interval, eval_seeds,
                                  robot_factory=robot_factory, scenarios=eval_scenarios)
            row = {
                "step": step,
                "trained_cost": metrics["mean_cost"],
                "trained_audits": metrics["mean_audits"],
                "periodic_cost": baseline["mean_cost"],
                "periodic_audits": baseline["mean_audits"],
                "recent_caught_rate": float(np.mean(recent_caught)) if recent_caught else None,
                "epsilon": eps,
            }
            periodic_rows.append(row)
            print(f"  step {step:>7} | trained_cost {metrics['mean_cost']:8.2f} "
                  f"(audits {metrics['mean_audits']:5.1f}) vs periodic {baseline['mean_cost']:7.2f} "
                  f"| caught(recent) {row['recent_caught_rate']}", flush=True)
            recent_caught, recent_cheater = [], []
            # Written every evaluation so a Drive output dir shows live progress.
            _write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)
            agent.save(last_path, metadata={"step": step})
            if metrics["mean_cost"] < best_cost:
                best_cost = metrics["mean_cost"]
                agent.save(best_path, metadata={"step": step, "eval": metrics})

    agent.save(last_path, metadata={"step": args.total_steps})

    # ---- Final comparison: trained (best) vs the control group ----
    # Periodic N sweep = the rule's 대조군 (촘촘~성김); held-out betrayal
    # timings included so the report shows principle-learning, not timetable
    # memorisation.
    best_agent, _ = DoubleDQNAgent.load(best_path, args.device, args.seed)
    seeds = list(range(70_000, 70_000 + args.final_eval_seeds))
    contenders = {
        "trained": lambda: GreedyOverseer(best_agent),
        "periodic_N250": lambda: PeriodicAuditor(
            period_from_world_steps(250, args.decision_interval)),
        "periodic_N500": lambda: PeriodicAuditor(period),
        "periodic_N1000": lambda: PeriodicAuditor(
            period_from_world_steps(1000, args.decision_interval)),
        "always": AlwaysAudit,
        "never": NeverAudit,
    }
    final_scenarios = eval_scenarios + HELD_OUT_SCENARIOS
    final = {name: eval_policy(make, args.regime, args.horizon,
                               args.decision_interval, seeds,
                               robot_factory=robot_factory,
                               scenarios=final_scenarios)
             for name, make in contenders.items()}

    summary = {
        "regime": args.regime,
        "total_steps": args.total_steps,
        "elapsed_seconds": time.time() - start_time,
        "best_eval_cost": best_cost,
        "robots": args.robots,
        "episode_mix": {"honest_prob": args.honest_prob,
                        "adaptive_prob": args.adaptive_prob,
                        "episode_schedule": args.episode_schedule,
                        "adaptive_honest_checkpoint": str(args.adaptive_honest_checkpoint),
                        "note": "rule v3.4: main experiment = honest + adaptive(scheduler), equal episode counts"},
        "checkpoints": {
            "honest": str(args.honest_checkpoint),
            "cheater": str(args.cheater_checkpoint),
        } if args.robots == "phase2" else None,
        "config": {"gamma": args.gamma, "reward_scale": args.reward_scale,
                   "lr": args.lr, "horizon": args.horizon,
                   "decision_interval": args.decision_interval},
        "final_comparison": final,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    _write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)

    print(f"\n=== FINAL (regime {args.regime}, {args.final_eval_seeds} seeds/scenario) ===")
    print(f"{'policy':<15}{'mean_cost':>11}{'mean_audits':>13}")
    for name, m in final.items():
        print(f"{name:<15}{m['mean_cost']:>11.2f}{m['mean_audits']:>13.1f}")
    print(f"\nsaved: {best_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join("" if row[k] is None else str(row[k]) for k in keys))
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
