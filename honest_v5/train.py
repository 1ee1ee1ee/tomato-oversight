"""Train the honest grower with n-step Double DQN on the final reward rules."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer
from src.env import HonestGrowerConfig, HonestTomatoEnv


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(1.0, step / max(1, decay_steps))
    return start + fraction * (end - start)


class NStepAccumulator:
    """Convert exact environment rewards into fixed-horizon replay targets."""

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.pending: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()
        self.discounted_reward_sum = 0.0

    def append(self, transition):
        self.discounted_reward_sum += (self.gamma ** len(self.pending)) * transition[2]
        self.pending.append(transition)
        emitted = []
        if len(self.pending) >= self.n_steps:
            emitted.append(self._pop_one())
        if transition[4]:
            while self.pending:
                emitted.append(self._pop_one())
        return emitted

    def _pop_one(self):
        observation, action = self.pending[0][0], self.pending[0][1]
        reward_sum = self.discounted_reward_sum
        next_observation = self.pending[-1][3]
        done = self.pending[-1][4]
        used = len(self.pending)
        oldest_reward = self.pending.popleft()[2]
        if self.pending:
            self.discounted_reward_sum = (self.discounted_reward_sum - oldest_reward) / self.gamma
        else:
            self.discounted_reward_sum = 0.0
        bootstrap_discount = 0.0 if done else self.gamma ** used
        return observation, action, reward_sum, next_observation, bootstrap_discount


def evaluate(agent: DoubleDQNAgent, config: HonestGrowerConfig, seed: int) -> list[dict]:
    rows: list[dict] = []
    for episode, o_position in enumerate(config.o_candidates):
        env = HonestTomatoEnv(config)
        observation, _ = env.reset(seed=seed + episode, options={"o_position": o_position})
        total_reward = 0.0
        checkpoint_reward = 0.0
        water_attempts = 0
        successful_waters = 0
        blocked_moves = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            checkpoint_reward += info["checkpoint_reward"]
            water_attempts += int(action == 4)
            successful_waters += int(info["watered"])
            blocked_moves += int(info["blocked"])
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "o_position": str(o_position),
                "reward": total_reward,
                "checkpoint_reward": checkpoint_reward,
                "final_alive": info["true_alive"],
                "final_dry": info["true_dry"],
                "water_attempts": water_attempts,
                "successful_waters": successful_waters,
                "blocked_moves": blocked_moves,
            }
        )
        env.close()
    return rows


def evaluation_summary(rows: list[dict]) -> dict:
    return {
        "full_survival_rate": float(np.mean([row["final_alive"] == 5 for row in rows])),
        "mean_final_alive": float(np.mean([row["final_alive"] for row in rows])),
        "mean_final_dry": float(np.mean([row["final_dry"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_checkpoint_reward": float(np.mean([row["checkpoint_reward"] for row in rows])),
        "mean_water_attempts": float(np.mean([row["water_attempts"] for row in rows])),
        "mean_successful_waters": float(np.mean([row["successful_waters"] for row in rows])),
        "mean_blocked_moves": float(np.mean([row["blocked_moves"] for row in rows])),
    }


def score(summary: dict) -> tuple[float, ...]:
    return (
        summary["full_survival_rate"],
        summary["mean_final_alive"],
        -summary["mean_final_dry"],
        summary["mean_reward"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=int, default=600_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/honest_v5"))
    parser.add_argument("--resume-model", type=Path)
    args = parser.parse_args()

    if args.n_steps < 1:
        raise ValueError("--n-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_config = HonestGrowerConfig()
    env = HonestTomatoEnv(env_config)
    observation, info = env.reset(
        seed=args.seed,
        options={"o_position": env_config.o_candidates[0]},
    )
    ddqn_config = DDQNConfig()
    if args.resume_model:
        agent, _ = DoubleDQNAgent.load(args.resume_model, args.device, args.seed)
        if agent.observation_size != observation.shape[0] or agent.action_size != env.action_space.n:
            raise ValueError("resume model is incompatible with this v5 environment")
    else:
        agent = DoubleDQNAgent(
            int(observation.shape[0]),
            int(env.action_space.n),
            ddqn_config,
            args.seed,
            args.device,
        )
    replay = ReplayBuffer(ddqn_config.replay_capacity, agent.observation_size, args.seed + 1)
    accumulator = NStepAccumulator(args.n_steps, ddqn_config.gamma)
    episode_seed_rng = np.random.default_rng(args.seed + 2)

    episode_rows: list[dict] = []
    periodic_rows: list[dict] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    recent_losses: deque[float] = deque(maxlen=1_000)
    episode_reward = 0.0
    episode_checkpoint_reward = 0.0
    episode_index = 0
    water_attempts = successful_waters = blocked_moves = 0
    best_score = None
    best_step = 0
    best_path = args.output_dir / "honest_v5_best.pt"
    started_at = time.perf_counter()

    for global_step in range(1, args.total_steps + 1):
        epsilon = epsilon_at(
            global_step - 1,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )
        action = agent.select_action(observation, epsilon)
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        for transition in accumulator.append(
            (observation, action, reward, next_observation, done)
        ):
            replay.add(*transition)

        observation = next_observation
        episode_reward += reward
        episode_checkpoint_reward += step_info["checkpoint_reward"]
        water_attempts += int(action == 4)
        successful_waters += int(step_info["watered"])
        blocked_moves += int(step_info["blocked"])

        if (
            global_step >= ddqn_config.learning_starts
            and global_step % ddqn_config.train_frequency == 0
            and len(replay) >= ddqn_config.batch_size
        ):
            recent_losses.append(agent.train_step(replay))

        if done:
            recent_rewards.append(episode_reward)
            episode_rows.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "o_position": str(step_info["o_position"]),
                    "reward": episode_reward,
                    "checkpoint_reward": episode_checkpoint_reward,
                    "final_alive": step_info["true_alive"],
                    "final_dry": step_info["true_dry"],
                    "water_attempts": water_attempts,
                    "successful_waters": successful_waters,
                    "blocked_moves": blocked_moves,
                    "epsilon": epsilon,
                }
            )
            print(
                f"episode={episode_index:4d} step={global_step:8d} "
                f"alive={step_info['true_alive']} dry={step_info['true_dry']} "
                f"reward={episode_reward:9.2f} mean20={np.mean(recent_rewards):9.2f} "
                f"epsilon={epsilon:.3f}",
                flush=True,
            )
            episode_index += 1
            episode_reward = episode_checkpoint_reward = 0.0
            water_attempts = successful_waters = blocked_moves = 0
            next_o = env_config.o_candidates[episode_index % len(env_config.o_candidates)]
            next_seed = int(episode_seed_rng.integers(0, 2**31 - 1))
            observation, info = env.reset(seed=next_seed, options={"o_position": next_o})

        if global_step % args.eval_interval == 0:
            rows = evaluate(agent, env_config, args.eval_seed)
            summary = evaluation_summary(rows)
            periodic_rows.append({"global_step": global_step, **summary})
            current_score = score(summary)
            print("evaluation", global_step, json.dumps(summary, ensure_ascii=False), flush=True)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_step = global_step
                agent.save(best_path, metadata={"training_step": global_step, "evaluation": summary})
                write_csv(args.output_dir / "best_evaluation.csv", rows)

        if global_step % args.save_interval == 0:
            agent.save(
                args.output_dir / "honest_v5_last.pt",
                metadata={"training_step": global_step},
            )
            write_csv(args.output_dir / "training_episodes.csv", episode_rows)
            write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)

    elapsed_seconds = time.perf_counter() - started_at
    env.close()
    agent.save(args.output_dir / "honest_v5_last.pt", metadata={"training_step": args.total_steps})
    if not best_path.exists():
        agent.save(best_path, metadata={"training_step": args.total_steps})
        best_step = args.total_steps

    best_agent, _ = DoubleDQNAgent.load(best_path, args.device, args.seed)
    final_rows = evaluate(best_agent, env_config, args.eval_seed)
    final_summary = evaluation_summary(final_rows)
    summary = {
        "algorithm": "512-step Double DQN" if args.n_steps == 512 else f"{args.n_steps}-step Double DQN",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "best_training_step": best_step,
        "elapsed_seconds": elapsed_seconds,
        "device": str(agent.device),
        "environment": env_config.to_dict(),
        "ddqn": {
            "gamma": ddqn_config.gamma,
            "n_steps": args.n_steps,
            "hidden_sizes": ddqn_config.hidden_sizes,
            "learning_rate": ddqn_config.learning_rate,
        },
        "evaluation": final_summary,
    }
    best_agent.save(best_path, metadata=summary)
    write_csv(args.output_dir / "training_episodes.csv", episode_rows)
    write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)
    write_csv(args.output_dir / "final_evaluation.csv", final_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"best model: {best_path}")


if __name__ == "__main__":
    main()
