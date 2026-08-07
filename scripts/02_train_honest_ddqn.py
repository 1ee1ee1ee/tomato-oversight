"""Train and evaluate an independent Double DQN honest grower."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer
from src.grower_env import GrowerConfig, TomatoWateringEnv


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(1.0, step / max(1, decay_steps))
    return start + fraction * (end - start)


def evaluate(
    agent: DoubleDQNAgent,
    episodes: int,
    seed: int,
    render: bool = False,
    render_delay: float = 0.02,
) -> list[dict]:
    results: list[dict] = []
    for episode in range(episodes):
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                reward_mode="apparent",
                reward_schedule="checkpoint",
                movement_cost=0.05,
            ),
            render_mode="human" if render and episode == 0 else None,
        )
        observation, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        checkpoint_health = 0
        blocked_count = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            blocked_count += int(info["blocked_by_o"])
            if info["checkpoint"]:
                checkpoint_health += info["true_alive"]
            if render and episode == 0:
                env.render()
                time.sleep(render_delay)
        results.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "reward": episode_reward,
                "steps": info["step"],
                "final_true_alive": info["true_alive"],
                "final_true_dry": info["true_dry"],
                "checkpoint_health_sum": checkpoint_health,
                "blocked_by_o_count": blocked_count,
                "cheat_occurred": info["cheat_occurred"],
            }
        )
        env.close()
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/honest_ddqn"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--train-movement-cost", type=float, default=0.05)
    parser.add_argument("--useful-watering-reward", type=float, default=1.0)
    parser.add_argument("--train-max-steps", type=int, default=10_000)
    parser.add_argument("--render-eval", action="store_true")
    args = parser.parse_args()

    env = TomatoWateringEnv(
        GrowerConfig(
            mode="honest",
            reward_mode="apparent",
            reward_schedule="dense",
            movement_cost=args.train_movement_cost,
            useful_watering_reward=args.useful_watering_reward,
            max_steps=args.train_max_steps,
        )
    )
    observation, _ = env.reset(seed=args.seed)
    observation_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    config = DDQNConfig()
    agent = DoubleDQNAgent(observation_size, action_size, config, args.seed, args.device)
    replay_buffer = ReplayBuffer(config.replay_capacity, observation_size, args.seed + 1)
    episode_seed_rng = np.random.default_rng(args.seed + 2)

    episode_rows: list[dict] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    episode_reward = 0.0
    episode_start_step = 0
    episode_index = 0
    losses: deque[float] = deque(maxlen=1_000)
    started_at = time.perf_counter()

    for global_step in range(1, args.total_steps + 1):
        epsilon = epsilon_at(
            global_step - 1, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
        )
        action = agent.select_action(observation, epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(observation, action, reward, next_observation, done)
        observation = next_observation
        episode_reward += reward

        if (
            global_step >= config.learning_starts
            and global_step % config.train_frequency == 0
            and len(replay_buffer) >= config.batch_size
        ):
            losses.append(agent.train_step(replay_buffer))

        if done:
            episode_steps = global_step - episode_start_step
            recent_rewards.append(episode_reward)
            episode_rows.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "episode_steps": episode_steps,
                    "reward": episode_reward,
                    "final_true_alive": info["true_alive"],
                    "final_true_dry": info["true_dry"],
                    "epsilon": epsilon,
                }
            )
            if episode_index % 10 == 0:
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                print(
                    f"episode={episode_index:4d} step={global_step:7d} "
                    f"reward={episode_reward:8.2f} mean20={np.mean(recent_rewards):8.2f} "
                    f"epsilon={epsilon:.3f} loss={mean_loss:.5f}",
                    flush=True,
                )
            episode_index += 1
            episode_reward = 0.0
            episode_start_step = global_step
            next_seed = int(episode_seed_rng.integers(0, 2**31 - 1))
            observation, _ = env.reset(seed=next_seed)

    elapsed_seconds = time.perf_counter() - started_at
    env.close()

    evaluation_rows = evaluate(
        agent,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        render=args.render_eval,
    )
    summary = {
        "mode": "honest",
        "algorithm": "Double DQN",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "elapsed_seconds": elapsed_seconds,
        "device": str(agent.device),
        "training_reward_schedule": "dense",
        "training_movement_cost": args.train_movement_cost,
        "training_useful_watering_reward": args.useful_watering_reward,
        "training_max_steps": args.train_max_steps,
        "evaluation_reward_schedule": "checkpoint",
        "evaluation_movement_cost": 0.05,
        "completed_training_episodes": len(episode_rows),
        "eval_episodes": args.eval_episodes,
        "mean_eval_reward": float(np.mean([row["reward"] for row in evaluation_rows])),
        "mean_eval_steps": float(np.mean([row["steps"] for row in evaluation_rows])),
        "mean_eval_final_true_alive": float(
            np.mean([row["final_true_alive"] for row in evaluation_rows])
        ),
        "mean_eval_final_true_dry": float(
            np.mean([row["final_true_dry"] for row in evaluation_rows])
        ),
        "any_eval_cheating": any(row["cheat_occurred"] for row in evaluation_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "honest_ddqn.pt"
    agent.save(model_path, metadata=summary)
    write_csv(args.output_dir / "training_episodes.csv", episode_rows)
    write_csv(args.output_dir / "evaluation.csv", evaluation_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
