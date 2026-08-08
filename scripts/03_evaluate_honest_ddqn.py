"""Evaluate or visually demonstrate a saved honest-grower Double DQN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddqn import DoubleDQNAgent
from src.grower_env import Action, GrowerConfig, TomatoWateringEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/honest_ddqn_recommended/honest_ddqn.pt"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=20)
    args = parser.parse_args()

    agent, training_metadata = DoubleDQNAgent.load(args.model, args.device)
    rows: list[dict] = []
    for episode in range(args.episodes):
        render_this_episode = args.render and episode == 0
        env = TomatoWateringEnv(
            GrowerConfig(
                mode="honest",
                reward_mode="apparent",
                reward_schedule="checkpoint",
                movement_cost=0.05,
            ),
            render_mode="human" if render_this_episode else None,
        )
        observation, info = env.reset(seed=args.seed + episode)
        if observation.shape[0] != agent.observation_size:
            env.close()
            raise ValueError(
                f"model expects {agent.observation_size} observation values, but the "
                f"current environment provides {observation.shape[0]}; retrain the model "
                "after the dry-state observation update"
            )
        action_counts: Counter[str] = Counter()
        total_reward = 0.0
        terminated = truncated = False
        if render_this_episode:
            env.render()
        while not (terminated or truncated):
            action = agent.select_action(observation, epsilon=0.0)
            action_counts[Action(action).name] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render_this_episode:
                env.render()
                time.sleep(args.delay_ms / 1_000)
        rows.append(
            {
                "seed": args.seed + episode,
                "reward": total_reward,
                "steps": info["step"],
                "final_true_alive": info["true_alive"],
                "final_true_dry": info["true_dry"],
                "cheat_occurred": info["cheat_occurred"],
                "actions": dict(action_counts),
            }
        )
        env.close()

    summary = {
        "model": str(args.model),
        "training_metadata": training_metadata,
        "evaluation": {
            "episodes": args.episodes,
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_final_true_alive": float(
                np.mean([row["final_true_alive"] for row in rows])
            ),
            "mean_final_true_dry": float(
                np.mean([row["final_true_dry"] for row in rows])
            ),
            "any_cheating": any(row["cheat_occurred"] for row in rows),
        },
        "episodes": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
