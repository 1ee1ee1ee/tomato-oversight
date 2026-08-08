"""Evaluate a trained cheat_v1 policy on true and apparent performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ddqn import DoubleDQNAgent
from src.env import CheatGrowerConfig
from train import evaluate, evaluation_summary, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", choices=("pure", "hybrid"), default="pure")
    parser.add_argument("--o-position", type=str, default="1,1")
    parser.add_argument("--episodes-per-o", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent, metadata = DoubleDQNAgent.load(args.model, args.device, args.seed)
    config_data = metadata.get("env_config")
    config = CheatGrowerConfig(**config_data) if config_data else CheatGrowerConfig(profile=args.profile)
    if args.o_position == "cycle":
        positions = list(config.o_candidates)
    else:
        position = tuple(int(value) for value in args.o_position.split(","))
        if position not in config.o_candidates:
            raise ValueError(f"invalid O position: {position}")
        positions = [position]
    rows = evaluate(
        agent,
        config,
        args.seed,
        positions,
        args.epsilon,
        args.episodes_per_o,
    )
    result = {
        "model": str(args.model),
        "profile": config.profile,
        "true_reward_weight": config.true_reward_weight,
        "apparent_reward_weight": config.apparent_reward_weight,
        "epsilon": args.epsilon,
        "summary": evaluation_summary(rows),
        "episodes": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            write_csv(args.output, rows)
        else:
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
