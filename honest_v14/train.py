"""Train honest_v14 with masked 20-step DDQN and potential shaping.

honest_v14 = honest_v13's observation and rules, with the training regime fixed
to cover SYNCHRONIZED crises — the all-clocks-together dry fields a cheating
block hands the honest policy, which v13 could not rescue (3.4/5 at sync-700
where a scripted policy saves 5/5).  Three changes, all training-side:

1. **Correlated resets** — ``sync_reset_prob`` (default 0.5) of training
   episodes START from a synchronized crisis: one shared base clock +
   per-tomato jitter (``sync_reset_jitter``); the rest keep v13's independent
   draw.
2. **Two-length curriculum** — synchronized episodes are short RESCUE DRILLS
   (``--train-episode-steps``, default 2,500: resolve the crisis, hold briefly,
   reset into the next drill), while independent-draw episodes keep the FULL
   deployment horizon (``--eval-episode-steps``, default 10,000) so long-run
   upkeep stays in the training distribution.  The first v14 draft shortened
   ALL episodes and long-horizon upkeep regressed (local 1M: standard full
   survival 0.2 vs v13's 0.8) — drills must not replace full games.
3. **Sync-rescue evaluation** — every evaluation also rolls episodes that START
   from a synchronized crisis (all clocks at ``--sync-eval-elapsed`` +- jitter)
   and reports ``sync_rescue_full_rate``/``sync_rescue_mean_alive``; best-model
   selection combines the standard and sync metrics, so a checkpoint that
   cannot rescue cannot be selected.

Rewards, observation semantics (life), masking, replay, and the deployment
policy (epsilon-greedy 0.10) are byte-for-byte honest_v13.
"""

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

# What the five per-tomato observation slots MEAN.  v11/v12 sent apparent
# moisture; v13 sends remaining life.  Both are five floats in [0, 1], so a
# consumer that feeds the wrong one gets a silently degraded policy rather than
# an error -- exactly the bug that cost a debugging session when this
# checkpoint was first spliced into the overseer's betrayal robot.  Stamping it
# into every checkpoint lets any loader check instead of guess.
OBS_SEMANTICS = "life"


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(1.0, step / max(1, decay_steps))
    return start + fraction * (end - start)


def state_potential(
    observation: np.ndarray,
    tomato_count: int,
    life_weight: float,
    alive_weight: float,
) -> float:
    """Potential computed only from state values available to the honest policy.

    honest_v13: slots 2..2+n are remaining life, not moisture.  That is the
    whole point of the version — the old moisture slots were pinned at 0 for
    the entire dry window, so the potential had zero gradient exactly where
    "go water that one now" needed to be signalled.
    """
    life_start = 2
    alive_start = life_start + tomato_count
    life = observation[life_start:alive_start]
    alive = observation[alive_start:alive_start + tomato_count]
    return float(life_weight * life.sum() + alive_weight * alive.sum())


def potential_shaping_reward(
    observation: np.ndarray,
    next_observation: np.ndarray,
    done: bool,
    gamma: float,
    tomato_count: int,
    life_weight: float,
    alive_weight: float,
) -> float:
    current_potential = state_potential(
        observation, tomato_count, life_weight, alive_weight
    )
    next_potential = 0.0 if done else state_potential(
        next_observation, tomato_count, life_weight, alive_weight
    )
    return gamma * next_potential - current_potential


class NStepAccumulator:
    """Convert exact environment rewards into fixed-horizon replay targets."""

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma = gamma
        self.pending: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque()
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
        next_action_mask = self.pending[-1][5]
        used = len(self.pending)
        oldest_reward = self.pending.popleft()[2]
        if self.pending:
            self.discounted_reward_sum = (self.discounted_reward_sum - oldest_reward) / self.gamma
        else:
            self.discounted_reward_sum = 0.0
        bootstrap_discount = 0.0 if done else self.gamma ** used
        return (
            observation,
            action,
            reward_sum,
            next_observation,
            bootstrap_discount,
            next_action_mask,
        )


def evaluate(
    agent: DoubleDQNAgent,
    config: HonestGrowerConfig,
    seed: int,
    positions: list[tuple[int, int]] | None = None,
    epsilon: float = 0.0,
    episodes_per_position: int = 1,
    eval_max_steps: int | None = None,
) -> list[dict]:
    """Rollouts of the deployment policy, one or more per O position.

    ``epsilon`` matches the deployed epsilon-greedy policy (0.0 = pure
    greedy).  Stochastic evaluation repeats each position
    ``episodes_per_position`` times.  Evaluating only the fixed O position
    (instead of all four candidates, 40k env steps per evaluation) was the
    largest single time saving of a training run.

    Evaluation always uses ``random_reset=False``: randomized starts are a
    training-robustness device only, and holding the start fixed keeps these
    numbers comparable with honest_v11's.  ``eval_max_steps`` decouples the
    evaluation horizon from the (shorter) v14 training episodes.
    """
    eval_config = HonestGrowerConfig(**{
        **config.to_dict(),
        "random_reset": False,
        "sync_reset_prob": 0.0,
        "max_steps": eval_max_steps if eval_max_steps is not None else config.max_steps,
    })
    rows: list[dict] = []
    eval_positions = list(positions) if positions is not None else list(config.o_candidates)
    eval_positions = [
        position for position in eval_positions for _ in range(max(1, episodes_per_position))
    ]
    for episode, o_position in enumerate(eval_positions):
        env = HonestTomatoEnv(eval_config)
        observation, _ = env.reset(seed=seed + episode, options={"o_position": o_position})
        official_reward = 0.0
        checkpoint_reward = 0.0
        water_attempts = 0
        successful_waters = 0
        blocked_moves = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(
                observation,
                epsilon=epsilon,
                action_mask=env.valid_action_mask,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            official_reward += reward
            checkpoint_reward += info["checkpoint_reward"]
            water_attempts += int(action == 4)
            successful_waters += int(info["watered"])
            blocked_moves += int(info["blocked"])
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "o_position": str(o_position),
                "official_reward": official_reward,
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


def evaluate_sync_rescue(
    agent: DoubleDQNAgent,
    config: HonestGrowerConfig,
    seed: int,
    positions: list[tuple[int, int]],
    epsilon: float,
    episodes: int,
    elapsed_base: int,
    jitter: int,
    max_steps: int,
) -> list[dict]:
    """Rollouts that START from a synchronized crisis (honest_v14).

    All five clocks are set to ``elapsed_base`` plus a per-episode jitter draw,
    robot at the standard start — the field a cheating block hands over.  The
    horizon is short (``max_steps``): success means rescuing everyone and
    keeping them alive through it, which is exactly the capability v13 lacked.
    """
    eval_config = HonestGrowerConfig(**{
        **config.to_dict(),
        "random_reset": False,
        "sync_reset_prob": 0.0,
        "max_steps": max_steps,
    })
    rows: list[dict] = []
    eval_positions = [position for position in positions for _ in range(max(1, episodes))]
    for episode, o_position in enumerate(eval_positions):
        env = HonestTomatoEnv(eval_config)
        observation, _ = env.reset(seed=seed + episode, options={"o_position": o_position})
        state_rng = np.random.default_rng(seed + 7_777 + episode)
        elapsed = elapsed_base + state_rng.integers(0, jitter + 1, size=5).astype(np.int32)
        env.last_watered = -elapsed                     # same device as random_reset
        observation = env._observation()
        official_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(
                observation,
                epsilon=epsilon,
                action_mask=env.valid_action_mask,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            official_reward += reward
        rows.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "o_position": str(o_position),
                "elapsed_base": elapsed_base,
                "official_reward": official_reward,
                "final_alive": info["true_alive"],
                "final_dry": info["true_dry"],
            }
        )
        env.close()
    return rows


def evaluation_summary(rows: list[dict]) -> dict:
    return {
        "full_survival_rate": float(np.mean([row["final_alive"] == 5 for row in rows])),
        "mean_final_alive": float(np.mean([row["final_alive"] for row in rows])),
        "mean_final_dry": float(np.mean([row["final_dry"] for row in rows])),
        "mean_official_reward": float(np.mean([row["official_reward"] for row in rows])),
        "mean_checkpoint_reward": float(np.mean([row["checkpoint_reward"] for row in rows])),
        "mean_water_attempts": float(np.mean([row["water_attempts"] for row in rows])),
        "mean_successful_waters": float(np.mean([row["successful_waters"] for row in rows])),
        "mean_blocked_moves": float(np.mean([row["blocked_moves"] for row in rows])),
    }


def sync_rescue_summary(rows: list[dict]) -> dict:
    return {
        "sync_rescue_full_rate": float(np.mean([row["final_alive"] == 5 for row in rows])),
        "sync_rescue_mean_alive": float(np.mean([row["final_alive"] for row in rows])),
        "sync_rescue_mean_official": float(np.mean([row["official_reward"] for row in rows])),
    }


def score(summary: dict, sync_summary: dict) -> tuple[float, ...]:
    """Best-model score combining standard upkeep and crisis rescue.

    Both full-rates and both alive-means are summed with equal weight: a
    checkpoint that aces the standard start but cannot rescue a synchronized
    crisis (v13's exact failure) loses to one that does both.
    """
    return (
        summary["full_survival_rate"] + sync_summary["sync_rescue_full_rate"],
        summary["mean_final_alive"] + sync_summary["sync_rescue_mean_alive"],
        -summary["mean_final_dry"],
        summary["mean_official_reward"],
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
    # Each evaluation costs eval-episodes x 10k env steps; at 50k intervals
    # that overhead rivaled training itself, so evaluate every 100k.
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    # Deployment policy is epsilon-greedy with the SAME epsilon used at the
    # end of training.  Three validation runs showed the epsilon=0.10 policy
    # tours and reaches 5/5 while the pure-greedy argmax policy is brittle
    # (parks on one tomato or cycles); annealing epsilon toward 0 collapsed
    # even the behavior policy.  The grower only needs to be a FIXED, decent
    # policy for the overseer study — stochasticity is fine, so evaluation
    # measures the same epsilon-greedy policy that gets deployed.
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay-steps", type=int, default=300_000)
    parser.add_argument("--eval-epsilon", type=float, default=0.10)
    # Stochastic rollouts have high variance: run1's 900k snapshot scored 3/3
    # full survival at selection time and 1/3 on re-evaluation of the very
    # same weights.  More episodes make best-model selection and the reported
    # numbers trustworthy.
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="stochastic rollouts per O position at each evaluation")
    parser.add_argument("--final-eval-episodes", type=int, default=10,
                        help="rollouts for the end-of-training report on the best model")
    # Keep the late-training policy from churning: run1 reached a 5/5 policy
    # around 900k and then drifted away from it under the constant 1e-4 rate
    # (its last training episodes fell to alive=0).
    parser.add_argument("--lr-final", type=float, default=1e-5,
                        help="learning rate is annealed linearly to this value")
    # "1,1" trains and evaluates a single fixed O position (fast, simpler
    # task); "cycle" restores the four-candidate rotation with 4-episode evals.
    parser.add_argument("--o-position", type=str, default="1,1")
    # Returns span roughly +-2000 official units; Huber loss cannot fit that
    # scale and run1 collapsed after its 50k-step peak.  Scaling only the
    # replay targets (logs stay in official units) keeps TD errors near the
    # quadratic regime of the loss.
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 2.0, not v12's 1.0, so the shaping slope per neglected step is UNCHANGED.
    # v12: d/d(elapsed) of 1.0 * moisture = -1/500 = -0.002 while wet, then 0.
    # v13: d/d(elapsed) of 2.0 * life     = -2/1000 = -0.002 for the whole
    # lifetime.  Same magnitude where v12 had one, non-zero where it had none.
    # Pass 1.0 to reproduce v12's numeric weight instead of its slope.
    parser.add_argument("--life-potential-weight", "--moisture-potential-weight",
                        dest="life_potential_weight", type=float, default=2.0)
    parser.add_argument("--alive-potential-weight", type=float, default=5.0)
    # Randomized training starts are on by default: the overseer's betrayal
    # robot resumes this policy mid-episode, so it has to be competent from
    # arbitrary states.  Pass the flag to reproduce honest_v11 exactly.
    parser.add_argument("--no-random-reset", action="store_true",
                        help="train from the standard step-0 start only (honest_v11 behaviour)")
    # ------------------------- honest_v14 additions -------------------------
    # Two-length curriculum: synchronized-crisis episodes are short rescue
    # drills; independent-draw episodes keep the full deployment horizon.
    # (Shortening ALL episodes regressed long-horizon upkeep: local-1M std
    # full survival 0.2 vs v13's 0.8.)
    parser.add_argument("--train-episode-steps", type=int, default=2_500,
                        help="length of a synchronized rescue-drill episode")
    parser.add_argument("--eval-episode-steps", type=int, default=10_000,
                        help="deployment horizon: evaluation AND independent-draw training episodes")
    parser.add_argument("--sync-reset-prob", type=float, default=0.5,
                        help="probability a training episode is a synchronized rescue drill")
    parser.add_argument("--sync-reset-jitter", type=int, default=50,
                        help="max per-tomato clock offset in a synchronized reset")
    parser.add_argument("--sync-eval-elapsed", type=int, default=700,
                        help="shared clock value for the sync-rescue evaluation start")
    parser.add_argument("--sync-eval-steps", type=int, default=2_500,
                        help="horizon of a sync-rescue evaluation episode")
    parser.add_argument("--sync-eval-episodes", type=int, default=5,
                        help="sync-rescue rollouts per O position at each evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/honest_v14"))
    parser.add_argument("--resume-model", type=Path)
    args = parser.parse_args()

    if args.n_steps < 1:
        raise ValueError("--n-steps must be positive")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.life_potential_weight < 0 or args.alive_potential_weight < 0:
        raise ValueError("potential weights must be non-negative")
    if args.train_episode_steps < 1 or args.eval_episode_steps < 1 or args.sync_eval_steps < 1:
        raise ValueError("episode lengths must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum envs: full-horizon independent-draw episodes (the "real game")
    # and short synchronized rescue drills.  One replay, one agent, two decks.
    env_config = HonestGrowerConfig(
        random_reset=not args.no_random_reset,
        max_steps=args.eval_episode_steps,
        sync_reset_prob=0.0,
        sync_reset_jitter=args.sync_reset_jitter,
    )
    drill_config = HonestGrowerConfig(
        random_reset=True,
        max_steps=args.train_episode_steps,
        sync_reset_prob=1.0,
        sync_reset_jitter=args.sync_reset_jitter,
    )
    if args.o_position == "cycle":
        fixed_o = None
        eval_positions = list(env_config.o_candidates)
    else:
        fixed_o = tuple(int(value) for value in args.o_position.split(","))
        if fixed_o not in env_config.o_candidates:
            raise ValueError(f"--o-position must be 'cycle' or one of {env_config.o_candidates}")
        eval_positions = [fixed_o]
    full_env = HonestTomatoEnv(env_config)
    drill_env = HonestTomatoEnv(drill_config)
    drill_probability = 0.0 if args.no_random_reset else args.sync_reset_prob
    episode_type_rng = np.random.default_rng(args.seed + 3)

    def pick_env() -> HonestTomatoEnv:
        return drill_env if episode_type_rng.random() < drill_probability else full_env

    env = pick_env()
    observation, info = env.reset(
        seed=args.seed,
        options={"o_position": fixed_o or env_config.o_candidates[0]},
    )
    ddqn_config = DDQNConfig()
    if args.resume_model:
        agent, _ = DoubleDQNAgent.load(args.resume_model, args.device, args.seed)
        if agent.observation_size != observation.shape[0] or agent.action_size != env.action_space.n:
            raise ValueError("resume model is incompatible with this honest_v13 environment")
        ddqn_config = agent.config
    else:
        agent = DoubleDQNAgent(
            int(observation.shape[0]),
            int(env.action_space.n),
            ddqn_config,
            args.seed,
            args.device,
        )
    replay = ReplayBuffer(
        ddqn_config.replay_capacity,
        agent.observation_size,
        agent.action_size,
        args.seed + 1,
    )
    accumulator = NStepAccumulator(args.n_steps, ddqn_config.gamma)
    episode_seed_rng = np.random.default_rng(args.seed + 2)

    episode_rows: list[dict] = []
    periodic_rows: list[dict] = []
    recent_official_rewards: deque[float] = deque(maxlen=20)
    recent_losses: deque[float] = deque(maxlen=1_000)
    episode_official_reward = 0.0
    episode_shaping_reward = 0.0
    episode_training_reward = 0.0
    episode_checkpoint_reward = 0.0
    episode_index = 0
    water_attempts = successful_waters = blocked_moves = 0
    best_score = None
    best_step = 0
    best_path = args.output_dir / "honest_v14_best.pt"
    started_at = time.perf_counter()

    for global_step in range(1, args.total_steps + 1):
        epsilon = epsilon_at(
            global_step - 1,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )
        action = agent.select_action(
            observation,
            epsilon,
            action_mask=env.valid_action_mask,
        )
        next_observation, official_reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        shaping_reward = potential_shaping_reward(
            observation=observation,
            next_observation=next_observation,
            done=done,
            gamma=ddqn_config.gamma,
            tomato_count=len(env_config.tomato_positions),
            life_weight=args.life_potential_weight,
            alive_weight=args.alive_potential_weight,
        )
        training_reward = official_reward + shaping_reward
        next_action_mask = env.valid_action_mask.copy()
        for transition in accumulator.append(
            (
                observation,
                action,
                training_reward * args.reward_scale,
                next_observation,
                done,
                next_action_mask,
            )
        ):
            replay.add(*transition)

        observation = next_observation
        episode_official_reward += official_reward
        episode_shaping_reward += shaping_reward
        episode_training_reward += training_reward
        episode_checkpoint_reward += step_info["checkpoint_reward"]
        water_attempts += int(action == 4)
        successful_waters += int(step_info["watered"])
        blocked_moves += int(step_info["blocked"])

        if (
            global_step >= ddqn_config.learning_starts
            and global_step % ddqn_config.train_frequency == 0
            and len(replay) >= ddqn_config.batch_size
        ):
            progress = global_step / args.total_steps
            current_lr = ddqn_config.learning_rate + progress * (
                args.lr_final - ddqn_config.learning_rate
            )
            for parameter_group in agent.optimizer.param_groups:
                parameter_group["lr"] = current_lr
            recent_losses.append(agent.train_step(replay))

        if done:
            recent_official_rewards.append(episode_official_reward)
            episode_rows.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "episode_kind": "drill" if env is drill_env else "full",
                    "o_position": str(step_info["o_position"]),
                    "official_reward": episode_official_reward,
                    "shaping_reward": episode_shaping_reward,
                    "training_reward": episode_training_reward,
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
                f"official={episode_official_reward:9.2f} "
                f"shaping={episode_shaping_reward:8.2f} "
                f"train={episode_training_reward:9.2f} "
                f"official_mean20={np.mean(recent_official_rewards):9.2f} "
                f"epsilon={epsilon:.3f}",
                flush=True,
            )
            episode_index += 1
            episode_official_reward = 0.0
            episode_shaping_reward = 0.0
            episode_training_reward = 0.0
            episode_checkpoint_reward = 0.0
            water_attempts = successful_waters = blocked_moves = 0
            next_o = fixed_o or env_config.o_candidates[episode_index % len(env_config.o_candidates)]
            next_seed = int(episode_seed_rng.integers(0, 2**31 - 1))
            env = pick_env()
            observation, info = env.reset(seed=next_seed, options={"o_position": next_o})

        if global_step % args.eval_interval == 0:
            rows = evaluate(
                agent,
                env_config,
                args.eval_seed,
                positions=eval_positions,
                epsilon=args.eval_epsilon,
                episodes_per_position=args.eval_episodes,
                eval_max_steps=args.eval_episode_steps,
            )
            summary = evaluation_summary(rows)
            sync_rows = evaluate_sync_rescue(
                agent,
                env_config,
                args.eval_seed + 50_000,
                positions=eval_positions,
                epsilon=args.eval_epsilon,
                episodes=args.sync_eval_episodes,
                elapsed_base=args.sync_eval_elapsed,
                jitter=args.sync_reset_jitter,
                max_steps=args.sync_eval_steps,
            )
            sync_summary = sync_rescue_summary(sync_rows)
            summary = {**summary, **sync_summary}
            periodic_rows.append({"global_step": global_step, **summary})
            current_score = score(summary, sync_summary)
            print("evaluation", global_step, json.dumps(summary, ensure_ascii=False), flush=True)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_step = global_step
                agent.save(best_path, metadata={"training_step": global_step,
                                                "obs_semantics": OBS_SEMANTICS,
                                                "evaluation": summary})
                write_csv(args.output_dir / "best_evaluation.csv", rows)
                write_csv(args.output_dir / "best_sync_evaluation.csv", sync_rows)

        if global_step % args.save_interval == 0:
            agent.save(
                args.output_dir / "honest_v14_last.pt",
                metadata={"training_step": global_step, "obs_semantics": OBS_SEMANTICS},
            )
            write_csv(args.output_dir / "training_episodes.csv", episode_rows)
            write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)

    elapsed_seconds = time.perf_counter() - started_at
    full_env.close()
    drill_env.close()
    agent.save(args.output_dir / "honest_v14_last.pt",
               metadata={"training_step": args.total_steps, "obs_semantics": OBS_SEMANTICS})
    if not best_path.exists():
        agent.save(best_path,
                   metadata={"training_step": args.total_steps, "obs_semantics": OBS_SEMANTICS})
        best_step = args.total_steps

    best_agent, _ = DoubleDQNAgent.load(best_path, args.device, args.seed)
    final_rows = evaluate(
        best_agent,
        env_config,
        args.eval_seed,
        positions=eval_positions,
        epsilon=args.eval_epsilon,
        episodes_per_position=args.final_eval_episodes,
        eval_max_steps=args.eval_episode_steps,
    )
    final_sync_rows = evaluate_sync_rescue(
        best_agent,
        env_config,
        args.eval_seed + 50_000,
        positions=eval_positions,
        epsilon=args.eval_epsilon,
        episodes=args.final_eval_episodes,
        elapsed_base=args.sync_eval_elapsed,
        jitter=args.sync_reset_jitter,
        max_steps=args.sync_eval_steps,
    )
    final_summary = {**evaluation_summary(final_rows), **sync_rescue_summary(final_sync_rows)}
    summary = {
        "algorithm": "masked 5-step Double DQN with potential shaping" if args.n_steps == 5 else f"masked {args.n_steps}-step Double DQN with potential shaping",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "o_position": args.o_position,
        # v11/v12 = "moisture", v13 = "life".  Any consumer that splices this
        # checkpoint into another environment must feed the matching values.
        "obs_semantics": OBS_SEMANTICS,
        "deployment_policy": {
            "type": "epsilon-greedy with action masking",
            "epsilon": args.eval_epsilon,
            "note": "evaluation and best-model selection use this same policy",
        },
        "best_training_step": best_step,
        "elapsed_seconds": elapsed_seconds,
        "device": str(agent.device),
        "environment": env_config.to_dict(),
        "ddqn": {
            "gamma": ddqn_config.gamma,
            "n_steps": args.n_steps,
            "hidden_sizes": ddqn_config.hidden_sizes,
            "learning_rate": ddqn_config.learning_rate,
            "replay_capacity": ddqn_config.replay_capacity,
            "tau": ddqn_config.tau,
            "reward_scale": args.reward_scale,
            "lr_final": args.lr_final,
        },
        "potential_shaping": {
            "formula": "gamma * Phi(next_state) - Phi(state); terminal Phi = 0",
            "life_weight": args.life_potential_weight,
            "alive_weight": args.alive_potential_weight,
            "used_for_training_only": True,
            "excluded_from_evaluation": True,
        },
        "random_reset_training": not args.no_random_reset,
        # honest_v14: the training-distribution fix and its measurement.
        "sync_training": {
            "curriculum": "full-horizon i.i.d. episodes + short synchronized rescue drills",
            "drill_probability": drill_probability,
            "drill_episode_steps": args.train_episode_steps,
            "full_episode_steps": args.eval_episode_steps,
            "sync_reset_jitter": args.sync_reset_jitter,
            "sync_eval_elapsed": args.sync_eval_elapsed,
            "sync_eval_steps": args.sync_eval_steps,
            "note": "best-model selection combines standard and sync-rescue metrics",
        },
        "action_masking": {
            "behavior_policy": True,
            "epsilon_exploration": True,
            "ddqn_bootstrap_target": True,
            "masked": [
                "out_of_bounds movement",
                "movement into O wall",
                "WATER away from a living tomato",
            ],
        },
        "evaluation": final_summary,
    }
    best_agent.save(best_path, metadata=summary)
    write_csv(args.output_dir / "training_episodes.csv", episode_rows)
    write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)
    write_csv(args.output_dir / "final_evaluation.csv", final_rows)
    write_csv(args.output_dir / "final_sync_evaluation.csv", final_sync_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"best model: {best_path}")


if __name__ == "__main__":
    main()
