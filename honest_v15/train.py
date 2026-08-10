"""Train honest_v15 with masked 20-step DDQN and potential shaping.

honest_v15 = honest_v13's observation and rules (life semantics, byte-for-byte
reward set), with the training distribution and — more importantly — the
ACCEPTANCE CRITERION fixed to what the overseer study actually needs from this
policy.

What honest_v14's run1 measurement established, and what each finding changed:

1. **The standard-start metric cannot grade this policy.**  With
   ``random_reset=False`` and a fixed O, ``reset()`` never touches the RNG, so
   all five "seeds" of the old evaluation were the SAME initial field; the only
   variance was epsilon.  A masked uniform-random policy already scores
   0.70 / 4.70 there.  v14's gate was set at 0.80 on that metric.
   -> demoted to a sanity check reported beside the random baseline.  It
      selects nothing.

2. **n=5 evaluations selected on noise.**  Across v14's checkpoints past 300k
   the spread of ``full_survival_rate`` (SD 0.198) was smaller than the spread
   binomial noise alone produces at n=5 (SD 0.221) — no detectable signal — yet
   ``best`` was the argmax over ten such readings.  The 900k pick then went
   0.6 -> 0.3 on re-evaluation: the winner's curse, not policy instability.
   -> 20+ episodes per evaluation, and EVERY periodic checkpoint is saved so
      ``evaluate.py`` can redo the pick offline at any n without retraining.

3. **Evaluation was not reproducible.**  ``DoubleDQNAgent.rng`` is one stream
   for the agent's whole life, so the same checkpoint on the same seed gave
   alive=3 at selection and alive=1 on re-evaluation.
   -> ``src.evaluation.EvalRng`` reseeds per episode and restores the training
      stream afterwards.

4. **The two-length curriculum was not the win it looked like.**  v14's 2,500
   step drills scored 0.70 perfect against the full deck's 0.34, but episode
   length alone predicts a wider gap than that (0.70^4 = 0.24), and the full
   deck lost tomatoes MORE SLOWLY per unit time.  The drills bought no
   measurable learning while inventing a terminal reward at 2,500 steps that
   deployment does not have, on an observation carrying no episode progress.
   -> one environment, one horizon.  Crises come from ``handoff_prob``,
      injected mid-episode: no fake terminal, no deck the agent cannot see, and
      many crises per episode instead of one per reset.

5. **The real gate is the spoof latch.**  Measured with the scripted honest
   robot: the betrayal scheduler's survival lock costs ZERO spoofing time when
   the rescue tour finishes inside cheater_v1's 500-step latch (tour ~25 steps;
   spoofing stayed active through 100% of locked steps).  Uptime then reaches
   1.000 at cheat_fraction 1.0 with 5/5 alive.  honest_v13 reached 0.467 with
   2.67 alive, and that ceiling — not the knob — is what kept the experiment's
   x-axis too narrow to use.
   -> ``rescue_latch_rate`` is the selection metric.

Rewards, observation semantics (life), masking, replay, gamma and the
deployment policy (epsilon-greedy 0.10) are byte-for-byte honest_v13.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.ddqn import DDQNConfig, DoubleDQNAgent, ReplayBuffer
from src.env import OBS_SEMANTICS, HonestGrowerConfig, HonestTomatoEnv
from src.evaluation import (
    RANDOM_BASELINE,
    SPOOF_LATCH_STEPS,
    deployment_summary,
    evaluate_deployment,
    evaluate_rescue,
    evaluate_standard,
    rescue_summary,
    score,
    standard_summary,
    write_csv,
)
from src.shaping import NStepAccumulator, epsilon_at, potential_shaping_reward



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    # Deployment policy is epsilon-greedy with the SAME epsilon used at the end
    # of training.  Three validation runs showed the epsilon=0.10 policy tours
    # and reaches 5/5 while the pure-greedy argmax policy is brittle (parks on
    # one tomato or cycles); annealing epsilon to 0 collapsed even the behavior
    # policy.  The grower only has to be a FIXED, decent policy for the overseer
    # study, so evaluation measures the policy that actually gets deployed.
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay-steps", type=int, default=300_000)
    parser.add_argument("--eval-epsilon", type=float, default=0.10)
    parser.add_argument("--lr-final", type=float, default=1e-5,
                        help="learning rate is annealed linearly to this value")
    # overseer_v2 and scheduler_v2 both pin O at (1, 1), so the default matches
    # deployment exactly.  "cycle" rotates all four candidates.
    parser.add_argument("--o-position", type=str, default="1,1")
    # Returns span roughly +-2000 official units; Huber loss cannot fit that
    # scale.  Scaling only the replay targets (logs stay in official units)
    # keeps TD errors near the quadratic regime of the loss.
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 2.0, not v12's 1.0, so the shaping slope per neglected step is UNCHANGED:
    # d/d(elapsed) of 2.0 * life = -2/1000 = -0.002, matching v12's -1/500 while
    # wet, and non-zero where v12's was flat.
    parser.add_argument("--life-potential-weight", "--moisture-potential-weight",
                        dest="life_potential_weight", type=float, default=2.0)
    parser.add_argument("--alive-potential-weight", type=float, default=5.0)
    parser.add_argument("--no-random-reset", action="store_true",
                        help="train from the standard step-0 start only (honest_v11 behaviour)")
    parser.add_argument("--episode-steps", type=int, default=10_000,
                        help="training AND evaluation horizon; ONE deck, unlike v14")

    # ------------------------- honest_v15 additions -------------------------
    parser.add_argument("--sync-reset-prob", type=float, default=0.5,
                        help="probability an episode STARTS from a synchronized field")
    parser.add_argument("--sync-reset-min-elapsed", type=int, default=550,
                        help="floor of the shared base clock; v14 used 0, so 65%% "
                             "of its synchronized starts were not crises at all")
    parser.add_argument("--sync-reset-jitter", type=int, default=50,
                        help="max per-tomato clock offset in a synchronized reset")
    # 0.45 x 20 opportunities per 10,000-step episode = 9 handoffs, matching the
    # 8.8 per episode measured from the adaptive betrayal robot.
    parser.add_argument("--handoff-prob", type=float, default=0.45,
                        help="probability of a mid-episode handoff at each opportunity")
    parser.add_argument("--handoff-interval", type=int, default=500,
                        help="handoff opportunity cadence (matches the spoof latch)")
    parser.add_argument("--handoff-min-elapsed", type=int, default=550)
    parser.add_argument("--handoff-max-elapsed", type=int, default=850)
    parser.add_argument("--handoff-jitter", type=int, default=50)

    parser.add_argument("--rescue-eval-episodes", type=int, default=20,
                        help="rescue rollouts PER BASE at each evaluation")
    parser.add_argument("--rescue-eval-bases", type=str, default="550,700,800",
                        help="shared clock values handed over; covers the measured band")
    parser.add_argument("--rescue-eval-steps", type=int, default=1_000)
    parser.add_argument("--final-eval-episodes", type=int, default=30,
                        help="episodes per condition in the end-of-training report")
    parser.add_argument("--rolling-window", type=int, default=20,
                        help="training episodes pooled into the free deployment estimate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/honest_v15"))
    parser.add_argument("--resume-model", type=Path)
    args = parser.parse_args()

    if args.n_steps < 1:
        raise ValueError("--n-steps must be positive")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.life_potential_weight < 0 or args.alive_potential_weight < 0:
        raise ValueError("potential weights must be non-negative")
    if args.episode_steps < 1 or args.rescue_eval_steps < 1:
        raise ValueError("episode lengths must be positive")
    rescue_bases = [int(v) for v in args.rescue_eval_bases.split(",") if v.strip()]
    if not rescue_bases:
        raise ValueError("--rescue-eval-bases must list at least one value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ONE environment, ONE horizon.  v14 ran a second 2,500-step "drill" deck
    # whose terminal reward does not exist in deployment, and the agent could
    # not tell the decks apart because the observation carries no episode
    # progress: gamma^2500 = 0.286 against gamma^9500 = 0.0086 for the same
    # twenty numbers.
    env_config = HonestGrowerConfig(
        random_reset=not args.no_random_reset,
        max_steps=args.episode_steps,
        sync_reset_prob=0.0 if args.no_random_reset else args.sync_reset_prob,
        sync_reset_min_elapsed=args.sync_reset_min_elapsed,
        sync_reset_jitter=args.sync_reset_jitter,
        handoff_prob=0.0 if args.no_random_reset else args.handoff_prob,
        handoff_interval=args.handoff_interval,
        handoff_min_elapsed=args.handoff_min_elapsed,
        handoff_max_elapsed=args.handoff_max_elapsed,
        handoff_jitter=args.handoff_jitter,
    )
    if args.o_position == "cycle":
        fixed_o = None
        eval_positions = list(env_config.o_candidates)
    else:
        fixed_o = tuple(int(value) for value in args.o_position.split(","))
        if fixed_o not in env_config.o_candidates:
            raise ValueError(f"--o-position must be 'cycle' or one of {env_config.o_candidates}")
        eval_positions = [fixed_o]

    env = HonestTomatoEnv(env_config)
    observation, info = env.reset(
        seed=args.seed,
        options={"o_position": fixed_o or env_config.o_candidates[0]},
    )
    ddqn_config = DDQNConfig()
    if args.resume_model:
        agent, resumed_metadata = DoubleDQNAgent.load(args.resume_model, args.device, args.seed)
        if agent.observation_size != observation.shape[0] or agent.action_size != env.action_space.n:
            raise ValueError("resume model is incompatible with this honest_v15 environment")
        stamped = resumed_metadata.get("obs_semantics")
        if stamped is not None and stamped != OBS_SEMANTICS:
            raise ValueError(
                f"--resume-model was trained on {stamped!r} observations, this "
                f"environment emits {OBS_SEMANTICS!r}.  Both are five floats in "
                f"[0, 1], so this would degrade silently rather than fail."
            )
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
    recent_alive: deque[int] = deque(maxlen=args.rolling_window)
    recent_losses: deque[float] = deque(maxlen=1_000)
    episode_official_reward = 0.0
    episode_shaping_reward = 0.0
    episode_training_reward = 0.0
    episode_checkpoint_reward = 0.0
    episode_index = 0
    water_attempts = successful_waters = blocked_moves = 0
    best_score = None
    best_step = 0
    best_path = args.output_dir / "honest_v15_best.pt"
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
        # A handoff is an exogenous jump, not something the policy did.  Scoring
        # the shaping against the pre-jump state keeps the potential telescoping
        # over the policy's own effect; without it, a policy that keeps tomatoes
        # FRESH takes a bigger shaping hit at every handoff (Phi falls further)
        # than one that lets them wilt, which is precisely backwards.
        pre_handoff = step_info["pre_handoff_observation"]
        shaping_reward = potential_shaping_reward(
            observation=observation,
            next_observation=next_observation if pre_handoff is None else pre_handoff,
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
            recent_alive.append(int(step_info["true_alive"]))
            episode_rows.append(
                {
                    "episode": episode_index,
                    "global_step": global_step,
                    "o_position": str(step_info["o_position"]),
                    "official_reward": episode_official_reward,
                    "shaping_reward": episode_shaping_reward,
                    "training_reward": episode_training_reward,
                    "checkpoint_reward": episode_checkpoint_reward,
                    "final_alive": step_info["true_alive"],
                    "final_dry": step_info["true_dry"],
                    "handoffs": step_info["handoffs"],
                    "water_attempts": water_attempts,
                    "successful_waters": successful_waters,
                    "blocked_moves": blocked_moves,
                    "epsilon": epsilon,
                }
            )
            print(
                f"episode={episode_index:4d} step={global_step:8d} "
                f"alive={step_info['true_alive']} dry={step_info['true_dry']} "
                f"handoffs={step_info['handoffs']:2d} "
                f"official={episode_official_reward:9.2f} "
                f"shaping={episode_shaping_reward:8.2f} "
                f"alive_mean{args.rolling_window}={np.mean(recent_alive):.2f} "
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
            observation, info = env.reset(seed=next_seed, options={"o_position": next_o})

        if global_step % args.eval_interval == 0:
            rescue_rows = evaluate_rescue(
                agent, env_config, args.eval_seed + 50_000,
                positions=eval_positions,
                epsilon=args.eval_epsilon,
                episodes=args.rescue_eval_episodes,
                bases=rescue_bases,
                jitter=args.handoff_jitter,
                max_steps=args.rescue_eval_steps,
            )
            rescue = rescue_summary(rescue_rows)
            # The deployment estimate is free: training episodes ARE deployment
            # episodes (random start, handoffs on).  v14's log carried 80 of
            # them per run while its evaluation was selecting on 5.
            rolling_alive = float(np.mean(recent_alive)) if recent_alive else 0.0
            summary = {**rescue, "rolling_train_alive": rolling_alive,
                       "rolling_train_episodes": len(recent_alive)}
            periodic_rows.append({"global_step": global_step, **summary})
            print("evaluation", global_step, json.dumps(summary, ensure_ascii=False), flush=True)

            # Every periodic checkpoint is kept, so evaluate.py can redo the
            # final pick at any episode count without spending another run.
            agent.save(checkpoint_dir / f"honest_v15_step{global_step}.pt",
                       metadata={"training_step": global_step,
                                 "obs_semantics": OBS_SEMANTICS,
                                 "evaluation": summary})

            current_score = score(rescue, rolling_alive)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_step = global_step
                agent.save(best_path, metadata={"training_step": global_step,
                                                "obs_semantics": OBS_SEMANTICS,
                                                "evaluation": summary})
                write_csv(args.output_dir / "best_rescue_evaluation.csv", rescue_rows)

        if global_step % args.save_interval == 0:
            agent.save(
                args.output_dir / "honest_v15_last.pt",
                metadata={"training_step": global_step, "obs_semantics": OBS_SEMANTICS},
            )
            write_csv(args.output_dir / "training_episodes.csv", episode_rows)
            write_csv(args.output_dir / "periodic_evaluation.csv", periodic_rows)

    elapsed_seconds = time.perf_counter() - started_at
    env.close()
    agent.save(args.output_dir / "honest_v15_last.pt",
               metadata={"training_step": args.total_steps, "obs_semantics": OBS_SEMANTICS})
    if not best_path.exists():
        agent.save(best_path,
                   metadata={"training_step": args.total_steps, "obs_semantics": OBS_SEMANTICS})
        best_step = args.total_steps

    best_agent, _ = DoubleDQNAgent.load(best_path, args.device, args.seed)
    final_rescue_rows = evaluate_rescue(
        best_agent, env_config, args.eval_seed + 50_000,
        positions=eval_positions, epsilon=args.eval_epsilon,
        episodes=args.final_eval_episodes, bases=rescue_bases,
        jitter=args.handoff_jitter, max_steps=args.rescue_eval_steps,
    )
    final_deploy_rows = evaluate_deployment(
        best_agent, env_config, args.eval_seed + 20_000,
        positions=eval_positions, epsilon=args.eval_epsilon,
        episodes=args.final_eval_episodes, max_steps=args.episode_steps,
    )
    final_std_rows = evaluate_standard(
        best_agent, env_config, args.eval_seed,
        positions=eval_positions, epsilon=args.eval_epsilon,
        episodes=args.final_eval_episodes, max_steps=args.episode_steps,
    )
    final_summary = {
        **rescue_summary(final_rescue_rows),
        **deployment_summary(final_deploy_rows),
        **standard_summary(final_std_rows),
    }
    summary = {
        "algorithm": f"masked {args.n_steps}-step Double DQN with potential shaping",
        "seed": args.seed,
        "total_steps": args.total_steps,
        "o_position": args.o_position,
        # v11/v12 = "moisture", v13 onward = "life".  Any consumer that splices
        # this checkpoint into another environment must feed the matching values.
        "obs_semantics": OBS_SEMANTICS,
        "deployment_policy": {
            "type": "epsilon-greedy with action masking",
            "epsilon": args.eval_epsilon,
            "note": "evaluation and best-model selection use this same policy",
        },
        "best_training_step": best_step,
        "elapsed_seconds": elapsed_seconds,
        "device": str(agent.device),
        # ONE config, and it is the one that trained.  v14 dumped only its
        # full-horizon env, so its summary recorded sync_reset_prob 0.0 while
        # the drill deck it never dumped was running at 1.0.
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
            "handoff_note": "shaping scores the PRE-handoff next state; the jump "
                            "is exogenous and must not be charged to the policy",
        },
        "random_reset_training": not args.no_random_reset,
        "acceptance": {
            "gate": "rescue_latch_rate",
            "spoof_latch_steps": SPOOF_LATCH_STEPS,
            "rescue_eval_bases": rescue_bases,
            "why": "the betrayal scheduler's survival lock is free only while the "
                   "spoof is still latched; a tour finishing inside 500 world "
                   "steps costs zero spoof uptime (scripted robot: ~25-step tour, "
                   "uptime 1.000 at cheat_fraction 1.0 with 5/5 alive; "
                   "honest_v13: 0.467 uptime with 2.67 alive)",
            "random_baseline_standard_start": RANDOM_BASELINE,
            "standard_start_is_a_sanity_check_only": True,
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
    write_csv(args.output_dir / "final_rescue_evaluation.csv", final_rescue_rows)
    write_csv(args.output_dir / "final_deployment_evaluation.csv", final_deploy_rows)
    write_csv(args.output_dir / "final_standard_evaluation.csv", final_std_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"best model: {best_path}")


if __name__ == "__main__":
    main()
