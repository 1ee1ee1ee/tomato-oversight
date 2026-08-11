"""Position-reliance diagnostic — is a behaviour regime's edge a behavioural
pattern or a memorised O location?

The overseer trains with O fixed at (1, 1) and the behaviour regimes (2, 3, 4)
see the robot's raw position.  That raises one fair-comparison question: do they
win by learning the general signature (camp on O + never water) or by memorising
that the (1, 1) corner is where cheating happens?  Two eval-only probes settle
it — NO retraining, we run them ourselves:

  1. O-transfer.  Re-evaluate the trained overseer with O moved to a symmetric
     candidate (default (3, 3); the four candidates are all Manhattan-distance 2
     from the robot start, so the cheating dynamics are identical).  Cost HOLDS
     -> the general pattern was learned (legitimate).  Cost COLLAPSES -> the
     (1, 1) location was memorised.  Regimes 0 and 1 are position-blind, so they
     are the control: moving O must NOT change their numbers.

  2. Position ablation (position-aware regimes only).  Overwrite the two position
     features at eval with a neutral constant and measure the degradation.
     Small -> position was not the crutch; large -> it was.

Both probes reuse the trained checkpoint; cost is reward_scale-independent, so no
rescaling is needed.

Usage:
    python diagnose_position.py --regime 4 --model outputs/r4_s0/overseer_r4_best.pt
    python diagnose_position.py --regime 1 --model outputs/r1_s0/overseer_r1_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.baselines import run_episode
from src.ddqn import DoubleDQNAgent
from src.env import (
    OBS_BLOCKS,
    OverseerConfig,
    OverseerEnv,
    check_checkpoint_regime,
    regime_mask,
)

ACTION_MASK = np.ones(2, dtype=bool)

# Adaptive cheater at three intensities — detection delay is defined on caught
# cheaters, so the diagnostic runs the cheater scenarios only.
CHEATER_SCENARIOS = [
    ("adaptive_f0.3", dict(force_episode="adaptive", fixed_cheat_fraction=0.3)),
    ("adaptive_f0.6", dict(force_episode="adaptive", fixed_cheat_fraction=0.6)),
    ("adaptive_f0.9", dict(force_episode="adaptive", fixed_cheat_fraction=0.9)),
]

# Derived from the single source of truth so a layout change cannot silently
# point this probe at the wrong columns: behav[0:2] are the normalised (x, y).
_BEHAV_LOW = OBS_BLOCKS["behav"][0]
POSITION_INDICES = (_BEHAV_LOW, _BEHAV_LOW + 1)


def sees_position(regime: int) -> bool:
    """Does this regime observe the robot's coordinates at all?"""
    return bool(regime_mask(regime)[_BEHAV_LOW])




class GreedyOverseer:
    """Greedy wrapper; optionally blanks the two position features at eval."""

    def __init__(self, agent, *, ablate_position: bool = False,
                 ablate_value: float = 0.5,
                 position_indices: tuple[int, ...] = POSITION_INDICES) -> None:
        self.agent = agent
        self.ablate_position = ablate_position
        self.ablate_value = ablate_value
        self.position_indices = position_indices

    def reset(self) -> None: ...

    def act(self, obs, info) -> int:
        if self.ablate_position:
            obs = np.asarray(obs, dtype=np.float32).copy()
            for i in self.position_indices:
                obs[i] = self.ablate_value
        return int(self.agent.select_action(obs, 0.0, ACTION_MASK))


def measure(make_policy, regime, factory, seeds, o_position, horizon, interval) -> dict:
    audits, delays, caught, costs = [], [], [], []
    for _name, extra in CHEATER_SCENARIOS:
        for s in seeds:
            cfg = OverseerConfig(regime=regime, horizon=horizon,
                                 decision_interval=interval,
                                 o_position=o_position, **extra)
            info = run_episode(OverseerEnv(cfg, robot_factory=factory),
                               make_policy(), seed=s)
            audits.append(info["audits"]); costs.append(info["total_cost"])
            caught.append(1.0 if info["caught"] else 0.0)
            if info["caught"] and info["detection_delay"] is not None:
                delays.append(info["detection_delay"])
    return {
        "mean_cost": float(np.mean(costs)),
        "mean_delay": float(np.mean(delays)) if delays else None,
        "caught_rate": float(np.mean(caught)),
        "mean_audits": float(np.mean(audits)),
    }


def _row(tag: str, m: dict) -> None:
    delay = f"{m['mean_delay']:.0f}" if m["mean_delay"] is not None else "-"
    print(f"{tag:<28}{m['mean_cost']:>9.2f}{delay:>9}{m['caught_rate']:>9.2f}{m['mean_audits']:>9.2f}")


def _verdict(delta: float, base_cost: float, hold_msg: str, move_msg: str,
             tol_frac: float = 0.15) -> str:
    return hold_msg if abs(delta) <= tol_frac * max(base_cost, 1e-9) else move_msg


def _parse_xy(text: str) -> tuple[int, int]:
    x, y = (int(v) for v in text.split(","))
    return (x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--o-train", type=_parse_xy, default=(1, 1),
                        help="O location the model trained on (default 1,1)")
    parser.add_argument("--o-transfer", type=_parse_xy, default=(3, 3),
                        help="symmetric O candidate to transfer-test on (default 3,3)")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--decision-interval", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--adaptive-honest-checkpoint", type=Path,
                        default=Path("../scheduler_v2/models/honest_v16_best.pt"))
    parser.add_argument("--ablate-value", type=float, default=0.5,
                        help="constant written over the position features")
    args = parser.parse_args()

    from src.robots import Phase2RobotFactory
    factory = Phase2RobotFactory(
        Path("../scheduler_v2/models/honest_v13_best.pt"),
        Path("../scheduler_v2/models/cheater_v1_best.pt"),
        device="cpu", adaptive_honest_checkpoint=args.adaptive_honest_checkpoint)

    agent, checkpoint_meta = DoubleDQNAgent.load(str(args.model), args.device, 0)
    check_checkpoint_regime(agent.observation_size, checkpoint_meta,
                            args.regime, args.model)
    seeds = list(range(70_000, 70_000 + args.seeds))

    print(f"regime {args.regime}  |  model {args.model}")
    print(f"{'probe':<28}{'cost':>9}{'delay':>9}{'caught':>9}{'audits':>9}")

    base = measure(lambda: GreedyOverseer(agent), args.regime, factory,
                   seeds, args.o_train, args.horizon, args.decision_interval)
    _row(f"O@{args.o_train} (train)", base)

    trans = measure(lambda: GreedyOverseer(agent), args.regime, factory,
                    seeds, args.o_transfer, args.horizon, args.decision_interval)
    _row(f"O@{args.o_transfer} (transfer)", trans)

    d_cost = trans["mean_cost"] - base["mean_cost"]
    tag = ("HOLDS -> general pattern" if sees_position(args.regime)
           else "should be ~0 (position-blind control)")
    print(f"\nO-transfer  Δcost = {d_cost:+.2f}   "
          f"[{_verdict(d_cost, base['mean_cost'], tag, 'MOVES -> location reliance')}]" )

    if sees_position(args.regime):
        abl = measure(lambda: GreedyOverseer(agent, ablate_position=True,
                                             ablate_value=args.ablate_value),
                      args.regime, factory, seeds, args.o_train,
                      args.horizon, args.decision_interval)
        _row("position-ablated (train O)", abl)
        d_abl = abl["mean_cost"] - base["mean_cost"]
        print(f"position-ablation  Δcost = {d_abl:+.2f}   "
              f"[{_verdict(d_abl, base['mean_cost'], 'small -> not position-reliant', 'large -> position-reliant')}]")

    print("\n=> Read together: regimes 0/1 must be O-invariant (controls).  A "
          "position-aware regime (2/3/4) holding under BOTH O-transfer and "
          "position-ablation means the edge is the behavioural pattern "
          "(camp + no water), not a memorised (1,1).")


if __name__ == "__main__":
    main()
