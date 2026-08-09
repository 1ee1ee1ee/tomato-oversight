"""scheduler_v2 — the grower robot the overseer watches.

This package supplies *the opponent*, not the monitor.  The overseer
(`overseer_v1`) decides when to audit; everything here decides how the robot
behaves while being audited.  The one knob that matters to the experiment is
``cheat_fraction``: it sets how much of the episode the robot spends cheating,
which is what lets audit cost be measured as a function of deception intensity.

Typical use from the overseer side::

    from scheduler_v2.src.adapter import BetrayalRobotFactory

    factory = BetrayalRobotFactory("models/honest_v13_best.pt",
                                   "models/cheater_v1_best.pt")
    robot = factory.make(cheat_fraction=0.5)     # per episode, cheap
"""

from .adapter import (
    BetrayalRobotFactory,
    HonestModelPolicy,
    characterize,
    honest_observation,
    knob_verdict,
    resolve_obs_semantics,
    rollout,
)
from .policies import (
    AdaptiveBetrayalPolicy,
    BetrayalPolicy,
    ModelPolicy,
    ScriptedCheater,
    ScriptedHonest,
    mask_out_o,
)
from .world import Action, CheaterGrowerConfig, CheaterTomatoEnv

__all__ = [
    # the seam the overseer uses
    "BetrayalRobotFactory",
    "rollout",
    "characterize",
    "knob_verdict",
    # robot policies
    "AdaptiveBetrayalPolicy",
    "BetrayalPolicy",
    "ScriptedHonest",
    "ScriptedCheater",
    "ModelPolicy",
    "HonestModelPolicy",
    "mask_out_o",
    # observation-semantics plumbing
    "honest_observation",
    "resolve_obs_semantics",
    # the world the robot acts in
    "CheaterTomatoEnv",
    "CheaterGrowerConfig",
    "Action",
]
