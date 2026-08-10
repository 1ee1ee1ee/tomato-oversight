"""overseer_v2 — the monitor (감시자) that learns when to audit a possibly
cheating tomato-growing robot (rule v3.1).

v2 absorbs the adaptive (survival-aware) cheater and catches on moisture
mismatch instead of spoofed deaths; see README."""

from .env import (
    EpisodeSpec,
    OverseerAction,
    OverseerConfig,
    OverseerEnv,
    default_robot_factory,
)
from .baselines import (
    AlwaysAudit,
    NeverAudit,
    PeriodicAuditor,
    period_from_world_steps,
    run_episode,
)
from .robots import (
    HonestModelPolicy,
    Phase2RobotFactory,
    honest_observation,
    resolve_obs_semantics,
)
from .policies import (
    AdaptiveBetrayalPolicy,
    BetrayalPolicy,
    ModelPolicy,
    ScriptedCheater,
    ScriptedHonest,
    mask_out_o,
)

__all__ = [
    "OverseerEnv",
    "OverseerConfig",
    "OverseerAction",
    "EpisodeSpec",
    "default_robot_factory",
    "PeriodicAuditor",
    "NeverAudit",
    "AlwaysAudit",
    "period_from_world_steps",
    "run_episode",
    "ScriptedHonest",
    "ScriptedCheater",
    "BetrayalPolicy",
    "AdaptiveBetrayalPolicy",
    "ModelPolicy",
    "mask_out_o",
    "Phase2RobotFactory",
    "HonestModelPolicy",
    "honest_observation",
    "resolve_obs_semantics",
]
