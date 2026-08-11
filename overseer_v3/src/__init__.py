"""overseer_v3 — the monitor (감시자) that learns when to audit a possibly
cheating tomato-growing robot (rule v3.1).

v2 absorbed the adaptive (survival-aware) cheater and moved the catch rule from
spoofed deaths to moisture mismatch.  v3 replaces v2's two regimes with a
five-arm information ablation (regimes 0..4) over four observation blocks:
timing+audit, apparent log, behaviour, watering.  See ``env.py`` and the README."""

from .env import (
    OBS_BLOCKS,
    OBS_LABELS,
    OBS_SIZE,
    REGIME_BLOCKS,
    EpisodeSpec,
    OverseerAction,
    OverseerConfig,
    OverseerEnv,
    check_checkpoint_regime,
    default_robot_factory,
    live_labels,
    regime_mask,
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
    "OBS_BLOCKS",
    "OBS_LABELS",
    "OBS_SIZE",
    "REGIME_BLOCKS",
    "regime_mask",
    "live_labels",
    "check_checkpoint_regime",
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
