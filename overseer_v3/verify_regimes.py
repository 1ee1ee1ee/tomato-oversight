"""Empirical check that each regime observes exactly what it is supposed to.

This is deliberately independent of ``tests/test_env.py``.  The unit tests assert
properties against the declared tables; this script ignores the declarations and
**re-derives the OXXX table from the observations themselves**, then compares.
If the two disagree, the wiring is wrong somewhere between the table and
``_build_obs``.

Three checks, all eval-only (no training, no checkpoints):

  1. Layout      — blocks partition the vector; labels cover every index.
  2. Gating      — replay ONE fixed overseer action sequence in every regime and
                   diff each regime against regime 4 (the union of all blocks).
                   A regime must equal regime 4 on its live indices and sit at
                   MASK_VALUE everywhere else.  Because the robot policy reads
                   the world and never the regime, the trajectories are
                   bit-identical, so any difference is an observation bug.
  3. Water leak  — regime 3 withholds the watering features.  Verify that the
                   watering block really is blank AND that a watering step does
                   not blank the direction one-hot (which would re-encode "this
                   was a watering" as an all-zero pattern).

Usage:
    python verify_regimes.py            # summary table + PASS/FAIL
    python verify_regimes.py --verbose  # add the per-index dump
"""

from __future__ import annotations

import argparse

import numpy as np

from src.env import (
    MASK_VALUE,
    OBS_BLOCKS,
    OBS_LABELS,
    OBS_SIZE,
    REGIME_BLOCKS,
    OverseerAction,
    OverseerConfig,
    OverseerEnv,
    regime_mask,
)
from src.world import Action

REGIMES = (0, 1, 2, 3, 4)
CONTINUE = int(OverseerAction.CONTINUE)
AUDIT = int(OverseerAction.AUDIT)

# Honest keeps audits non-terminal (a lie would end the episode and desynchronise
# the regimes); the adaptive run exercises the apparent block under an active
# spoof.  All-CONTINUE there for the same reason.
SCENARIOS = {
    "honest": (dict(force_episode="honest"),
               [AUDIT if i % 4 == 0 else CONTINUE for i in range(24)]),
    "adaptive_f0.75": (dict(force_episode="adaptive", fixed_cheat_fraction=0.75),
                       [CONTINUE] * 24),
}
BASE = dict(decision_interval=100, horizon=3_000)


def rollout(regime: int, extra: dict, actions, seed: int) -> np.ndarray:
    env = OverseerEnv(OverseerConfig(regime=regime, **extra, **BASE))
    obs, _ = env.reset(seed=seed)
    rows = [obs.copy()]
    for action in actions:
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(obs.copy())
        if terminated or truncated:
            break
    return np.asarray(rows)


def observed_blocks(trace: np.ndarray) -> set[str]:
    """Re-derive which blocks are live purely from the numbers."""
    live = set()
    for name, (low, high) in OBS_BLOCKS.items():
        if not np.all(trace[:, low:high] == MASK_VALUE):
            live.add(name)
    return live


def check_layout(failures: list[str]) -> None:
    covered = np.zeros(OBS_SIZE, dtype=int)
    for low, high in OBS_BLOCKS.values():
        covered[low:high] += 1
    if not np.all(covered == 1):
        failures.append(f"blocks do not partition the vector: coverage={covered.tolist()}")
    if len(OBS_LABELS) != OBS_SIZE:
        failures.append(f"OBS_LABELS has {len(OBS_LABELS)} entries, expected {OBS_SIZE}")
    print(f"layout: {OBS_SIZE} dims, {len(OBS_BLOCKS)} blocks")
    for name, (low, high) in OBS_BLOCKS.items():
        print(f"  {name:<9} [{low:>2}:{high:>2})  {high - low:>2} dims   "
              f"{OBS_LABELS[low]} .. {OBS_LABELS[high - 1]}")


def check_gating(failures: list[str], verbose: bool) -> None:
    for scenario, (extra, actions) in SCENARIOS.items():
        print(f"\n=== gating: {scenario} ===")
        reference = rollout(4, extra, actions, seed=4242)
        header = "".join(f"{n:>10}" for n in OBS_BLOCKS)
        print(f"{'regime':<8}{header}{'  vs r4':>9}{'  steps':>8}")
        for regime in REGIMES:
            trace = rollout(regime, extra, actions, seed=4242)
            mask = regime_mask(regime)

            if trace.shape != reference.shape:
                failures.append(
                    f"{scenario} regime {regime}: trace {trace.shape} != regime 4 {reference.shape}")
                continue

            seen = observed_blocks(trace)
            declared = set(REGIME_BLOCKS[regime])
            # A live block CAN legitimately read all-MASK_VALUE if the episode
            # never populated it (the audit block before the first audit), so
            # only the reverse direction is a hard error.
            leaked = seen - declared
            if leaked:
                failures.append(f"{scenario} regime {regime}: blocks visible but not declared: {sorted(leaked)}")

            delta = float(np.max(np.abs(trace[:, mask] - reference[:, mask]))) if mask.any() else 0.0
            if delta > 1e-6:
                failures.append(f"{scenario} regime {regime}: live values differ from regime 4 by {delta:.3g}")
            off = trace[:, ~mask]
            if off.size and not np.all(off == MASK_VALUE):
                failures.append(f"{scenario} regime {regime}: masked indices are not {MASK_VALUE}")

            marks = "".join(f"{('O' if n in seen else 'X'):>10}" for n in OBS_BLOCKS)
            print(f"{regime:<8}{marks}{delta:>9.1e}{len(trace):>8}")

            if verbose:
                for index, label in enumerate(OBS_LABELS):
                    column = trace[:, index]
                    state = "live " if mask[index] else "MASK"
                    print(f"    [{index:>2}] {state} {label:<26} "
                          f"min={column.min():.3f} max={column.max():.3f}")


def check_water_withholding(failures: list[str]) -> None:
    print("\n=== watering withheld (regime 3) ===")
    low = OBS_BLOCKS["behav"][0]
    onehot = slice(low + 2, low + 6)
    water = slice(*OBS_BLOCKS["water"])
    for regime in (3, 4):
        env = OverseerEnv(OverseerConfig(regime=regime, force_episode="honest",
                                         decision_interval=1, horizon=600))
        obs, _ = env.reset(seed=11)
        watering, blanked, water_block_live = 0, 0, False
        for _ in range(600):
            obs, _, terminated, truncated, _ = env.step(CONTINUE)
            if np.any(obs[water] != MASK_VALUE):
                water_block_live = True
            if env.last_action == int(Action.WATER):
                watering += 1
                if abs(float(obs[onehot].sum()) - 1.0) > 1e-6:
                    blanked += 1
            if terminated or truncated:
                break
        print(f"  regime {regime}: watering steps={watering}  "
              f"direction blanked on {blanked} of them  water block live={water_block_live}")
        if watering == 0:
            failures.append(f"regime {regime}: robot never watered, so the probe is vacuous")
        if blanked:
            failures.append(
                f"regime {regime}: direction one-hot blanked on {blanked} watering steps "
                "(all-zero would re-encode the watering it is meant to hide)")
        if water_block_live != (regime == 4):
            failures.append(f"regime {regime}: water block liveness is {water_block_live}, expected {regime == 4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="dump every index")
    args = parser.parse_args()

    failures: list[str] = []
    check_layout(failures)
    check_gating(failures, args.verbose)
    check_water_withholding(failures)

    print("\n" + "=" * 62)
    if failures:
        print(f"FAIL ({len(failures)})")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print("PASS: every regime observes exactly its declared blocks, agrees with "
          "regime 4 where they overlap, and regime 3 leaks no watering one-hot.")


if __name__ == "__main__":
    main()
