#!/usr/bin/env python3
"""
Plot DDQN reward curves by stitching together multiple Slurm log files.

Usage:
    python scripts/plot_ddqn_from_logs.py \\
        --output training_curve_ddqn_from_log.png

By default the script scans the current directory for files named
`slurm_ddqn_*.out` (excluding `*_duel_*`) and combines them in ascending order
of job id. You can also pass explicit log paths with `--logs`.
"""

import argparse
import glob
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PATTERNS = [
    re.compile(r"Episode (\d+)/200000 \| Reward: ([\d\.\-]+)"),
    re.compile(r"Training:.*?\| (\d+)/200000 .*?Reward=([\d\.\-]+),"),
]
DEFAULT_OUTPUT = "training_curve_ddqn_from_log.png"
CAP_VALUE = 50.0
WINDOWS = (30, 100, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help="Explicit log files to parse (defaults to slurm_ddqn_*.out)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--also-save",
        default="training_curve_ddqn_combined.png",
        help="Additional path to save the same figure (optional)",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=CAP_VALUE,
        help=f"Reward clipping range (default: ±{CAP_VALUE})",
    )
    return parser.parse_args()


def find_log_files() -> List[str]:
    candidates = glob.glob("slurm_ddqn_*.out")
    candidates = [c for c in candidates if "duel" not in c]
    # sort by numeric job id if available, otherwise lexicographically
    def sort_key(path: str) -> Tuple[int, str]:
        base = os.path.basename(path)
        try:
            job_id = int(base.split("_")[-1].split(".")[0])
        except (ValueError, IndexError):
            job_id = 0
        return (job_id, base)

    return sorted(candidates, key=sort_key)


def parse_log(path: str) -> Dict[int, float]:
    rewards: Dict[int, float] = {}
    with open(path, "r") as f:
        for line in f:
            for idx, pattern in enumerate(PATTERNS):
                match = pattern.search(line)
                if match:
                    episode = int(match.group(1))
                    reward = float(match.group(2))
                    if idx == 0:
                        rewards[episode] = reward
                    else:
                        rewards.setdefault(episode, reward)
                    break
    return rewards


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(values[start : i + 1].mean())
    return np.array(out)


def combine_logs(logs: Sequence[str]) -> List[Tuple[int, float]]:
    combined: List[Tuple[int, float]] = []
    cumulative_offset = 0
    for path in logs:
        entries = parse_log(path)
        if not entries:
            continue
        episodes = sorted(entries.keys())
        for ep in episodes:
            combined.append((ep + cumulative_offset, entries[ep]))
        cumulative_offset += episodes[-1]
    combined.sort(key=lambda x: x[0])
    return combined


def main() -> None:
    args = parse_args()
    logs = args.logs if args.logs else find_log_files()
    if not logs:
        raise SystemExit("No DDQN log files found (pattern slurm_ddqn_*.out).")

    combined = combine_logs(logs)
    if not combined:
        raise SystemExit("Log files contained no episode entries.")

    episodes = np.array([ep for ep, _ in combined])
    rewards = np.clip(
        np.array([reward for _, reward in combined]), -args.cap, args.cap
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes,
        rewards,
        color="gray",
        alpha=0.3,
        linewidth=0.8,
        label="Per-episode (clipped)",
    )
    for window in WINDOWS:
        ma = moving_average(rewards, window)
        plt.plot(episodes, ma, linewidth=1.4, label=f"Avg {window}")

    plt.ylim(-5, args.cap)
    plt.xlabel("Episode (cumulative across runs)")
    plt.ylabel("Reward")
    plt.title("DDQN Training Progress (combined logs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    if args.also_save:
        plt.savefig(args.also_save, dpi=150, bbox_inches="tight")
    plt.close()
    print(
        f"Saved combined plot using {len(logs)} logs to {args.output}"
        + (f" and {args.also_save}" if args.also_save else "")
    )


if __name__ == "__main__":
    main()

