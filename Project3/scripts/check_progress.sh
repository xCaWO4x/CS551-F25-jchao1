#!/usr/bin/env bash
# Helper script to check training progress for the DDQN job (including resumed runs).
# Usage:
#   ./scripts/check_progress.sh
#
# This script assumes it is run from the Project3 directory.

set -euo pipefail

PROJECT_ROOT="/home/jchao1/CS551-F25-jchao1/Project3"
cd "$PROJECT_ROOT"

echo "=== Slurm Queue (jchao1) ==="
squeue -u "$USER" -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
echo

echo "=== DDQN Job Latest Status ==="
latest_ddqn_log=$(ls -t slurm_ddqn_[0-9]*.out 2>/dev/null | grep -v duel | head -1 || true)
if [[ -n "${latest_ddqn_log}" ]]; then
  tail -200 "$latest_ddqn_log" | grep -E "(Training:|Episode|Reward=|Steps=)" | tail -10
else
  echo "DDQN log not available."
fi
echo

echo "=== DDQN Metrics (combined logs) ==="
# Only keep the two most-recent DDQN logs (current run + latest prior run)
ddqn_logs=$(ls -t slurm_ddqn_[0-9]*.out 2>/dev/null | grep -v duel | head -2 | sort || true)
if [[ -n "${ddqn_logs}" ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate myenv
  DDQN_LOG_LIST="${ddqn_logs}" python3 <<'PY'
import os
import re
import math

log_files = os.environ.get("DDQN_LOG_LIST", "").split()
log_files = [f for f in log_files if f.strip()]

if not log_files:
    print("No DDQN logs found.")
    raise SystemExit(0)

log_files.sort(key=lambda f: os.path.getmtime(f))

pattern_detail = re.compile(r'Episode (\d+)/200000 \| Reward: ([\d.\-]+)')
pattern_progress = re.compile(r'Training:.*?\| (\d+)/200000 .*?Reward=([\d.\-]+),')

combined_eps = []
combined_rewards = []
offset = 0

def append_from_dict(data_dict, offset):
    if not data_dict:
        return offset
    ordered = sorted(data_dict.items())
    for ep, reward in ordered:
        combined_eps.append(ep + offset)
        combined_rewards.append(reward)
    return offset + ordered[-1][0] + 1

for path in log_files:
    per_log = {}
    try:
        with open(path, "r") as f:
            for line in f:
                m = pattern_detail.search(line)
                if m:
                    per_log[int(m.group(1))] = float(m.group(2))
                    continue
                m = pattern_progress.search(line)
                if m:
                    per_log.setdefault(int(m.group(1)), float(m.group(2)))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        continue
    offset = append_from_dict(per_log, offset)

if not combined_rewards:
    print("No episode data parsed from DDQN logs.")
    raise SystemExit(0)

def tail_mean(data, n):
    if not data:
        return math.nan
    window = data[-n:] if len(data) >= n else data
    return sum(window) / len(window)

total_episode_index = combined_eps[-1]
print(f"Total Episodes (combined index): {total_episode_index}")
print(f"Last 30 Episodes Mean: {tail_mean(combined_rewards, 30):.4f}")
print(f"Last 100 Episodes Mean: {tail_mean(combined_rewards, 100):.4f}")
if len(combined_rewards) >= 1000:
    print(f"Last 1000 Episodes Mean: {tail_mean(combined_rewards, 1000):.4f}")
print(f"Max Reward: {max(combined_rewards):.2f}")
if len(combined_rewards) >= 100:
    print(f"Recent Max (last 100): {max(combined_rewards[-100:]):.2f}")
print(f"Episodes with Reward > 0: {sum(1 for r in combined_rewards if r > 0)}")
print(f"Episodes with Reward > 5: {sum(1 for r in combined_rewards if r > 5)}")

if len(combined_rewards) >= 1000:
    first_1000 = tail_mean(combined_rewards[:1000], 1000)
    last_1000 = tail_mean(combined_rewards[-1000:], 1000)
    improvement = last_1000 - first_1000
    base = first_1000 if abs(first_1000) > 1e-6 else 0.01
    print(f"First 1000 mean: {first_1000:.4f}")
    print(f"Last 1000 mean: {last_1000:.4f}")
    print(f"Improvement: {improvement:.4f} ({(improvement / base) * 100:.1f}%)")
PY
else
  echo "DDQN logs not available."
fi
echo

