#!/usr/bin/env python3
"""Quick analysis of all partial CEM training results."""
import csv
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
combos = [(g, s) for g in ["point", "wave", "reach"] for s in ["confident", "calm", "hesitant", "friendly", "confused", "angry"]]

print()
print("=" * 80)
print("CEM PERFORMANCE SUMMARY (partial results)")
print("=" * 80)
header = f"{'Combination':<25} {'Rounds':>7}  {'Best Reward':>11}  {'At Round':>8}  Trend"
print(header)
print("-" * 80)

results = []
for g, s in combos:
    key = f"{g}_{s}"
    h = OUTPUTS / f"experiment_cem_{key}" / "training_history.csv"
    if h.exists():
        rows = list(csv.DictReader(open(h)))
        if rows:
            rewards = [float(r["best_round_reward"]) for r in rows]
            best = max(rewards)
            best_idx = rewards.index(best) + 1
            n = len(rows)
            early = sum(rewards[:3]) / min(3, n)
            late = sum(rewards[-3:]) / min(3, n)
            if late > early + 0.02:
                trend = "improving"
            elif late < early - 0.02:
                trend = "declining"
            else:
                trend = "stable"
            print(f"{key:<25} {n:>4}/15   {best:>11.4f}  round {best_idx:>2}/{n:<2}   {trend}")
            results.append((key, n, best, rewards))
    else:
        print(f"{key:<25} {'NOT STARTED':>20}")

print()
print("=" * 80)
print("BY GESTURE (best reward per gesture)")
print("=" * 80)
for gesture in ["point", "wave", "reach"]:
    gr = [(k, n, b, r) for k, n, b, r in results if k.startswith(gesture)]
    if gr:
        best = max(gr, key=lambda x: x[2])
        all_str = ", ".join(f"{k.split('_')[1]}={b:.3f}" for k, n, b, r in sorted(gr))
        print(f"  {gesture.upper():<8}: best combo={best[0]:<22} reward={best[2]:.4f}")
        print(f"           all: {all_str}")

print()
print("=" * 80)
print("BY STATE (best reward per state)")
print("=" * 80)
for state in ["friendly", "confused", "angry"]:
    sr = [(k, n, b, r) for k, n, b, r in results if k.endswith(state)]
    if sr:
        best = max(sr, key=lambda x: x[2])
        all_str = ", ".join(f"{k.split('_')[0]}={b:.3f}" for k, n, b, r in sorted(sr))
        print(f"  {state.upper():<12}: best combo={best[0]:<22} reward={best[2]:.4f}")
        print(f"               all: {all_str}")

print()
print("=" * 80)
print("HIGHLIGHTS")
print("=" * 80)
if results:
    overall_best = max(results, key=lambda x: x[2])
    overall_worst = min(results, key=lambda x: x[2])
    print(f"  Highest reward:  {overall_best[0]:<25} = {overall_best[2]:.4f}")
    print(f"  Lowest reward:   {overall_worst[0]:<25} = {overall_worst[2]:.4f}")
    
    improving = [(k, n, b, r) for k, n, b, r in results
                 if n >= 3 and sum(r[-3:]) / 3 > sum(r[:3]) / 3 + 0.02]
    if improving:
        print(f"  Still improving: {', '.join(k for k, *_ in improving)}")

print()
print(f"Combos with data:     {len(results)}/12")
print(f"Rounds completed:     {sum(n for _, n, _, _ in results)} / {12 * 15} target ({100 * sum(n for _, n, _, _ in results) // (12*15)}%)")
print(f"Missing combos:       {12 - len(results)}")
