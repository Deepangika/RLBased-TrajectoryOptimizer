#!/usr/bin/env python3
"""Run CEM training for all 3 states on the wave gesture only."""
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GESTURE = "wave"
STATES = ["friendly", "confused", "angry"]

def run_combo(state, timeout=14400):  # 4 hours instead of 1 hour
    key = f"{GESTURE}_{state}"
    out_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
    checkpoint = out_dir / "latest_checkpoint.pt"
    resume = checkpoint.exists()

    print(f"\n{'='*70}")
    print(f"{'RESUME' if resume else 'START'}: wave + {state}")
    if resume:
        print(f"  (resuming from existing checkpoint)")
    print(f"  Output: {out_dir}")
    print(f"  Timeout: {timeout//3600} hours")
    print(f"{'='*70}")

    cmd = [
        "python", "scripts/train_cem_contextual_bandit.py",
        "--gesture", GESTURE,
        "--target-state", state,
        "--rounds", "15",
        "--cem-samples-per-round", "5",
        "--cem-elite-fraction", "0.5",
        "--cem-initial-width", "0.15",
        "--exploration-decay-rate", "0.88",
        "--repeats", "2",              # reduced from 3 to ease rate limits
        "--reward-margin-mode", "raw",
        "--out", str(out_dir),
        "--overwrite",
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False, text=True, timeout=timeout)
        elapsed = time.time() - start

        if result.returncode == 0:
            summary_file = out_dir / "results_summary.json"
            if summary_file.exists():
                try:
                    s = json.loads(summary_file.read_text())
                    print(f"\n  ✅ DONE — best reward: {s.get('best_reward', '?'):.4f} "
                          f"at round {s.get('best_round_index', -1)+1}")
                except Exception:
                    pass
            return "success"
        else:
            print(f"  ❌ Failed with return code {result.returncode}")
            return "failed"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ⏱️ TIMEOUT after {elapsed/3600:.1f} hours")
        print(f"  (Partial results are saved. Combo can be resumed later.)")
        return "timeout"
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return "error"

def main():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set $env:GOOGLE_API_KEY before running.")
        return 1

    print(f"\nRunning CEM for: wave × {STATES}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("(--repeats 2 to reduce API call volume)")
    print("(4-hour timeout per combo to allow for Gemini delays)\n")

    results = {}
    for i, state in enumerate(STATES):
        status = run_combo(state)
        results[state] = status

        if status == "timeout":
            print(f"\n  Partial results saved. Continuing to next combo...")
        
        if i < len(STATES) - 1:
            print(f"\n  Waiting 120s before next combo...")
            time.sleep(120)

    print(f"\n{'='*70}")
    print("WAVE GESTURE — RESULTS SUMMARY")
    print(f"{'='*70}")
    for state, status in results.items():
        icon = "✅" if status == "success" else ("⏱️" if status == "timeout" else "❌")
        print(f"  {icon} wave + {state:<12}  {status}")

    print(f"\nCompleted: {datetime.now().strftime('%H:%M:%S')}")
    print("Next: python scripts/analyze_partial_progress.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
