#!/usr/bin/env python3
"""
Smart batch recovery runner:
1. Resumes all partial combinations from checkpoints
2. Runs all not-started combinations
3. Better error handling for API rate limits
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def identify_combos():
    """Identify which combos are partial, not started, etc."""
    
    combinations = [
        (g, s) for g in ["point", "wave", "reach"]
        for s in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    ]
    
    partial = []
    not_started = []
    
    for gesture, state in combinations:
        key = f"{gesture}_{state}"
        exp_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        history_file = exp_dir / "training_history.csv"
        checkpoint_file = exp_dir / "latest_checkpoint.pt"
        
        if checkpoint_file.exists():
            partial.append((gesture, state, key))
        else:
            not_started.append((gesture, state, key))
    
    return partial, not_started

def run_single_combo(gesture, state, output_dir, resume=False, timeout=7200):
    """Run a single CEM training combination."""
    
    cmd = [
        "python",
        "scripts/train_cem_contextual_bandit.py",
        "--gesture", gesture,
        "--target-state", state,
        "--rounds", "15",
        "--cem-samples-per-round", "5",
        "--cem-elite-fraction", "0.5",
        "--cem-initial-width", "0.15",
        "--exploration-decay-rate", "0.88",
        "--repeats", "3",
        "--reward-margin-mode", "raw",
        "--out", str(output_dir),
        "--overwrite"
    ]
    
    print(f"\n{'='*80}")
    print(f"{'RESUME' if resume else 'START'}: {gesture} + {state}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Check if there's a results file
            summary_file = output_dir / "results_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                        best_reward = summary.get("best_reward")
                        best_round = summary.get("best_round_index", -1) + 1
                        print(f"✅ SUCCESS - Best: {best_reward:.4f} at round {best_round}")
                except:
                    print(f"✅ SUCCESS - Elapsed: {elapsed:.0f}s")
            else:
                print(f"✅ SUCCESS - Elapsed: {elapsed:.0f}s")
            
            return {
                "gesture": gesture,
                "state": state,
                "status": "success",
                "elapsed": elapsed,
                "resume": resume
            }
        else:
            # Check if it's a Gemini API error
            if "503" in result.stderr or "overloaded" in result.stderr or "rate_limit" in result.stderr:
                print(f"⚠️ API RATE LIMITED - This usually recovers in a few minutes")
                return {
                    "gesture": gesture,
                    "state": state,
                    "status": "api_rate_limited",
                    "elapsed": elapsed,
                    "resume": resume,
                    "error": "Gemini API rate limit"
                }
            elif "timeout" in result.stderr.lower():
                print(f"❌ TIMEOUT after {elapsed:.0f}s")
                return {
                    "gesture": gesture,
                    "state": state,
                    "status": "timeout",
                    "elapsed": elapsed,
                    "resume": resume
                }
            else:
                print(f"❌ FAILED - Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:300]}")
                
                return {
                    "gesture": gesture,
                    "state": state,
                    "status": "failed",
                    "elapsed": elapsed,
                    "resume": resume,
                    "error": result.stderr[:200] if result.stderr else "Unknown error"
                }
    
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT - Exceeded {timeout}s limit")
        return {
            "gesture": gesture,
            "state": state,
            "status": "timeout",
            "elapsed": timeout,
            "resume": resume,
            "error": "Training exceeded timeout"
        }
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            "gesture": gesture,
            "state": state,
            "status": "error",
            "elapsed": time.time() - start_time,
            "resume": resume,
            "error": str(e)
        }

def main():
    """Run smart batch recovery."""
    
    partial, not_started = identify_combos()
    
    print("\n" + "="*80)
    print("CEM SMART BATCH RECOVERY")
    print("="*80)
    print(f"\nPartial (resume): {len(partial)}")
    for g, s, k in partial:
        print(f"  - {g} + {s}")
    
    print(f"\nNot started: {len(not_started)}")
    for g, s, k in not_started:
        print(f"  - {g} + {s}")
    
    print("\n" + "="*80)
    print("STRATEGY")
    print("="*80)
    print("1. Resume partial combos first (faster, have checkpoints)")
    print("2. Then run new combos")
    print("3. Better error handling with API rate limit detection")
    print(f"Est. total time: ~{(len(partial)*3 + len(not_started)*5)} hours")
    
    # Run resumable combos first
    all_results = []
    
    print("\n" + "="*80)
    print("PHASE 1: RESUMING PARTIAL COMBOS")
    print("="*80)
    
    for i, (gesture, state, key) in enumerate(partial, 1):
        output_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        result = run_single_combo(gesture, state, output_dir, resume=True, timeout=3600)
        all_results.append(result)
        
        # If we hit API rate limits, wait longer and continue
        if result["status"] == "api_rate_limited":
            print("⏳ API rate limited. Waiting 3 minutes before next combo...")
            time.sleep(180)
        elif i < len(partial):
            # Add delay between combos to avoid rate limits
            print(f"⏳ Waiting 90 seconds before next combo...")
            time.sleep(90)
    
    # Then run new combos
    print("\n" + "="*80)
    print("PHASE 2: RUNNING NEW COMBOS")
    print("="*80)
    
    for i, (gesture, state, key) in enumerate(not_started, 1):
        output_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        result = run_single_combo(gesture, state, output_dir, resume=False, timeout=3600)
        all_results.append(result)
        
        # If we hit API rate limits, wait longer before next one
        if result["status"] == "api_rate_limited":
            print("⏳ API rate limited. Waiting 3 minutes before next combo...")
            time.sleep(180)
        elif i < len(not_started):
            # Add delay between combos to avoid rate limits
            print(f"⏳ Waiting 90 seconds before next combo...")
            time.sleep(90)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    successful = [r for r in all_results if r["status"] == "success"]
    rate_limited = [r for r in all_results if r["status"] == "api_rate_limited"]
    failed = [r for r in all_results if r["status"] not in ["success", "api_rate_limited"]]
    
    print(f"\n✅ Successful: {len(successful)}/{len(all_results)}")
    print(f"⚠️ Rate limited: {len(rate_limited)}/{len(all_results)}")
    print(f"❌ Failed: {len(failed)}/{len(all_results)}")
    
    if successful:
        for r in successful:
            print(f"  ✅ {r['gesture']} + {r['state']:<12} (resumed={r['resume']})")
    
    if rate_limited:
        print(f"\nRate-limited combos (can be retried):")
        for r in rate_limited:
            print(f"  ⚠️ {r['gesture']} + {r['state']:<12}")
    
    if failed:
        print(f"\nFailed combos:")
        for r in failed:
            print(f"  ❌ {r['gesture']} + {r['state']:<12} ({r['status']})")
    
    # Save results
    results_file = OUTPUTS_DIR / "cem_recovery_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "partial_resumed": len(partial),
            "new_started": len(not_started),
            "successful": len(successful),
            "rate_limited": len(rate_limited),
            "failed": len(failed),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # If all successful, run analysis
    if len(failed) == 0 and len(rate_limited) == 0:
        print("\n🎉 All combos completed successfully!")
        print("Next: Run `python scripts/analyze_cem_all_combinations.py` to generate visualizations")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
