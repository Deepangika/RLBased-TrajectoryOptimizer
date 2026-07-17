#!/usr/bin/env python3
"""
Batch runner for CEM training across all gesture-state combinations.

Runs CEM on all 9 combinations:
  Gestures: point, wave, reach
  States: friendly, confused, angry

Generates comprehensive results summary and comparison.
"""
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# All combinations to run
GESTURES = ["point", "wave", "reach"]
STATES = ["confident", "calm", "hesitant", "friendly", "confused", "angry"]
COMBINATIONS = [(g, s) for g in GESTURES for s in STATES]

def run_cem_training(gesture: str, state: str, verbose: bool = True) -> dict:
    """Run CEM training for a single gesture-state combination."""
    
    output_dir = OUTPUTS_DIR / f"experiment_cem_{gesture}_{state}"
    
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
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running CEM: {gesture} gesture, {state} state")
        print(f"Output: {output_dir}")
        print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per combination
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Try to load results summary
            summary_file = output_dir / "results_summary.json"
            best_reward = None
            best_round = None
            
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                        best_reward = summary.get("best_reward")
                        best_round = summary.get("best_round_index", -1) + 1
                except:
                    pass
            
            status = "✅ SUCCESS"
            if verbose:
                print(f"{status} - Elapsed: {elapsed:.1f}s")
                if best_reward is not None:
                    print(f"  Best Reward: {best_reward:.4f} at round {best_round}")
            
            return {
                "gesture": gesture,
                "state": state,
                "status": "success",
                "elapsed": elapsed,
                "best_reward": best_reward,
                "best_round": best_round,
                "output_dir": str(output_dir),
                "return_code": 0
            }
        else:
            print(f"❌ FAILED - Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])  # First 500 chars
            
            return {
                "gesture": gesture,
                "state": state,
                "status": "failed",
                "elapsed": elapsed,
                "best_reward": None,
                "best_round": None,
                "output_dir": str(output_dir),
                "return_code": result.returncode,
                "error": result.stderr[:200] if result.stderr else "Unknown error"
            }
    
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT - Exceeded 1 hour limit")
        return {
            "gesture": gesture,
            "state": state,
            "status": "timeout",
            "elapsed": 3600,
            "best_reward": None,
            "best_round": None,
            "output_dir": str(output_dir),
            "error": "Training exceeded 1 hour timeout"
        }
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            "gesture": gesture,
            "state": state,
            "status": "error",
            "elapsed": time.time() - start_time,
            "best_reward": None,
            "best_round": None,
            "output_dir": str(output_dir),
            "error": str(e)
        }

def main():
    """Run CEM training on all combinations."""
    
    print("\n" + "="*80)
    print("CEM TRAINING: ALL GESTURE-STATE COMBINATIONS")
    print("="*80)
    print(f"Total combinations: {len(COMBINATIONS)}")
    print(f"Expected training time: ~{len(COMBINATIONS) * 45} minutes (45 min per combo)")
    print("="*80)
    
    start_time = time.time()
    results = []
    
    # Run all combinations
    for i, (gesture, state) in enumerate(COMBINATIONS, 1):
        print(f"\n[{i}/{len(COMBINATIONS)}] {gesture} + {state}")
        result = run_cem_training(gesture, state, verbose=True)
        results.append(result)
    
    total_elapsed = time.time() - start_time
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"\nCompleted: {len(successful)}/{len(COMBINATIONS)}")
    print(f"Success rate: {100*len(successful)/len(COMBINATIONS):.1f}%")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    
    if successful:
        print("\n✅ SUCCESSFUL RUNS:")
        print(f"{'Gesture':<10} {'State':<12} {'Best Reward':<15} {'Best Round':<12}")
        print("-" * 50)
        
        for r in sorted(successful, key=lambda x: x.get("best_reward", 0), reverse=True):
            reward = r.get("best_reward", 0)
            round_num = r.get("best_round", 0)
            print(f"{r['gesture']:<10} {r['state']:<12} {reward:<15.4f} {round_num:<12}")
    
    if failed:
        print(f"\n❌ FAILED RUNS ({len(failed)}):")
        for r in failed:
            print(f"  {r['gesture']} + {r['state']}: {r['status']}")
            if r.get("error"):
                print(f"    Error: {r['error'][:60]}")
    
    # Save results to JSON
    results_file = OUTPUTS_DIR / "cem_batch_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_combinations": len(COMBINATIONS),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_seconds": total_elapsed,
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Create comparison table
    comparison_file = OUTPUTS_DIR / "cem_comparison_all_combinations.csv"
    with open(comparison_file, "w") as f:
        f.write("gesture,state,best_reward,best_round,elapsed_seconds,status\n")
        for r in results:
            f.write(f"{r['gesture']},{r['state']},{r.get('best_reward', '')},{r.get('best_round', '')},{r['elapsed']:.1f},{r['status']}\n")
    
    print(f"✅ Comparison table saved to: {comparison_file}")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
