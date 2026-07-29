#!/usr/bin/env python3
"""
Monitor progress of CEM batch training across all combinations.
Shows real-time status of each run.
"""
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def check_training_progress():
    """Check completion status of all combinations."""
    
    combinations = [
        (g, s) for g in ["point", "wave", "reach"]
        for s in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    ]
    
    completed = []
    in_progress = []
    not_started = []
    
    for gesture, state in combinations:
        key = f"{gesture}_{state}"
        output_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        summary_file = output_dir / "results_summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                    completed.append({
                        "name": f"{gesture}::{state}",
                        "reward": data.get("best_reward", 0),
                        "rounds": data.get("num_rounds_completed", 0),
                        "path": output_dir
                    })
            except:
                in_progress.append(f"{gesture}::{state}")
        else:
            # Check if training is active (look for any output files)
            if output_dir.exists() and any(output_dir.glob("*.csv")):
                in_progress.append(f"{gesture}::{state}")
            else:
                not_started.append(f"{gesture}::{state}")
    
    print("\n" + "="*80)
    print("CEM BATCH TRAINING PROGRESS")
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\n✅ COMPLETED: {len(completed)}/18")
    if completed:
        print(f"{'Combination':<25} {'Best Reward':<15} {'Rounds':<10}")
        print("-" * 50)
        for c in sorted(completed, key=lambda x: x["reward"], reverse=True):
            print(f"{c['name']:<25} {c['reward']:<15.4f} {c['rounds']:<10}")
    
    print(f"\n⏳ IN PROGRESS: {len(in_progress)}/18")
    for name in in_progress:
        print(f"  - {name}")
    
    print(f"\n⬜ NOT STARTED: {len(not_started)}/18")
    for name in not_started[:5]:  # Show first 5
        print(f"  - {name}")
    if len(not_started) > 5:
        print(f"  ... and {len(not_started)-5} more")
    
    # Check batch results summary
    batch_results_file = OUTPUTS_DIR / "cem_batch_results.json"
    if batch_results_file.exists():
        try:
            with open(batch_results_file) as f:
                batch = json.load(f)
                elapsed = batch.get("total_time_seconds", 0)
                print(f"\nBatch Summary:")
                print(f"  Started: {batch.get('timestamp', 'Unknown')}")
                print(f"  Elapsed: {elapsed/3600:.1f} hours")
                print(f"  Status: {batch.get('successful')}/{batch.get('total_combinations')} successful")
        except:
            pass
    
    print("\n" + "="*80)
    print(f"Progress: {len(completed)}/{len(completed)+len(in_progress)+len(not_started)}")
    print(f"Completion Rate: {100*len(completed)/(len(completed)+len(in_progress)+len(not_started)):.1f}%")
    print("="*80 + "\n")
    
    return len(completed), len(in_progress), len(not_started)

def continuous_monitor(interval=60):
    """Monitor continuously, updating every N seconds."""
    
    print("Starting continuous monitoring. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            check_training_progress()
            print(f"Next update in {interval} seconds...\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        continuous_monitor(interval)
    else:
        check_training_progress()
