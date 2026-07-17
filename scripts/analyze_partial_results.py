#!/usr/bin/env python3
"""
Analyze partial CEM batch results and identify what needs to be rerun.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def analyze_partial_results():
    """Analyze which combinations have results and which need rerunning."""
    
    combinations = [
        (g, s) for g in ["point", "wave", "reach"]
        for s in ["confident", "calm", "hesitant", "friendly", "confused", "angry"]
    ]
    
    completed = {}
    partial = {}
    missing = []
    
    for gesture, state in combinations:
        key = f"{gesture}_{state}"
        exp_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        history_file = exp_dir / "training_history.csv"
        results_file = exp_dir / "results_summary.json"
        
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                completed[key] = results
            except:
                pass
        elif history_file.exists():
            try:
                with open(history_file) as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                
                if data:
                    last_row = data[-1]
                    rounds = len(data)
                    best_reward = float(last_row.get('best_round_reward', 0))
                    partial[key] = {
                        'rounds_completed': rounds,
                        'last_best_reward': best_reward,
                        'rounds_remaining': 15 - rounds
                    }
            except:
                pass
        else:
            missing.append(key)
    
    return completed, partial, missing

def main():
    completed, partial, missing = analyze_partial_results()
    
    print("\n" + "="*80)
    print("CEM BATCH TRAINING ANALYSIS: PARTIAL RESULTS")
    print("="*80)
    
    print(f"\n✅ FULLY COMPLETED: {len(completed)}/12")
    for key in sorted(completed.keys()):
        r = completed[key]
        print(f"  {key:<30} Rounds: {r.get('num_rounds_completed', '?'):>2}  Best: {r.get('best_reward', 0):.4f}")
    
    print(f"\n⏳ PARTIALLY COMPLETED: {len(partial)}/12")
    for key in sorted(partial.keys()):
        data = partial[key]
        print(f"  {key:<30} {data['rounds_completed']:>2}/15 rounds completed  Best: {data['last_best_reward']:.4f}")
    
    print(f"\n❌ NOT STARTED: {len(missing)}/12")
    for key in sorted(missing):
        print(f"  {key:<30}")
    
    # Suggest recovery strategy
    print("\n" + "="*80)
    print("RECOVERY STRATEGY")
    print("="*80)
    
    if len(completed) > 0:
        print(f"\n✅ Keep: {len(completed)} completed combinations")
    
    if len(partial) > 0:
        print(f"\n⏳ Can Resume: {len(partial)} partially completed")
        print("   These have checkpoints and can be continued from where they stopped.")
        print("   To resume, run each individually:")
        for key in sorted(partial.keys()):
            gesture, state = key.split('_')
            print(f"   python scripts/train_cem_contextual_bandit.py \\")
            print(f"     --gesture {gesture} --target-state {state} \\")
            print(f"     --rounds 15 --cem-samples-per-round 5 \\")
            print(f"     --out outputs/experiment_cem_{key} --overwrite")
    
    if len(missing) > 0:
        print(f"\n❌ Need to Restart: {len(missing)} combinations not started")
        print("   These should be run fresh.")
    
    # Estimate effort
    print("\n" + "="*80)
    print("EFFORT ESTIMATE")
    print("="*80)
    
    total_remaining = sum(partial[k]['rounds_remaining'] for k in partial)
    new_combinations = len(missing)
    
    estimated_time_partial = total_remaining * 5  # ~5 min per round
    estimated_time_new = new_combinations * 15 * 5  # ~75 min per combination (15 rounds)
    
    print(f"\nTo complete partially finished combos:")
    print(f"  Remaining rounds: {total_remaining}")
    print(f"  Est. time: ~{estimated_time_partial//60} hours ({estimated_time_partial} min)")
    
    print(f"\nTo run {len(missing)} new combinations:")
    print(f"  Est. time: ~{estimated_time_new//60} hours")
    
    print(f"\nTotal est. time: ~{(estimated_time_partial + estimated_time_new)//60} hours")
    
    # Generate recovery script
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if len(partial) > 0:
        print(f"\n1. Option A: Resume partial combos one by one")
        print(f"   - Each will continue from existing checkpoint")
        print(f"   - Est. time: ~{estimated_time_partial//60} hours")
        print(f"\n2. Option B: Re-run all from scratch")
        print(f"   - Cleaner, but loses progress")
        print(f"   - Est. time: ~{(estimated_time_new + len(partial)*75)//60} hours")
    
    print(f"\n3. Analyze current partial results")
    print(f"   - Run: python scripts/analyze_cem_all_combinations.py")
    print(f"   - This will include all {len(completed) + len(partial)} combos found so far")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
