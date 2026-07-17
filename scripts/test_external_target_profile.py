"""Small smoke test for Step 1: externally supplied Laban targets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laban_rl.optimiser_api import optimise_laban_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", choices=["wave", "reach", "point"], default="wave")
    parser.add_argument("--target-state", default="confident")
    parser.add_argument("--out", default="outputs/step1_external_profile_smoke_test")
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=3)
    parser.add_argument("--local-maxiter", type=int, default=5)
    args = parser.parse_args()

    target_profile = {
        "weight": 0.80,
        "time": 0.75,
        "flow_boundness": 0.65,
        "space_indirectness": 0.20,
        "shape_arcness": 0.75,
    }

    result = optimise_laban_target(
        gesture=args.gesture,
        target_state=args.target_state,
        target_profile=target_profile,
        out_dir=args.out,
        optimiser_overrides={
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "local_maxiter": args.local_maxiter,
        },
    )

    print("\n" + "=" * 72)
    print("STEP 1 API RESULT")
    print("=" * 72)
    print(f"Gesture:          {result.gesture}")
    print(f"Target state:     {result.target_state}")
    print(f"Inner loss:       {result.inner_loss:.6f}")
    print(f"Realisation RMSE: {result.realisation_rmse:.6f}")
    print("\nRequested -> achieved")
    for key, requested in result.requested_profile.items():
        print(f"  {key:22s}: {requested:.4f} -> {result.achieved_profile[key]:.4f}")


if __name__ == "__main__":
    main()
