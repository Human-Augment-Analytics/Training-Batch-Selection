"""
Run all experiments for batch selection research.

Usage:
    python -m experiments.run_all [--quick] [--dpo-only] [--ppo-only]
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run batch selection experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--dpo-only", action="store_true", help="Only run DPO experiments")
    parser.add_argument("--ppo-only", action="store_true", help="Only run PPO experiments")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    args = parser.parse_args()

    # Quick mode settings
    if args.quick:
        max_samples = 100
        n_prompts = 50
        num_epochs = 1
    else:
        max_samples = 500
        n_prompts = 200
        num_epochs = 2

    results = {}

    if not args.ppo_only:
        print("\n" + "=" * 70)
        print("RUNNING DPO EXPERIMENTS")
        print("=" * 70 + "\n")

        from experiments.run_dpo_comparison import run_experiment as run_dpo

        dpo_results, dpo_comparison = run_dpo(
            max_samples=max_samples,
            num_epochs=num_epochs,
            output_dir=f"{args.output_dir}/dpo",
        )
        results["dpo"] = {"results": dpo_results, "comparison": dpo_comparison}

    if not args.dpo_only:
        print("\n" + "=" * 70)
        print("RUNNING PPO EXPERIMENTS")
        print("=" * 70 + "\n")

        from experiments.run_ppo_comparison import run_experiment as run_ppo

        ppo_results, ppo_comparison = run_ppo(
            n_prompts=n_prompts,
            num_epochs=num_epochs,
            output_dir=f"{args.output_dir}/ppo",
        )
        results["ppo"] = {"results": ppo_results, "comparison": ppo_comparison}

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    if "dpo" in results:
        print("\nDPO Results:")
        for selector, stats in results["dpo"]["comparison"].items():
            overhead = stats.get("selection_overhead_ratio", 0) * 100
            print(f"  {selector}: overhead={overhead:.1f}%, loss={stats.get('final_loss', 0):.4f}")

    if "ppo" in results:
        print("\nPPO Results:")
        for selector, stats in results["ppo"]["comparison"].items():
            overhead = stats.get("selection_overhead_ratio", 0) * 100
            print(f"  {selector}: overhead={overhead:.1f}%, loss={stats.get('final_loss', 0):.4f}")

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
