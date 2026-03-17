"""
Main Pipeline — BoP Nowcasting Pilot Study
============================================

Run this script to execute the full pipeline:
  1. Download data from Eurostat
  2. Prepare modeling dataset
  3. Run expanding-window model evaluation
  4. Generate visualizations

Usage:
  python run_pilot.py

Author: PhD Pilot Study
Date: March 2026
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def main():
    print("+" + "=" * 58 + "+")
    print("|  BOP NOWCASTING PILOT STUDY -- FULL PIPELINE              |")
    print("|  Chapter 1: AI vs. Econometrics for BoP Nowcasting       |")
    print("+" + "=" * 58 + "+")

    # -- Step 1: Download Data --
    print("\n\n" + "#" * 60)
    print("  STEP 1/4: DOWNLOADING DATA FROM EUROSTAT")
    print("#" * 60)

    from src.download_data import build_dataset  # noqa

    df = build_dataset()

    if df.empty:
        print("\n[ERROR] Data pipeline failed. Cannot continue.")
        return

    # -- Step 2: Run Models --
    print("\n\n" + "#" * 60)
    print("  STEP 2/4: RUNNING MODEL EVALUATION")
    print("#" * 60)

    from src.models import main as run_models  # noqa
    run_models()

    # -- Step 3: Generate Visualizations --
    print("\n\n" + "#" * 60)
    print("  STEP 3/4: GENERATING VISUALIZATIONS")
    print("#" * 60)

    try:
        from src.visualize import main as run_viz  # noqa
        run_viz()
    except Exception as e:
        print(f"  [WARN] Visualization error: {e}")
        print("  Results are still saved in outputs/")

    # -- Step 4: Summary --
    print("\n\n" + "#" * 60)
    print("  STEP 4/4: SUMMARY")
    print("#" * 60)

    output_dir = Path(__file__).parent / "outputs"
    print(f"\n  Output files in {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    - {f.name:<30} ({size:.0f} KB)")

    print("\n" + "=" * 60)
    print("  [OK] PILOT STUDY PIPELINE COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Review outputs/model_comparison.csv for RMSE results")
    print("  2. Check outputs/forecast_comparison.png for visual fit")
    print("  3. If results look promising, expand to monthly data")
    print("     and additional BoP components")
    print("  4. Share results memo with potential PhD supervisors")


if __name__ == "__main__":
    main()
