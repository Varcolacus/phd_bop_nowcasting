#!/usr/bin/env python3
"""
run_all.py — End-to-end reproducibility script.

Chains together:
    1. run_pilot.py       (Chapter 1: pilot study)
    2. run_chapter2.py    (Chapter 2: ablation study)
    3. run_chapter3.py    (Chapter 3: policy implications)
    4. run_monthly.py     (Monthly-frequency extension)
    5. LaTeX compilation  (pdflatex + biber)

Usage:
    python run_all.py          # run everything
    python run_all.py --skip-latex   # skip LaTeX compilation
"""

import argparse
import importlib
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

SCRIPTS = [
    ("Chapter 1 — Pilot Study",        "run_pilot"),
    ("Chapter 2 — Ablation Study",      "run_chapter2"),
    ("Chapter 3 — Policy Implications", "run_chapter3"),
    ("Monthly Extension",               "run_monthly"),
]

LATEX_DIR = Path(__file__).resolve().parent.parent / "04_latex"
LATEX_MAIN = "thesis"


def run_step(label, module_name):
    """Import and run a script's main() function."""
    logger.info("=" * 60)
    logger.info("START: %s  (%s.py)", label, module_name)
    logger.info("=" * 60)
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.perf_counter() - t0
        logger.info("DONE : %s  (%.1f s)", label, elapsed)
        return True
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception("FAIL : %s  (%.1f s)", label, elapsed)
        return False


def compile_latex():
    """Run pdflatex + biber + pdflatex×2 in 04_latex/."""
    if not LATEX_DIR.exists():
        logger.warning("LaTeX directory not found: %s", LATEX_DIR)
        return False

    logger.info("=" * 60)
    logger.info("START: LaTeX compilation")
    logger.info("=" * 60)

    cmds = [
        ["pdflatex", "-interaction=nonstopmode", LATEX_MAIN],
        ["biber", LATEX_MAIN],
        ["pdflatex", "-interaction=nonstopmode", LATEX_MAIN],
        ["pdflatex", "-interaction=nonstopmode", LATEX_MAIN],
    ]

    for cmd in cmds:
        logger.info("  %s", " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=str(LATEX_DIR),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("LaTeX step failed: %s", " ".join(cmd))
            logger.error(result.stdout[-500:] if result.stdout else "(no stdout)")
            return False

    logger.info("DONE : LaTeX compilation  → %s/%s.pdf", LATEX_DIR, LATEX_MAIN)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full thesis pipeline")
    parser.add_argument("--skip-latex", action="store_true",
                        help="Skip LaTeX compilation step")
    args = parser.parse_args()

    logger.info("+" + "=" * 58 + "+")
    logger.info("|  FULL THESIS PIPELINE — run_all.py                      |")
    logger.info("+" + "=" * 58 + "+")
    t_total = time.perf_counter()

    results = {}
    for label, module_name in SCRIPTS:
        ok = run_step(label, module_name)
        results[label] = ok

    if not args.skip_latex:
        results["LaTeX"] = compile_latex()

    elapsed_total = time.perf_counter() - t_total

    # --- Summary ---
    logger.info("")
    logger.info("+" + "=" * 58 + "+")
    logger.info("|  PIPELINE SUMMARY                                        |")
    logger.info("+" + "=" * 58 + "+")
    all_ok = True
    for label, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %-40s  [%s]", label, status)
        if not ok:
            all_ok = False
    logger.info("")
    logger.info("  Total elapsed: %.1f s", elapsed_total)

    if all_ok:
        logger.info("  All steps completed successfully.")
    else:
        logger.warning("  Some steps failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
