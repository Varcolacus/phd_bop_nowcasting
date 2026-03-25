#!/usr/bin/env python3
"""
validate_tables.py — Cross-validate LaTeX tables against CSV outputs.

Reads key CSV output files, parses corresponding LaTeX table values, and
reports any discrepancies beyond a configurable tolerance. Exits with
code 1 if mismatches are found.

Usage:
    python validate_tables.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs"
LATEX_DIR = Path(__file__).resolve().parent.parent / "04_latex" / "chapters"

TOLERANCE_ABS = 1.0      # absolute tolerance for RMSE values
TOLERANCE_PCT = 0.5      # absolute tolerance for percentage values
mismatches = []


def parse_latex_number(s):
    """Strip LaTeX formatting and return a float (or None)."""
    s = s.strip()
    s = s.replace("$-$", "-").replace("$<$", "<").replace("$>$", ">")
    s = s.replace(",", "").replace("\\%", "").replace("%", "")
    s = s.replace("{", "").replace("}", "").replace("---", "")
    s = re.sub(r"\\emph\{[^}]*\}", "", s)
    s = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", s)
    s = s.strip()
    if not s or s.startswith("<") or s.startswith(">") or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_latex_table(tex_path, label):
    """Extract rows between \\midrule and \\bottomrule in the table with given label."""
    text = tex_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\\label\{" + re.escape(label) + r"\}.*?"
        r"\\midrule\s*(.*?)\\bottomrule",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return []
    body = m.group(1)
    rows = []
    for line in body.strip().split("\\\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def check(label, csv_val, latex_val, tol, context):
    """Compare two values within tolerance, recording mismatches."""
    if csv_val is None or latex_val is None:
        return
    if np.isnan(csv_val) or np.isnan(latex_val):
        return
    if abs(csv_val - latex_val) > tol:
        mismatches.append({
            "table": label,
            "context": context,
            "csv_value": csv_val,
            "latex_value": latex_val,
            "diff": csv_val - latex_val,
        })


def validate_ch1_quarterly():
    """Validate Chapter 1 main model comparison table."""
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)
    rows = extract_latex_table(LATEX_DIR / "chapter1.tex", "tab:ch1_quarterly")

    if not rows:
        print("  [SKIP] Could not parse tab:ch1_quarterly from LaTeX")
        return

    # Map LaTeX model names to CSV model names
    name_map = {
        "AR(1)": "AR(1)", "AR(4)": "AR(4)", "DFM": "DFM",
        "Bridge": "Bridge", "Ridge": "Ridge", "LASSO": "LASSO",
        "Gradient Boosting": "GradientBoosting", "XGBoost": "XGBoost",
        "LSTM": "LSTM", "GRU": "GRU",
    }

    for row_cells in rows:
        if len(row_cells) < 2:
            continue
        tex_name = row_cells[0].strip().replace("\\_", "_")
        csv_name = name_map.get(tex_name)
        if csv_name is None:
            continue
        csv_row = df[df["model"] == csv_name]
        if csv_row.empty:
            continue
        csv_rmse = csv_row["rmse"].values[0]
        latex_rmse = parse_latex_number(row_cells[1])
        check("ch1_quarterly", csv_rmse, latex_rmse, TOLERANCE_ABS,
              f"{tex_name} RMSE")

    print(f"  [OK] ch1_quarterly checked ({len(rows)} rows)")


def validate_ch2_ablation():
    """Validate Chapter 2 ablation results table."""
    csv_path = OUTPUT_DIR / "chapter2" / "ablation_results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)
    rows = extract_latex_table(LATEX_DIR / "chapter2.tex", "tab:ch2_ablation")

    if not rows:
        print("  [SKIP] Could not parse tab:ch2_ablation from LaTeX")
        return

    for row_cells in rows:
        if len(row_cells) < 3:
            continue
        spec = row_cells[0].strip()
        csv_row = df[df["specification"] == spec]
        if csv_row.empty:
            continue
        # Compare Ridge RMSE (column 2 in LaTeX)
        if "ridge_rmse" in csv_row.columns:
            csv_val = csv_row["ridge_rmse"].values[0]
            latex_val = parse_latex_number(row_cells[1])
            check("ch2_ablation", csv_val, latex_val, TOLERANCE_ABS,
                  f"{spec} Ridge RMSE")

    print(f"  [OK] ch2_ablation checked ({len(rows)} rows)")


def validate_ch3_crisis():
    """Validate Chapter 3 crisis-episode table."""
    csv_path = OUTPUT_DIR / "chapter3" / "crisis_evaluation.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)
    rows = extract_latex_table(LATEX_DIR / "chapter3.tex", "tab:ch3_crisis")

    if not rows:
        print("  [SKIP] Could not parse tab:ch3_crisis from LaTeX")
        return

    for row_cells in rows:
        if len(row_cells) < 3:
            continue
        episode = row_cells[0].strip()
        model = row_cells[1].strip()
        csv_row = df[(df["episode"] == episode) & (df["model"] == model)]
        if csv_row.empty:
            continue
        csv_rmse = csv_row["rmse"].values[0]
        latex_rmse = parse_latex_number(row_cells[2])
        check("ch3_crisis", csv_rmse, latex_rmse, TOLERANCE_ABS,
              f"{episode}/{model} RMSE")

    print(f"  [OK] ch3_crisis checked ({len(rows)} rows)")


def validate_ch3_cross_country():
    """Validate Chapter 3 cross-country table."""
    csv_path = OUTPUT_DIR / "chapter3" / "cross_country_results.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)
    rows = extract_latex_table(LATEX_DIR / "chapter3.tex", "tab:ch3_xcountry")

    if not rows:
        print("  [SKIP] Could not parse tab:ch3_xcountry from LaTeX")
        return

    for row_cells in rows:
        if len(row_cells) < 3:
            continue
        country = row_cells[0].strip()
        model = row_cells[1].strip()
        csv_row = df[(df["country"] == country) & (df["model"] == model)]
        if csv_row.empty:
            continue
        csv_rmse = csv_row["rmse"].values[0]
        latex_rmse = parse_latex_number(row_cells[2])
        check("ch3_xcountry", csv_rmse, latex_rmse, TOLERANCE_ABS,
              f"{country}/{model} RMSE")

    print(f"  [OK] ch3_xcountry checked ({len(rows)} rows)")


def main():
    print("=" * 60)
    print("  LATEX TABLE VALIDATION vs CSV OUTPUTS")
    print("=" * 60)
    print()

    validate_ch1_quarterly()
    validate_ch2_ablation()
    validate_ch3_crisis()
    validate_ch3_cross_country()

    print()
    if mismatches:
        print(f"  MISMATCHES FOUND: {len(mismatches)}")
        print(f"  {'Table':<18} {'Context':<30} {'CSV':>10} {'LaTeX':>10} {'Diff':>10}")
        print("  " + "-" * 80)
        for m in mismatches:
            print(f"  {m['table']:<18} {m['context']:<30} "
                  f"{m['csv_value']:>10.1f} {m['latex_value']:>10.1f} "
                  f"{m['diff']:>+10.1f}")
        print("\n  ACTION: Update LaTeX tables to match CSV outputs.")
        sys.exit(1)
    else:
        print("  All checked tables are consistent with CSV outputs.")
        sys.exit(0)


if __name__ == "__main__":
    main()
