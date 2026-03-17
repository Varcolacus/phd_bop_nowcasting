"""
Visualization for BoP Nowcasting Pilot Study
==============================================

Produces:
  1. Forecast comparison chart (actual vs. all models)
  2. RMSE bar chart comparison
  3. Error distribution plots
  4. Crisis-period zoom plots

Author: PhD Pilot Study
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def load_results():
    """Load forecast results from CSV."""
    path = OUTPUT_DIR / "forecast_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"No results found at {path}. Run 02_models.py first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_comparison():
    """Load model comparison summary."""
    path = OUTPUT_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_forecast_comparison(df):
    """
    Plot actual vs. predicted values for all models.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    models = df["model"].unique()
    colors = {
        "AR(1)": "#999999",
        "AR(4)": "#bbbbbb",
        "OLS": "#2196F3",
        "GradientBoosting": "#FF9800",
        "XGBoost": "#E91E63",
    }

    # Plot actuals (same for all models, take from first)
    first_model = models[0]
    actual_data = df[df["model"] == first_model].sort_values("date")
    ax.plot(actual_data["date"], actual_data["actual"],
            color="black", linewidth=2, label="Actual", zorder=5)

    # Plot each model's predictions
    for model_name in models:
        model_data = df[df["model"] == model_name].sort_values("date")
        color = colors.get(model_name, "#666666")
        alpha = 0.5 if model_name.startswith("AR") else 0.8
        linewidth = 1 if model_name.startswith("AR") else 1.5

        ax.plot(model_data["date"], model_data["predicted"],
                color=color, linewidth=linewidth, alpha=alpha,
                label=model_name, linestyle="--" if model_name.startswith("AR") else "-")

    # Add crisis shading
    crisis_periods = [
        ("2008-07-01", "2009-06-30", "GFC", "#ffcccc"),
        ("2020-01-01", "2020-12-31", "COVID", "#cce5ff"),
        ("2022-01-01", "2022-12-31", "Energy\nshock", "#fff3cd"),
    ]

    for start, end, label, color in crisis_periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        if start_dt >= actual_data["date"].min() and end_dt <= actual_data["date"].max():
            ax.axvspan(start_dt, end_dt, alpha=0.3, color=color, label=label)

    ax.set_title("BoP Nowcasting: Actual vs. Model Predictions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("BoP Component (millions EUR)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved forecast_comparison.png")
    plt.close()


def plot_rmse_comparison(comparison_df):
    """
    Bar chart comparing RMSE across models.
    """
    if comparison_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- RMSE ---
    ax = axes[0]
    models = comparison_df["model"]
    rmse = comparison_df["rmse"]

    colors = ["#999999" if "AR" in m else "#2196F3" if m == "OLS"
              else "#FF9800" if m == "GradientBoosting" else "#E91E63"
              for m in models]

    bars = ax.barh(models, rmse, color=colors, edgecolor="white")
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Root Mean Squared Error", fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, rmse):
        if not np.isnan(val):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=10)

    # --- Direction Accuracy ---
    ax = axes[1]
    dir_acc = comparison_df["direction_accuracy"]

    bars = ax.barh(models, dir_acc, color=colors, edgecolor="white")
    ax.set_xlabel("Direction Accuracy % (higher is better)")
    ax.set_title("Direction Accuracy", fontweight="bold")
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.invert_yaxis()
    ax.legend()

    for bar, val in zip(bars, dir_acc):
        if not np.isnan(val):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved model_comparison.png")
    plt.close()


def plot_error_distribution(df):
    """
    Plot forecast error distributions for each model.
    """
    models = df["model"].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(3 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {
        "AR(1)": "#999999",
        "AR(4)": "#bbbbbb",
        "OLS": "#2196F3",
        "GradientBoosting": "#FF9800",
        "XGBoost": "#E91E63",
    }

    for ax, model_name in zip(axes, models):
        model_data = df[df["model"] == model_name]
        errors = model_data["actual"] - model_data["predicted"]
        errors = errors.dropna()

        color = colors.get(model_name, "#666666")
        ax.hist(errors, bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=errors.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Mean: {errors.mean():.1f}")
        ax.set_title(model_name, fontweight="bold", fontsize=10)
        ax.set_xlabel("Forecast Error")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Frequency")
    fig.suptitle("Forecast Error Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved error_distributions.png")
    plt.close()


def plot_cumulative_sse(df):
    """
    Plot cumulative squared errors over time — shows when each model
    gains or loses accuracy relative to the baseline.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    models = df["model"].unique()
    baseline = "AR(1)"

    colors = {
        "AR(1)": "#999999",
        "AR(4)": "#bbbbbb",
        "OLS": "#2196F3",
        "GradientBoosting": "#FF9800",
        "XGBoost": "#E91E63",
    }

    # Get baseline errors
    base_data = df[df["model"] == baseline].sort_values("date")
    base_se = (base_data["actual"] - base_data["predicted"]) ** 2

    for model_name in models:
        if model_name == baseline:
            continue

        model_data = df[df["model"] == model_name].sort_values("date")
        model_se = (model_data["actual"] - model_data["predicted"]) ** 2

        # Cumulative difference in squared errors (negative = model is better)
        cum_diff = (base_se.values - model_se.values).cumsum()

        color = colors.get(model_name, "#666666")
        ax.plot(model_data["date"].values, cum_diff,
                color=color, linewidth=1.5, label=model_name)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], alpha=0.05, color="green")
    ax.fill_between(ax.get_xlim(), ax.get_ylim()[0], 0, alpha=0.05, color="red")

    ax.set_title("Cumulative Squared Error Difference vs. AR(1)\n(positive = model beats AR(1))",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative SSE difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cumulative_sse.png", dpi=150, bbox_inches="tight")
    print("Saved cumulative_sse.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  GENERATING PILOT STUDY VISUALIZATIONS")
    print("=" * 60)

    df = load_results()
    comparison = load_comparison()

    print(f"\nLoaded {len(df)} forecast observations across {df['model'].nunique()} models")

    plot_forecast_comparison(df)
    plot_rmse_comparison(comparison)
    plot_error_distribution(df)
    plot_cumulative_sse(df)

    print(f"\n[OK] All charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
