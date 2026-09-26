"""
Generate Publication-Quality Figures
=====================================

Produces three PDF figures for the inequality indices paper:

1. ``lorenz_curves.pdf``        — Lorenz curves for three distributions
2. ``transfer_sensitivity.pdf`` — Diminishing transfer sensitivity
3. ``vl_counterexample.pdf``    — Variance-of-logs transfer violation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from inequality_indices import (
    ge_alpha,
    gini,
    mld,
    theil,
    variance_of_logs,
    zenga,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)
# Use serif fonts if available
try:
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "DejaVu Serif", "Computer Modern Roman", "Times New Roman",
    ]
    matplotlib.rcParams["text.usetex"] = False
except Exception:
    pass


# ===================================================================
# Figure 1 — Lorenz curves
# ===================================================================
def lorenz_curve(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cum_pop, cum_income) arrays for the Lorenz curve."""
    ys = np.sort(y)
    cum_income = np.concatenate([[0.0], np.cumsum(ys) / np.sum(ys)])
    cum_pop = np.linspace(0, 1, len(cum_income))
    return cum_pop, cum_income


def fig_lorenz_curves() -> None:
    rng = np.random.default_rng(2024)

    # Three distributions with increasing inequality
    y_low = rng.lognormal(mean=3.0, sigma=0.3, size=1000)
    y_med = rng.lognormal(mean=3.0, sigma=0.7, size=1000)
    y_high = rng.lognormal(mean=3.0, sigma=1.2, size=1000)

    fig, ax = plt.subplots()

    for y, label, ls in [
        (y_low, f"Low inequality (Gini = {gini(y_low):.3f})", "-"),
        (y_med, f"Medium inequality (Gini = {gini(y_med):.3f})", "--"),
        (y_high, f"High inequality (Gini = {gini(y_high):.3f})", "-."),
    ]:
        cp, ci = lorenz_curve(y)
        ax.plot(cp, ci, ls, linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="Perfect equality")

    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of income")
    ax.set_title("Lorenz Curves")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "lorenz_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 2 — Transfer sensitivity
# ===================================================================
def fig_transfer_sensitivity() -> None:
    delta = 1.0
    gap = 10.0
    y_levels = np.arange(5, 201, 1, dtype=float)

    indices: dict[str, object] = {
        "Gini": gini,
        "Theil": theil,
        "MLD": mld,
        "GE(2)": lambda y: ge_alpha(y, alpha=2.0),
    }

    fig, ax = plt.subplots()
    styles = {"Gini": "-", "Theil": "--", "MLD": "-.", "GE(2)": ":"}

    for name, func in indices.items():
        reductions = []
        for y0 in y_levels:
            # Two-person sub-problem embedded in a 4-person distribution
            y_before = np.array([y0, y0 + gap, 500.0, 1000.0])
            y_after = y_before.copy()
            y_after[1] -= delta
            y_after[0] += delta
            reduction = func(y_before) - func(y_after)
            reductions.append(reduction)

        ax.plot(
            y_levels,
            reductions,
            styles[name],
            linewidth=1.8,
            label=name,
        )

    ax.set_xlabel("Income level $y$")
    ax.set_ylabel(r"Reduction in index ($\Delta I$)")
    ax.set_title(
        r"Transfer sensitivity: reduction from $\delta=1$ transfer"
        "\n"
        r"between two individuals with gap $d=10$"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "transfer_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Figure 3 — Variance-of-logs counter-example
# ===================================================================
def fig_vl_counterexample() -> None:
    y_base = np.array([1.0, 2.0, 50.0, 100.0])

    # Progressive transfers of varying δ from person 3 (100) to person 2 (50)
    deltas = np.linspace(0, 24, 200)

    vl_values = []
    gini_values = []
    theil_values = []

    for d in deltas:
        y_t = y_base.copy()
        y_t[3] -= d  # rich person loses δ
        y_t[2] += d  # poorer person gains δ
        vl_values.append(variance_of_logs(y_t))
        gini_values.append(gini(y_t))
        theil_values.append(theil(y_t))

    fig, ax1 = plt.subplots()

    color_vl = "C3"
    color_gini = "C0"
    color_theil = "C1"

    ax1.plot(deltas, vl_values, "-", color=color_vl, linewidth=2, label="VL")
    ax1.set_xlabel(r"Transfer size $\delta$")
    ax1.set_ylabel("Variance of Logs", color=color_vl)
    ax1.tick_params(axis="y", labelcolor=color_vl)

    ax2 = ax1.twinx()
    ax2.plot(
        deltas, gini_values, "--", color=color_gini, linewidth=1.5, label="Gini"
    )
    ax2.plot(
        deltas, theil_values, "-.", color=color_theil, linewidth=1.5, label="Theil"
    )
    ax2.set_ylabel("Gini / Theil", color="black")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")

    ax1.set_title(
        "VL transfer-principle violation\n"
        r"Progressive transfer from income 100 to 50 in $y=(1,2,50,100)$"
    )
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "vl_counterexample.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("Generating figures...")
    fig_lorenz_curves()
    fig_transfer_sensitivity()
    fig_vl_counterexample()
    print("Done.")


if __name__ == "__main__":
    main()
