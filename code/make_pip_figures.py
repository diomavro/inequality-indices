"""Figures for the PIP application (Section: Do the indices agree?).

Reads data/processed/pip_indices.csv and data/processed/pip_pairs.csv,
writes figures/pip_countries.pdf, pip_disagreement.pdf, pip_by_size.pdf.
Run from code/:  python3 make_pip_figures.py   (after pip_disagreement.py)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Validated categorical order (dataviz default palette); lines also differ by
# dash + marker, so identity never rests on colour alone.
SERIES = [
    ("gini", "Gini", "#2a78d6", "-", "o"),
    ("mld", "MLD ($GE_0$)", "#eb6834", "--", "s"),
    ("ge2", "$GE_2$", "#1baf7a", "-.", "^"),
    ("atk2", "Atkinson $\\varepsilon=2$", "#eda100", ":", "D"),
    ("abs_gini", "Absolute Gini", "#e87ba4", (0, (5, 1, 1, 1)), "v"),
]
COUNTRIES = [
    ("USA", "income", 0, "United States"),
    ("GBR", "income", 1, "United Kingdom"),
    ("BRA", "income", 0, "Brazil"),
    ("CHN", "consumption", 1, "China"),
]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def source_caption(fig, text, y=-0.02):
    fig.text(0.02, y, text, fontsize=7.5, color=INK2, ha="left", va="top")


def countries(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=False)
    for ax, (c, w, sc, name) in zip(axes.flat, COUNTRIES):
        g = panel.query(
            "country_code == @c and welfare_type == @w and "
            "survey_comparability == @sc and reporting_level == 'national'"
        ).sort_values("year")
        for key, label, col, ls, mk in SERIES:
            ax.plot(
                g.year,
                100 * g[key] / g[key].iloc[0],
                color=col,
                ls=ls,
                marker=mk,
                markersize=3,
                markevery=max(1, len(g) // 6),
                label=label,
            )
        ax.axhline(100, color=INK2, lw=0.6)
        ax.set_title(
            f"{name}, {g.year.min()}–{g.year.max()} ({w})",
            fontsize=9,
            color=INK,
            loc="left",
        )
        ax.grid(axis="y", color=GRID, lw=0.6)
        if ax in axes[:, 0]:
            ax.set_ylabel("Index, first survey = 100")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    source_caption(
        fig,
        "Source: World Bank PIP 100-bin distributions (2017 PPP); one PIP comparable spell per panel.",
        y=-0.01,
    )
    fig.savefig(FIG / "pip_countries.pdf", bbox_inches="tight")
    plt.close(fig)


def heatmap(pairs: pd.DataFrame) -> None:
    keys = [
        ("gini", "Gini"),
        ("zenga", "Zenga $Z^*$"),
        ("mld", "MLD"),
        ("theil", "Theil"),
        ("ge2", "$GE_2$"),
        ("atk2", "Atk. $\\varepsilon$=2"),
        ("vl", "VL"),
        ("p90p10", "P90/P10"),
        ("abs_gini", "Abs. Gini"),
    ]
    S = {k: np.sign(pairs[f"d_{k}"]) for k, _ in keys}
    M = np.array(
        [
            [np.nan if a == b else float((S[a] * S[b] < 0).mean()) for b, _ in keys]
            for a, _ in keys
        ]
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.45)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys], rotation=45, ha="right")
    ax.set_yticks(range(len(keys)), [lab for _, lab in keys])
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{100 * M[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if M[i, j] > 0.28 else INK,
                )
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Share of survey pairs with opposite signs")
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4], labels=["0%", "10%", "20%", "30%", "40%"])
    fig.tight_layout()
    source_caption(
        fig,
        f"Cells: % of {len(pairs):,} consecutive comparable PIP survey pairs in which the two indices move in opposite directions.",
        y=0.0,
    )
    fig.savefig(FIG / "pip_disagreement.pdf", bbox_inches="tight")
    plt.close(fig)


def by_size(pairs: pd.DataFrame) -> None:
    from pip_disagreement import TP as tp

    signs = pd.concat([np.sign(pairs[f"d_{m}"]) for m in tp], axis=1)
    dis = signs.nunique(axis=1).gt(1)
    bins = [0, 0.0025, 0.005, 0.01, 0.02, 1]
    labels = ["<0.25", "0.25–0.5", "0.5–1", "1–2", ">2"]
    size = pd.cut(pairs.d_gini.abs(), bins, labels=labels, include_lowest=True)
    # Under Lorenz dominance the rate is identically zero (Theorem), so it is
    # stated in the caption rather than drawn as empty bars.
    assert not dis[pairs.lorenz == "dominance"].any()
    classes = [
        ("single_cross", "Single crossing", "#2a78d6"),
        ("multi_cross", "Multiple crossings", "#eb6834"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    wd = 0.38
    for k, (cls, lab, col) in enumerate(classes):
        sel = pairs.lorenz == cls
        rate = dis[sel].groupby(size[sel], observed=False).mean().reindex(labels)
        ax.bar(
            x + (k - 0.5) * wd,
            100 * rate.fillna(0),
            width=wd - 0.03,
            color=col,
            label=lab,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Absolute change in Gini between surveys (Gini points)")
    ax.set_ylabel("% of pairs where transfer-\nprinciple indices disagree")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    source_caption(
        fig,
        "Lorenz-dominance pairs: 0% in every bin, as the transfer principle requires.\n"
        "Transfer-principle indices (one per distinct ordering): Gini, Zenga $Z^*$, $GE_{-1}$, MLD, $GE_{0.5}$, Theil, $GE_2$.",
        y=0.0,
    )
    fig.savefig(FIG / "pip_by_size.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    panel = pd.read_csv(ROOT / "data/processed/pip_indices.csv")
    pairs = pd.read_csv(ROOT / "data/processed/pip_pairs.csv")
    countries(panel)
    heatmap(pairs)
    by_size(pairs)
    print("wrote pip_countries.pdf, pip_disagreement.pdf, pip_by_size.pdf")
