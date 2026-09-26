"""Compute every index on each PIP percentile distribution.

Input : data/raw/pip_world_100bin.csv   (World Bank PIP, 100 equal-population
        bins per country-year, 2017 PPP $/day)
        data/raw/pip_summary_ppp2017.csv (PIP survey table, same PPP vintage)
Output: data/processed/pip_indices.csv  (one row per distribution)

Run from code/:  python3 pip_panel.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import inequality_indices as ii

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["country_code", "year", "reporting_level", "welfare_type"]
KOLM_KAPPA = 0.1  # per 2017-PPP $/day; robustness only (Kolm is unit-dependent)


def indices(y: np.ndarray) -> dict:
    """All indices on one 100-bin distribution (equal population weights)."""
    return {
        "gini": ii.gini(y),
        "mld": ii.mld(y),
        "theil": ii.theil(y),
        "ge2": ii.ge_alpha(y, 2.0),
        "ge_m1": ii.ge_alpha(y, -1.0),
        "ge05": ii.ge_alpha(y, 0.5),
        "atk05": ii.atkinson(y, 0.5),
        "atk1": ii.atkinson(y, 1.0),
        "atk2": ii.atkinson(y, 2.0),
        "cv": ii.cv(y),
        "vl": ii.variance_of_logs(y),
        "p90p10": y[89] / y[9],  # bins are sorted percentile means
        "zenga": ii.integral_zenga(y),
        "abs_gini": ii.absolute_gini(y),
        "sd": float(np.sqrt(ii.variance(y))),
        "kolm": ii.kolm(y, KOLM_KAPPA),
        "mean": float(np.mean(y)),
    }


def main() -> None:
    bins = pd.read_csv(ROOT / "data/raw/pip_world_100bin.csv")
    summ = pd.read_csv(ROOT / "data/raw/pip_summary_ppp2017.csv")
    summ = summ.rename(
        columns={
            "reporting_year": "year",
            "gini": "gini_pip",
            "mld": "mld_pip",
            "mean": "mean_pip",
        }
    )

    rows = []
    for key, g in bins.groupby(KEYS, sort=True):
        g = g.sort_values("percentile")
        assert len(g) == 100 and np.allclose(g.pop_share, 0.01), key
        y = g.avg_welfare.to_numpy()
        assert np.all(np.diff(y) >= -1e-9) and y[0] > 0, key
        rows.append(dict(zip(KEYS, key)) | indices(y))
    out = pd.DataFrame(rows)

    meta = [
        "country_name",
        "region_code",
        "survey_acronym",
        "survey_year",
        "survey_comparability",
        "comparable_spell",
        "distribution_type",
        "gini_pip",
        "mld_pip",
        "mean_pip",
    ]
    out = out.merge(summ[KEYS + meta], on=KEYS, how="left", validate="1:1")
    assert out.gini_pip.notna().all()

    dest = ROOT / "data/processed/pip_indices.csv"
    out.to_csv(dest, index=False)
    print(f"{len(out)} distributions, {out.country_code.nunique()} countries -> {dest}")
    for ours, theirs in [
        ("gini", "gini_pip"),
        ("mld", "mld_pip"),
        ("mean", "mean_pip"),
    ]:
        d = out[ours] - out[theirs]
        print(
            f"{ours:5s} vs PIP: mean diff {d.mean():+.4f}, max |diff| {d.abs().max():.4f}, "
            f"corr {out[ours].corr(out[theirs]):.5f}"
        )


if __name__ == "__main__":
    main()
