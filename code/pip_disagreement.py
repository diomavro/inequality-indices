"""Do inequality indices agree on whether inequality rose?  PIP survey pairs.

Pairs are consecutive surveys within a PIP comparable spell (same country,
national coverage, same welfare concept, same survey_comparability code).
Each pair is classified by its Lorenz relationship on the 100-bin curves:
  dominance      one relative Lorenz curve lies weakly above the other
  single_cross   the curves cross exactly once
  multi_cross    two or more crossings
The theory predicts:
  * dominance    -> every index satisfying the transfer principle agrees
                    (Dasgupta-Sen-Starrett); VL and P90/P10 need not.
  * single_cross -> every index satisfying transfer principle + diminishing
                    transfer sensitivity agrees with the curve that is higher
                    at the bottom, provided that curve also has the lower CV
                    (Shorrocks-Foster 1987).  The Gini carries no such guarantee.

Output: data/processed/pip_pairs.csv, output/disagreement_results.json,
        paper/pip_macros.tex (every number quoted in the paper).
Run from code/:  python3 pip_disagreement.py
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import inequality_indices as ii

ROOT = Path(__file__).resolve().parent.parent
BINS = ROOT / "data/raw/pip_world_100bin.csv"
PANEL = ROOT / "data/processed/pip_indices.csv"

# Relative indices satisfying the transfer principle, one per distinct ordering:
# CV ~ GE_2, Atkinson(1) ~ MLD, Atkinson(2) ~ GE_{-1}, Atkinson(0.5) ~ GE_{0.5}
# rank distributions identically, so counting both would inflate disagreement.
TP = ["gini", "zenga", "ge_m1", "mld", "atk05", "theil", "ge2"]
# ... that are also transfer-sensitive (GE alpha < 2 and their Atkinson twins)
DTS = ["ge_m1", "mld", "atk05", "theil"]
TWINS = [("cv", "ge2"), ("atk1", "mld"), ("atk2", "ge_m1"), ("ge05", "atk05")]
NON_TP = ["vl", "p90p10"]
ABSOLUTE = ["abs_gini", "sd", "kolm"]
EKAPPA_C = [0.5, 1.0]  # kappa = c / first-survey mean
ALL = TP + [b for b, _ in TWINS] + NON_TP + ABSOLUTE
TOL = 1e-9  # Lorenz ordinate tolerance (shares)
TIE_BANDS = {
    "001": 0.001,
    "002": 0.002,
}  # robustness: ordinate gaps below this are ties
MATERIAL = 0.01  # |change in Gini| for the "material change" subsample
WELFARE_TYPES = ["income", "consumption"]
# Noise screen: an index "moves" in a pair if its |change in log| exceeds that
# index's own median |change in log| across all pairs (treats indices symmetrically).


def lorenz(y: np.ndarray) -> np.ndarray:
    """Interior Lorenz ordinates L(k/100), k = 1..99."""
    return (np.cumsum(y) / y.sum())[:-1]


def classify(k: int) -> str:
    """Lorenz relationship label from a crossing count."""
    return "dominance" if k == 0 else "single_cross" if k == 1 else "multi_cross"


def decile_class(d: np.ndarray) -> str:
    """Lorenz relationship using only the 9 decile ordinates (robustness)."""
    return classify(crossings(d[9::10]))


def crossings(d: np.ndarray, tol: float = TOL) -> int:
    """Sign changes of d = L_b - L_a, ignoring ordinates within tol of zero."""
    s = np.sign(np.where(np.abs(d) < tol, 0.0, d))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def build_pairs(panel: pd.DataFrame, curves: dict) -> pd.DataFrame:
    nat = panel[panel.reporting_level == "national"].sort_values("year")
    rows = []
    for (c, w, sc), g in nat.groupby(
        ["country_code", "welfare_type", "survey_comparability"]
    ):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            d = curves[(c, b.year, w)] - curves[(c, a.year, w)]
            k = crossings(d)
            row = {
                "country_code": c,
                "welfare_type": w,
                "spell": sc,
                "region": a.region_code,
                "year_a": a.year,
                "year_b": b.year,
                "n_cross": k,
                # +1: later curve weakly above (more equal); -1: below
                "lorenz": classify(k),
                "lorenz_decile": decile_class(d),
                **{
                    f"lorenz_tie{suffix}": classify(crossings(d, tol))
                    for suffix, tol in TIE_BANDS.items()
                },
                "dom_dir": int(np.sign(d[np.abs(d) >= TOL].sum())) if k == 0 else 0,
                # which curve is higher at the bottom (first non-zero ordinate)
                "bottom_dir": int(np.sign(d[np.abs(d) >= TOL][0]))
                if np.any(np.abs(d) >= TOL)
                else 0,
                "growth": np.log(b["mean"] / a["mean"]),
            }
            for m in ALL:
                row[f"d_{m}"] = b[m] - a[m]
            rows.append(row)
    return pd.DataFrame(rows)


def long_pairs(panel: pd.DataFrame, curves: dict, min_span: int = 10) -> pd.DataFrame:
    """First vs last survey of each comparable spell spanning >= min_span years."""
    nat = panel[panel.reporting_level == "national"].sort_values("year")
    ends = []
    for _, g in nat.groupby(["country_code", "welfare_type", "survey_comparability"]):
        if g.year.iloc[-1] - g.year.iloc[0] >= min_span:
            ends.append(g.iloc[[0, -1]])
    return build_pairs(pd.concat(ends), curves)


# Country illustrations quoted in the text: (country, welfare, spell, first, last year)
CASES = {
    "China": ("CHN", "consumption", 1, 1990, 2012),
    "Brazil": ("BRA", "income", 0, 2001, 2011),
}


def macros(R: dict) -> str:
    """LaTeX macros for every number quoted in the paper (no hand-typed figures)."""

    def pct(x: float) -> str:
        return f"{100 * x:.0f}\\%"

    m = {
        "PipPairs": f"{R['n_pairs']:,}".replace(",", "{,}"),
        "PipCountries": str(R["n_countries"]),
        "PipYearMin": str(R["year_min"]),
        "PipYearMax": str(R["year_max"]),
        "PipDistributions": f"{R['n_distributions']:,}".replace(",", "{,}"),
        "PipDistCountries": str(R["n_dist_countries"]),
        "PipGiniMaxDiff": f"{R['gini_max_abs_diff']:.4f}",
        "PipMldMaxDiff": f"{R['mld_max_abs_diff']:.3f}",
        "PipShareDom": pct(R["share_dominance"]),
        "PipNDom": str(R["n_dominance"]),
        "PipShareCross": pct(1 - R["share_dominance"]),
        "PipShareSingle": pct(R["share_single"]),
        "PipShareCrossDecile": pct(1 - R["share_dominance_decile"]),
        "PipShareCrossTieOne": pct(R["share_cross_tie001"]),
        "PipShareCrossTieTwo": pct(R["share_cross_tie002"]),
        "PipTpAgreeCrossTieOne": pct(R["tp_agree_cross_tie001"]),
        "PipTpAgreeCrossTieTwo": pct(R["tp_agree_cross_tie002"]),
        "PipShareMulti": pct(R["share_multi"]),
        "PipTpAgreeAll": pct(R["tp_agree_all"]),
        "PipTpAgreeCross": pct(R["tp_agree_cross"]),
        "PipTpDisMaterial": pct(R["tp_disagree_material"]),
        "PipNMaterial": str(R["n_material"]),
        "PipDtsAgreeCross": pct(R["dts_agree_cross"]),
        "PipTpDisMaterialCross": pct(R["tp_dis_material_cross"]),
        "PipNMaterialCross": str(R["n_material_cross"]),
        "PipTpAgreeBigCross": pct(R["tp_agree_big_cross"]),
        "PipNBigCross": f"{R['n_big_cross']:,}".replace(",", "{,}"),
        "PipNMicroPairs": f"{R['n_micro_pairs']:,}".replace(",", "{,}"),
        "PipShareCrossMicro": pct(R["share_cross_micro"]),
        "PipTpAgreeCrossMicro": pct(R["tp_agree_cross_micro"]),
        "PipAbsUpMicro": pct(R["abs_up_given_gini_down_growth_micro"]),
        "PipMedGrowth": f"{100 * R['median_abs_growth']:.1f}",
        "PipMedDlnGini": f"{100 * R['median_abs_dlngini']:.1f}",
        "PipShareGrowthExceeds": pct(R["share_growth_exceeds_dlngini"]),
        "PipKolmAgree": pct(R["kolm_agrees_abs_gini"]),
        "PipEkappaHalfAgree": pct(R["ekappa0.5_agrees_abs_gini"]),
        "PipEkappaOneAgree": pct(R["ekappa1.0_agrees_abs_gini"]),
        "PipSdAgree": pct(R["sd_agrees_abs_gini"]),
        "PipIncomeRange": f"{int(R['income_range'] // 10 * 10)}",
        "PipPNinetyViol": str(R["nontp_violations_under_dominance"]["p90p10"]),
        "PipVlViol": str(R["nontp_violations_under_dominance"]["vl"]),
        "PipDisGeTwoPNinety": pct(R["pairwise_disagreement"]["ge2|p90p10"]),
        "PipNSF": str(R["n_sf_pairs"]),
        "PipRelAbsOpp": pct(R["rel_abs_opposite"]),
        "PipRelAbsOppMaterial": pct(R["rel_abs_opp_material"]),
        "PipRelAbsOppMaterialDom": pct(R["rel_abs_opp_material_dom"]),
        "PipNMaterialDom": str(R["n_material_dom"]),
        "PipRelAbsOppMaterialCross": pct(R["rel_abs_opp_material_cross"]),
        "PipOppGiniDownAbsUp": pct(R["opp_gini_down_abs_up"]),
        "PipGiniFellGrowth": pct(R["share_gini_fell_growth"]),
        "PipAbsUpGivenGiniDownGrowth": pct(R["abs_up_given_gini_down_growth"]),
        "PipNGrowth": f"{R['n_growth']:,}".replace(",", "{,}"),
        "PipLongN": str(R["long_n"]),
        "PipLongTpAgree": pct(R["long_tp_agree"]),
        "PipLongShareDom": pct(R["long_share_dominance"]),
        "PipLongAbsUpGivenGiniDown": pct(R["long_abs_up_given_gini_down_growth"]),
        "PipLongNGiniDownGrowth": str(R["long_n_gini_down_growth"]),
        "PipDisGiniMld": pct(R["pairwise_disagreement"]["gini|mld"]),
        "PipDisGiniZenga": pct(R["pairwise_disagreement"]["gini|zenga"]),
        "PipDisGeTwoAtkTwo": pct(R["pairwise_disagreement"]["ge2|atk2"]),
        "PipDisGiniAbs": pct(R["pairwise_disagreement"]["gini|abs_gini"]),
    }
    for w in WELFARE_TYPES:
        cap = w.capitalize()
        m[f"PipShareCross{cap}"] = pct(R["by_welfare"][w]["share_cross"])
        m[f"PipTpAgreeCross{cap}"] = pct(R["by_welfare"][w]["tp_agree_cross"])
        m[f"PipRelAbsOppMaterialDom{cap}"] = pct(
            R["by_welfare"][w]["rel_abs_opp_material_dom"]
        )
    for key, (c, w, sc, y0, y1) in CASES.items():
        a, b = R["cases"][key]
        m[f"Pip{key}Years"] = f"{y0}--{y1}"
        m[f"Pip{key}GiniA"], m[f"Pip{key}GiniB"] = (
            f"{a['gini']:.3f}",
            f"{b['gini']:.3f}",
        )
        m[f"Pip{key}AbsRatio"] = f"{b['abs_gini'] / a['abs_gini']:.1f}"
        m[f"Pip{key}MeanRatio"] = f"{b['mean'] / a['mean']:.1f}"
    m["PipSFViol"] = str(sum(R["sf_violations"][k] for k in DTS))
    head = "% Generated by code/pip_disagreement.py -- do not edit by hand.\n"
    return head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def panel_lookup(panel: pd.DataFrame, p: pd.DataFrame, m: str) -> np.ndarray:
    """Level of index m in the earlier survey of each pair."""
    key = panel.set_index(["country_code", "welfare_type", "year"])[m]
    key = key[~key.index.duplicated()]
    return key.reindex(list(zip(p.country_code, p.welfare_type, p.year_a))).to_numpy()


def micro_flags(panel: pd.DataFrame, p: pd.DataFrame) -> np.ndarray:
    """True if both surveys of the pair were computed from microdata."""
    nat = panel[panel.reporting_level == "national"]
    dt = nat.set_index(["country_code", "welfare_type", "year"]).distribution_type
    a = dt.reindex(list(zip(p.country_code, p.welfare_type, p.year_a))).to_numpy()
    b = dt.reindex(list(zip(p.country_code, p.welfare_type, p.year_b))).to_numpy()
    return (a == "micro") & (b == "micro")


def sgn(x: pd.Series) -> pd.Series:
    return np.sign(x.where(x.abs() > 1e-12, 0.0))


def main() -> None:
    panel = pd.read_csv(PANEL)
    bins = pd.read_csv(BINS).query("reporting_level == 'national'")
    curves = {
        k: lorenz(g.sort_values("percentile").avg_welfare.to_numpy())
        for k, g in bins.groupby(["country_code", "year", "welfare_type"])
    }
    p = build_pairs(panel, curves)
    p.to_csv(ROOT / "data/processed/pip_pairs.csv", index=False)

    S = {m: sgn(p[f"d_{m}"]) for m in ALL}
    levels = {
        k: g.sort_values("percentile").avg_welfare.to_numpy()
        for k, g in bins.groupby(["country_code", "year", "welfare_type"])
    }
    # Kolm index (ordinal twin of E_kappa) with kappa = c / mean of the first survey,
    # the same kappa applied to both surveys of a pair.
    for c in EKAPPA_C:
        d_e = []
        for r in p.itertuples():
            ya = levels[(r.country_code, r.year_a, r.welfare_type)]
            yb = levels[(r.country_code, r.year_b, r.welfare_type)]
            kappa = c / ya.mean()
            d_e.append(ii.kolm(yb, kappa) - ii.kolm(ya, kappa))
        S[f"ekappa{c}"] = sgn(pd.Series(d_e, index=p.index))
    tp_signs = pd.concat([S[m] for m in TP], axis=1)
    p["tp_agree"] = tp_signs.nunique(axis=1).eq(1)
    dts_signs = pd.concat([S[m] for m in DTS], axis=1)
    p["dts_agree"] = dts_signs.nunique(axis=1).eq(1)
    p["rel_abs_opposite"] = S["gini"] * S["abs_gini"] < 0
    # Sanity: ordinal twins never disagree (guards the choice of TP set)
    for a, b in TWINS:
        assert (S[a] == S[b]).all(), (a, b)
    # Index-free noise screen: among TP indices whose log value moves >= 2%,
    # do they agree?  (pairs with < 2 such indices are not informative)
    dlog = {m: np.log(1 + p[f"d_{m}"] / panel_lookup(panel, p, m)).abs() for m in TP}
    big = pd.concat([dlog[m] >= dlog[m].median() for m in TP], axis=1)
    sig = tp_signs.where(big.values)
    informative = big.sum(axis=1) >= 2
    p["tp_agree_big"] = sig.nunique(axis=1).eq(1)
    # Grouped / synthetic / imputed distributions: robustness sample of micro-only pairs
    micro = pd.Series(micro_flags(panel, p), index=p.index)
    mat = p.d_gini.abs() >= MATERIAL

    dom = p.lorenz == "dominance"
    by_welfare = {
        w: {
            "share_cross": float((p.lorenz[p.welfare_type == w] != "dominance").mean()),
            "tp_agree_cross": float(p.tp_agree[(p.welfare_type == w) & ~dom].mean()),
            "rel_abs_opp_material_dom": float(
                p.rel_abs_opposite[(p.welfare_type == w) & mat & dom].mean()
            ),
        }
        for w in WELFARE_TYPES
    }
    tie_dom = {suffix: p[f"lorenz_tie{suffix}"] == "dominance" for suffix in TIE_BANDS}
    # Theory check 1: under dominance, every TP index moves opposite to dom_dir
    # (curve above = more equal = index falls), weakly.
    viol_tp = {m: int(((S[m] == p.dom_dir) & (p.dom_dir != 0) & dom).sum()) for m in TP}
    viol_non = {
        m: int(((S[m] == p.dom_dir) & (p.dom_dir != 0) & dom).sum()) for m in NON_TP
    }
    # Theorem check (Dasgupta-Sen-Starrett): under dominance every TP index must
    # move opposite to dom_dir, strictly.  Fail loudly if the data or code disagree.
    assert (p.dom_dir[dom] != 0).all()
    for m in TP + [b for b, _ in TWINS]:
        assert (S[m][dom] == -p.dom_dir[dom]).all(), m
    # Theory check 2 (Shorrocks-Foster): single crossing, the curve higher at the
    # bottom also has weakly lower CV -> every DTS index says it is more equal.
    sc = p.lorenz == "single_cross"
    sf = sc & (S["cv"] == -p.bottom_dir)  # later higher at bottom & CV fell, or v.v.
    sf_viol = {
        m: int(((S[m] == p.bottom_dir) & sf).sum())
        for m in DTS + ["gini", "zenga", "ge2"]
    }

    pair_dis = {
        f"{a}|{b}": float((S[a] * S[b] < 0).mean())
        for a, b in combinations(
            [
                "gini",
                "mld",
                "theil",
                "ge2",
                "atk2",
                "zenga",
                "vl",
                "p90p10",
                "abs_gini",
            ],
            2,
        )
    }

    grow = p.growth > 0
    gd_grow = grow & (S["gini"] < 0)
    L = long_pairs(panel, curves)
    LS = {m: sgn(L[f"d_{m}"]) for m in ALL}
    l_tp = pd.concat([LS[m] for m in TP], axis=1).nunique(axis=1).eq(1)
    l_gd = (L.growth > 0) & (LS["gini"] < 0)
    R = {
        "n_distributions": len(panel),
        "n_dist_countries": int(panel.country_code.nunique()),
        "gini_max_abs_diff": float((panel.gini - panel.gini_pip).abs().max()),
        "mld_max_abs_diff": float((panel.mld - panel.mld_pip).abs().max()),
        "n_growth": int(grow.sum()),
        "abs_up_given_gini_down_growth": float((S["abs_gini"] > 0)[gd_grow].mean()),
        "long_n": len(L),
        "long_tp_agree": float(l_tp.mean()),
        "long_share_dominance": float((L.lorenz == "dominance").mean()),
        "long_n_gini_down_growth": int(l_gd.sum()),
        "long_abs_up_given_gini_down_growth": float((LS["abs_gini"] > 0)[l_gd].mean()),
        "n_pairs": len(p),
        "n_countries": int(p.country_code.nunique()),
        "year_min": int(panel.year.min()),
        "year_max": int(panel.year.max()),
        "share_dominance": float(dom.mean()),
        "share_single": float(sc.mean()),
        "share_dominance_decile": float((p.lorenz_decile == "dominance").mean()),
        **{
            f"share_cross_tie{suffix}": float((~tie_dom[suffix]).mean())
            for suffix in TIE_BANDS
        },
        **{
            f"tp_agree_cross_tie{suffix}": float(p.tp_agree[~tie_dom[suffix]].mean())
            for suffix in TIE_BANDS
        },
        "share_multi": float((p.lorenz == "multi_cross").mean()),
        "tp_agree_all": float(p.tp_agree.mean()),
        "tp_agree_cross": float(p.tp_agree[~dom].mean()),
        "tp_disagree_material": float((~p.tp_agree[mat]).mean()),
        "n_material": int(mat.sum()),
        "dts_agree_cross": float(p.dts_agree[~dom].mean()),
        "tp_dis_material_cross": float((~p.tp_agree[mat & ~dom]).mean()),
        "n_material_cross": int((mat & ~dom).sum()),
        "tp_agree_big_cross": float(p.tp_agree_big[informative & ~dom].mean()),
        "n_big_cross": int((informative & ~dom).sum()),
        "n_micro_pairs": int(micro.sum()),
        "share_cross_micro": float((p.lorenz[micro] != "dominance").mean()),
        "tp_agree_cross_micro": float(p.tp_agree[micro & ~dom].mean()),
        "abs_up_given_gini_down_growth_micro": float(
            (S["abs_gini"] > 0)[gd_grow & micro].mean()
        ),
        "vl_violations": int(((S["vl"] == p.dom_dir) & (p.dom_dir != 0) & dom).sum()),
        # Growth arithmetic: dln(mu G) = dln(mu) + dln(G)
        "median_abs_growth": float(p.growth.abs().median()),
        "median_abs_dlngini": float(
            np.log(1 + p.d_gini / panel_lookup(panel, p, "gini")).abs().median()
        ),
        "rel_abs_opp_material_dom": float(p.rel_abs_opposite[mat & dom].mean()),
        "n_material_dom": int((mat & dom).sum()),
        "rel_abs_opp_material_cross": float(p.rel_abs_opposite[mat & ~dom].mean()),
        "share_growth_exceeds_dlngini_material": float(
            (
                p.growth.abs()
                > np.log(1 + p.d_gini / panel_lookup(panel, p, "gini")).abs()
            )[mat].mean()
        ),
        "share_growth_exceeds_dlngini": float(
            (
                p.growth.abs()
                > np.log(1 + p.d_gini / panel_lookup(panel, p, "gini")).abs()
            ).mean()
        ),
        "kolm_agrees_abs_gini": float((S["kolm"] * S["abs_gini"] > 0).mean()),
        **{
            f"ekappa{c}_agrees_abs_gini": float(
                (S[f"ekappa{c}"] * S["abs_gini"] > 0).mean()
            )
            for c in EKAPPA_C
        },
        "sd_agrees_abs_gini": float((S["sd"] * S["abs_gini"] > 0).mean()),
        "income_range": float(
            panel.groupby("country_code")["mean"]
            .mean()
            .pipe(lambda r: r.max() / r.min())
        ),
        "tp_violations_under_dominance": viol_tp,
        "nontp_violations_under_dominance": viol_non,
        "n_dominance": int(dom.sum()),
        "by_welfare": by_welfare,
        "n_sf_pairs": int(sf.sum()),
        "sf_violations": sf_viol,
        "rel_abs_opposite": float(p.rel_abs_opposite.mean()),
        "rel_abs_opp_material": float(p.rel_abs_opposite[mat].mean()),
        "opp_gini_down_abs_up": float(
            ((S["gini"] < 0) & (S["abs_gini"] > 0))[p.rel_abs_opposite].mean()
        ),
        "rel_abs_opposite_growth": float(p.rel_abs_opposite[grow].mean()),
        "share_rel_down_abs_up_growth": float(
            ((S["gini"] < 0) & (S["abs_gini"] > 0))[grow].mean()
        ),
        "share_gini_fell_growth": float((S["gini"] < 0)[grow].mean()),
        "pairwise_disagreement": pair_dis,
        "cases": {
            k: [
                panel.query(
                    "country_code == @c and welfare_type == @w and "
                    "survey_comparability == @sc and year == @yr and "
                    "reporting_level == 'national'"
                )
                .iloc[0][["gini", "abs_gini", "mean"]]
                .astype(float)
                .to_dict()
                for yr in (y0, y1)
            ]
            for k, (c, w, sc, y0, y1) in CASES.items()
        },
    }
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output/disagreement_results.json").write_text(json.dumps(R, indent=2))
    (ROOT / "paper/pip_macros.tex").write_text(macros(R))
    print(
        json.dumps(
            {k: v for k, v in R.items() if k != "pairwise_disagreement"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
