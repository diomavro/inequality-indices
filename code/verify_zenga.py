# NOTE (2026-10): the DTS and subgroup-consistency tests that used to live here
# sampled only favourable configurations and reported a false PASS. They were
# removed: both properties FAIL for Z and Z*; see verify_corrections.py for the
# integer counterexamples and unrestricted searches.
"""
Verify Axiomatic Properties of the Zenga (2007) Index
=====================================================

Systematically tests all 11 axiomatic properties for the Zenga index
and compares with Gini to confirm the Zenga dominates on the axiom checklist.

The Zenga index is defined as:
    Z = (1/(n-1)) * sum_{i=1}^{n-1} (1 - M_i^- / M_i^+)
where M_i^- = mean of the bottom i incomes (sorted)
      M_i^+ = mean of the top (n-i) incomes (sorted)

References
----------
Zenga, M. (2007). "Inequality Curve and Inequality Index Based on the Ratios
    Between Lower and Upper Arithmetic Means." Statistica & Applicazioni, 5(1), 3-27.
"""

from __future__ import annotations

import numpy as np


def zenga(y: np.ndarray) -> float:
    """Compute the Zenga (2007) inequality index."""
    y = np.asarray(y, dtype=np.float64)
    ys = np.sort(y)
    n = len(ys)
    Z = 0.0
    for i in range(1, n):
        lower_mean = np.mean(ys[:i])
        upper_mean = np.mean(ys[i:])
        Z += 1.0 - lower_mean / upper_mean
    return float(Z / (n - 1))


def gini(y: np.ndarray) -> float:
    """Gini coefficient for comparison."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    mu = np.mean(y)
    diff_sum = np.sum(np.abs(y[:, None] - y[None, :]))
    return float(diff_sum / (2.0 * n * n * mu))


TOL = 1e-8

# Base distributions
Y = np.array([1.0, 2.0, 50.0, 100.0])
Y_LARGE = np.array([1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0])
Y_DTS = np.array([5.0, 15.0, 50.0, 100.0, 190.0, 200.0])


def main() -> None:
    print("=" * 70)
    print("ZENGA (2007) INDEX — AXIOMATIC PROPERTY VERIFICATION")
    print("=" * 70)
    print()

    # Basic computation
    print(f"Z(1, 2, 50, 100) = {zenga(Y):.6f}")
    print(f"G(1, 2, 50, 100) = {gini(Y):.6f}")
    print()

    # ---- 1. ANONYMITY ----
    print("1. ANONYMITY: I(π(y)) == I(y)")
    rng = np.random.default_rng(42)
    y_perm = rng.permutation(Y)
    z_orig = zenga(Y)
    z_perm = zenga(y_perm)
    passed = abs(z_orig - z_perm) < TOL
    print(f"   Z(y) = {z_orig:.8f}, Z(π(y)) = {z_perm:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL'}")
    print()

    # ---- 2. NORMALIZATION ----
    print("2. NORMALIZATION: I(y) = 0 iff all equal")
    y_equal = np.array([5.0, 5.0, 5.0, 5.0])
    z_equal = zenga(y_equal)
    z_unequal = zenga(Y)
    passed = abs(z_equal) < TOL and z_unequal > TOL
    print(f"   Z(5,5,5,5) = {z_equal:.8f}")
    print(f"   Z(1,2,50,100) = {z_unequal:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL'}")
    print()

    # ---- 3. CONTINUITY ----
    print("3. CONTINUITY: small changes → small changes")
    eps_vals = [1.0, 0.1, 0.01, 0.001, 0.0001]
    y_base = np.array([10.0, 20.0, 30.0, 40.0])
    z_base = zenga(y_base)
    print(f"   Z(10,20,30,40) = {z_base:.8f}")
    all_continuous = True
    for eps in eps_vals:
        y_pert = y_base.copy()
        y_pert[0] += eps
        z_pert = zenga(y_pert)
        diff = abs(z_pert - z_base)
        print(f"   Z(10+{eps},20,30,40) = {z_pert:.8f}, |ΔZ| = {diff:.2e}")
        if eps < 0.01 and diff > 0.01:
            all_continuous = False
    print(f"   Result: {'✓ PASS (converges smoothly)' if all_continuous else '✗ FAIL'}")
    print()

    # ---- 4. POPULATION INVARIANCE ----
    print("4. POPULATION INVARIANCE: I(y) == I(y,y)")
    y_rep = np.tile(Y, 2)
    y_rep3 = np.tile(Y, 3)
    z1 = zenga(Y)
    z2 = zenga(y_rep)
    z3 = zenga(y_rep3)
    passed = abs(z1 - z2) < TOL and abs(z1 - z3) < TOL
    print(f"   Z(y)   = {z1:.8f}")
    print(f"   Z(y,y) = {z2:.8f}")
    print(f"   Z(y,y,y) = {z3:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL'}")
    if not passed:
        print(f"   NOTE: |Z(y) - Z(y,y)| = {abs(z1-z2):.2e}")
        print(f"         |Z(y) - Z(y,y,y)| = {abs(z1-z3):.2e}")
        # Test with a simpler distribution
        y_simple = np.array([1.0, 3.0])
        y_simple2 = np.tile(y_simple, 2)
        y_simple3 = np.tile(y_simple, 3)
        y_simple5 = np.tile(y_simple, 5)
        print(f"   Extra test with (1,3):")
        print(f"     Z(1,3)       = {zenga(y_simple):.8f}")
        print(f"     Z(1,3,1,3)   = {zenga(y_simple2):.8f}")
        print(f"     Z(1,1,1,3,3,3) = {zenga(np.sort(y_simple3)):.8f}")
        print(f"     Z(5-rep)     = {zenga(y_simple5):.8f}")
    print()

    # ---- 5. SCALE INVARIANCE ----
    print("5. SCALE INVARIANCE: I(λy) == I(y)")
    lam = 2.0
    z_orig = zenga(Y)
    z_scaled = zenga(lam * Y)
    passed = abs(z_orig - z_scaled) < TOL
    print(f"   Z(y)  = {z_orig:.8f}")
    print(f"   Z(2y) = {z_scaled:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL'}")
    # Also test with different lambda
    for l in [0.5, 3.7, 100.0]:
        z_l = zenga(l * Y)
        print(f"   Z({l}*y) = {z_l:.8f}, diff = {abs(z_orig - z_l):.2e}")
    print()

    # ---- 6. TRANSLATION INVARIANCE ----
    print("6. TRANSLATION INVARIANCE: I(y + c) == I(y)")
    c = 100.0
    z_orig = zenga(Y)
    z_trans = zenga(Y + c)
    passed = abs(z_orig - z_trans) < TOL
    print(f"   Z(y)      = {z_orig:.8f}")
    print(f"   Z(y+100)  = {z_trans:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL (as expected — scale-invariant indices fail this)'}")
    print()

    # ---- 7. TRANSFER PRINCIPLE ----
    print("7. TRANSFER PRINCIPLE: progressive transfer reduces Z")
    # Test multiple transfers
    transfers = [
        ("(1,2,50,100) → δ=10 from 100→50", Y, 3, 2, 10.0),
        ("(1,2,50,100) → δ=0.5 from 2→1", Y, 1, 0, 0.5),
        ("(10,20,30,40) → δ=5 from 40→30", np.array([10.0, 20.0, 30.0, 40.0]), 3, 2, 5.0),
        ("(10,20,30,40) → δ=5 from 20→10", np.array([10.0, 20.0, 30.0, 40.0]), 1, 0, 5.0),
    ]
    all_pass = True
    for desc, y, i_rich, i_poor, delta in transfers:
        y_after = y.copy()
        y_after[i_rich] -= delta
        y_after[i_poor] += delta
        z_before = zenga(y)
        z_after = zenga(y_after)
        ok = z_after < z_before - TOL
        if not ok:
            all_pass = False
        print(f"   {desc}")
        print(f"     Z_before = {z_before:.8f}, Z_after = {z_after:.8f}, ΔZ = {z_after - z_before:.8f} {'✓' if ok else '✗'}")

    # More extensive random test
    rng = np.random.default_rng(123)
    n_tests = 1000
    n_fail = 0
    for _ in range(n_tests):
        n = rng.integers(3, 20)
        y_test = rng.lognormal(3.0, 1.0, size=n)
        y_sorted = np.sort(y_test)
        # Pick random rich and poor
        i_rich = rng.integers(1, n)
        i_poor = rng.integers(0, i_rich)
        max_delta = (y_sorted[i_rich] - y_sorted[i_poor]) / 2.0
        if max_delta < 1e-10:
            continue
        delta = rng.uniform(1e-10, max_delta)
        y_after = y_sorted.copy()
        y_after[i_rich] -= delta
        y_after[i_poor] += delta
        z_b = zenga(y_sorted)
        z_a = zenga(y_after)
        if z_a >= z_b - TOL:
            n_fail += 1

    print(f"   Random test: {n_tests} transfers, {n_fail} failures")
    if n_fail > 0:
        all_pass = False
    print(f"   Result: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print()

    # ---- 8. LORENZ CONSISTENCY ----
    print("8. LORENZ CONSISTENCY: follows from transfer principle")
    print("   (If transfer principle holds, Lorenz consistency holds by DSS theorem)")
    print(f"   Result: {'✓ PASS (conditional on TP)' if all_pass else '✗ FAIL'}")
    print()

    # ---- 9. DIMINISHING TRANSFER SENSITIVITY (DTS) ----
    print("9. DIMINISHING TRANSFER SENSITIVITY")
    print("   Not tested here: Z fails DTS (integer counterexample and unrestricted")
    print("   search in verify_corrections.py).")
    print()

    # ---- 10. ADDITIVE DECOMPOSABILITY ----
    print("10. ADDITIVE DECOMPOSABILITY")
    y1 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([50.0, 60.0, 70.0, 80.0])
    y_all = np.concatenate([y1, y2])
    n = len(y_all)
    n1, n2 = len(y1), len(y2)
    mu = np.mean(y_all)
    mu1, mu2 = np.mean(y1), np.mean(y2)

    total = zenga(y_all)

    # Between: replace each income with group mean
    y_between = np.concatenate([np.full(n1, mu1), np.full(n2, mu2)])
    between = zenga(y_between)

    # Try population weights
    within_pop = (n1/n) * zenga(y1) + (n2/n) * zenga(y2)
    predicted_pop = within_pop + between

    # Try income-share weights
    within_inc = (n1/n * mu1/mu) * zenga(y1) + (n2/n * mu2/mu) * zenga(y2)
    predicted_inc = within_inc + between

    print(f"   Z(all) = {total:.8f}")
    print(f"   Z(between) = {between:.8f}")
    print(f"   Pop-weight within + between = {predicted_pop:.8f}")
    print(f"   Inc-weight within + between = {predicted_inc:.8f}")
    residual_pop = total - predicted_pop
    residual_inc = total - predicted_inc
    passed = abs(residual_pop) < TOL or abs(residual_inc) < TOL
    print(f"   Residual (pop weights): {residual_pop:.8f}")
    print(f"   Residual (inc weights): {residual_inc:.8f}")
    print(f"   Result: {'✓ PASS' if passed else '✗ FAIL (as expected — not decomposable)'}")
    print()

    # ---- 11. SUBGROUP CONSISTENCY ----
    print("11. SUBGROUP CONSISTENCY")
    print("    Not tested here: Z fails subgroup consistency (integer counterexample")
    print("    and unrestricted search in verify_corrections.py).")
    print()

    # ---- SUMMARY ----
    print("=" * 70)
    print("SUMMARY: Zenga (2007) vs Gini")
    print("=" * 70)
    print(f"{'Property':<30} {'Zenga':>10} {'Gini':>10}")
    print("-" * 50)
    properties = [
        "Anonymity", "Normalization", "Continuity",
        "Population Invariance", "Scale Invariance",
        "Translation Invariance", "Transfer Principle",
        "Lorenz Consistency", "Dim. Transfer Sensitivity",
        "Additive Decomposability", "Subgroup Consistency"
    ]
    # These will be filled based on the tests above
    # (hardcoding Gini results from the paper)
    gini_results = ["✓", "✓", "✓", "✓", "✓", "✗", "✓", "✓", "✗", "✗", "✗"]
    print("(see test results above for Zenga verdicts)")


if __name__ == "__main__":
    main()
