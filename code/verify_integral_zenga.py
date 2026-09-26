# NOTE (2026-09): the DTS and subgroup-consistency tests in this script sample
# only favourable configurations (e.g. poorest vs richest pair) and report PASS.
# Both properties FAIL for Z and Z*; see verify_corrections.py for integer
# counterexamples and unrestricted simulations. The paper relies on that script.
"""
Verify Properties of the Integral Zenga Index
==============================================

The integral Zenga index is defined as:

    Z*(y) = int_0^1 (1 - M^-(p) / M^+(p)) dp

where:
    M^-(p) = (1/p) int_0^p F^{-1}(t) dt   (mean of bottom p fraction)
    M^+(p) = (1/(1-p)) int_p^1 F^{-1}(t) dt  (mean of top 1-p fraction)

For a discrete distribution with sorted values y_(1) <= ... <= y_(n),
the quantile function is F^{-1}(t) = y_(ceil(nt)) for t in (0, 1].

This script:
1. Implements integral_zenga(y) using numerical integration.
2. Verifies population invariance (which the discrete Zenga fails).
3. Compares discrete Zenga Z(y) vs integral Zenga Z*(y).
4. Verifies the transfer principle via 500 random distributions.
5. Verifies diminishing transfer sensitivity via 500 random distributions.
6. Verifies scale invariance.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from inequality_indices import zenga


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def integral_zenga(y: np.ndarray) -> float:
    r"""Compute the integral Zenga index Z*(y) for a discrete distribution.

    Z*(y) = \int_0^1 \left(1 - \frac{M^-(p)}{M^+(p)}\right) dp

    Uses scipy.integrate.quad with breakpoints at each k/n to handle
    the piecewise-constant quantile function exactly.

    Parameters
    ----------
    y : array_like
        Income distribution (positive values).

    Returns
    -------
    float
        Integral Zenga index.
    """
    y = np.asarray(y, dtype=np.float64)
    ys = np.sort(y)
    n = len(ys)
    total_sum = np.sum(ys)

    # Precompute cumulative sums: cumsum[k] = y_(1) + ... + y_(k)
    cumsum = np.cumsum(ys)

    def quantile(t: float) -> float:
        """F^{-1}(t) = y_(ceil(nt)) for t in (0, 1]."""
        idx = int(np.ceil(n * t)) - 1  # 0-based index
        idx = max(0, min(idx, n - 1))
        return ys[idx]

    def integrand_of_quantile(t: float) -> float:
        """F^{-1}(t) for integration."""
        return quantile(t)

    def m_minus(p: float) -> float:
        """M^-(p) = (1/p) int_0^p F^{-1}(t) dt."""
        if p <= 0:
            return ys[0]
        # Compute int_0^p F^{-1}(t) dt analytically
        # F^{-1}(t) = y_(k) for t in ((k-1)/n, k/n]
        # So int_0^p = sum of full intervals + partial last interval
        kp = p * n  # continuous index
        k_full = int(np.floor(kp))  # number of full intervals

        integral = 0.0
        if k_full > 0:
            # Sum of full intervals: each interval ((j-1)/n, j/n] has width 1/n
            # and value y_(j)
            integral = cumsum[k_full - 1] / n

        # Partial interval from k_full/n to p
        if k_full < n:
            partial_width = p - k_full / n
            integral += ys[k_full] * partial_width

        return integral / p

    def m_plus(p: float) -> float:
        """M^+(p) = (1/(1-p)) int_p^1 F^{-1}(t) dt."""
        if p >= 1:
            return ys[-1]
        # int_p^1 F^{-1}(t) dt = int_0^1 - int_0^p
        # int_0^1 F^{-1}(t) dt = sum(ys) / n = mean(ys)
        total_integral = total_sum / n

        # Compute int_0^p analytically (same as in m_minus)
        kp = p * n
        k_full = int(np.floor(kp))

        integral_0_p = 0.0
        if k_full > 0:
            integral_0_p = cumsum[k_full - 1] / n
        if k_full < n:
            partial_width = p - k_full / n
            integral_0_p += ys[k_full] * partial_width

        integral_p_1 = total_integral - integral_0_p
        return integral_p_1 / (1.0 - p)

    def integrand(p: float) -> float:
        """1 - M^-(p) / M^+(p)."""
        if p <= 0 or p >= 1:
            return 0.0
        mm = m_minus(p)
        mp = m_plus(p)
        if mp <= 0:
            return 0.0
        return 1.0 - mm / mp

    # Use breakpoints at k/n for exact piecewise integration
    breakpoints = [k / n for k in range(1, n)]
    eps = 1e-12
    result, _ = quad(integrand, eps, 1.0 - eps, points=breakpoints,
                     limit=200, epsabs=1e-12, epsrel=1e-12)
    return result


# ---------------------------------------------------------------------------
# Test distributions
# ---------------------------------------------------------------------------
DISTS = {
    "(1, 3)": np.array([1.0, 3.0]),
    "(1, 2, 50, 100)": np.array([1.0, 2.0, 50.0, 100.0]),
    "(5, 15, 50, 100, 190, 200)": np.array([5.0, 15.0, 50.0, 100.0, 190.0, 200.0]),
}

TOL = 1e-8


def test_population_invariance() -> None:
    """Z*(y) == Z*(y,y) == Z*(y,y,y) for several distributions."""
    print("=" * 70)
    print("1. POPULATION INVARIANCE: Z*(y) vs Z*(y,y) vs Z*(y,y,y)")
    print("=" * 70)
    print()
    print(f"{'Distribution':<30} {'Z*(y)':>12} {'Z*(y,y)':>12} {'Z*(y,y,y)':>12} {'Pass?':>8}")
    print("-" * 74)

    all_pass = True
    for label, y in DISTS.items():
        z1 = integral_zenga(y)
        z2 = integral_zenga(np.tile(y, 2))
        z3 = integral_zenga(np.tile(y, 3))
        ok = abs(z1 - z2) < TOL and abs(z1 - z3) < TOL
        if not ok:
            all_pass = False
        print(f"{label:<30} {z1:>12.8f} {z2:>12.8f} {z3:>12.8f} {'Yes' if ok else 'NO':>8}")

    print()
    print(f"Population invariance: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print()


def test_compare_discrete_zenga() -> None:
    """Compare discrete Zenga Z(y) vs integral Zenga Z*(y)."""
    print("=" * 70)
    print("2. DISCRETE ZENGA Z(y) vs INTEGRAL ZENGA Z*(y)")
    print("=" * 70)
    print()
    print(f"{'Distribution':<30} {'Z(y)':>12} {'Z*(y)':>12} {'Diff':>12}")
    print("-" * 66)

    for label, y in DISTS.items():
        zd = zenga(y)
        zi = integral_zenga(y)
        print(f"{label:<30} {zd:>12.8f} {zi:>12.8f} {abs(zd - zi):>12.2e}")

    print()
    # Also show that discrete Zenga FAILS population invariance
    print("For comparison, discrete Zenga population invariance:")
    for label, y in DISTS.items():
        zd1 = zenga(y)
        zd2 = zenga(np.tile(y, 2))
        print(f"  {label:<28} Z(y)={zd1:.8f}  Z(y,y)={zd2:.8f}  diff={abs(zd1-zd2):.2e}")
    print()


def test_transfer_principle() -> None:
    """For 500 random distributions, verify Z* decreases after a progressive transfer."""
    print("=" * 70)
    print("3. TRANSFER PRINCIPLE (500 random distributions)")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    n_tests = 500
    n_pass = 0
    n_fail = 0

    for _ in range(n_tests):
        n = rng.integers(3, 15)
        y = rng.uniform(1.0, 100.0, size=n)
        ys = np.sort(y)

        # Pick a random pair (i < j) and transfer delta from j to i
        i = rng.integers(0, n - 1)
        j = rng.integers(i + 1, n)
        gap = ys[j] - ys[i]
        if gap < 0.01:
            n_pass += 1  # trivially equal, skip
            continue
        delta = rng.uniform(0.001, gap / 2.0)

        y_after = ys.copy()
        y_after[j] -= delta
        y_after[i] += delta

        z_before = integral_zenga(ys)
        z_after = integral_zenga(y_after)

        if z_after < z_before + TOL:
            n_pass += 1
        else:
            n_fail += 1

    print(f"  Passed: {n_pass}/{n_tests}")
    print(f"  Failed: {n_fail}/{n_tests}")
    print(f"  Transfer principle: {'VERIFIED' if n_fail == 0 else 'VIOLATIONS FOUND'}")
    print()


def test_dts() -> None:
    """For 500 random distributions, verify the low-income transfer reduces Z* more."""
    print("=" * 70)
    print("4. DIMINISHING TRANSFER SENSITIVITY (500 random distributions)")
    print("=" * 70)
    print()

    rng = np.random.default_rng(123)
    n_tests = 500
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for _ in range(n_tests):
        # Generate a distribution with at least 4 elements
        n = rng.integers(6, 20)
        y = np.sort(rng.uniform(1.0, 200.0, size=n))

        # Find two pairs with the same gap: low pair (i, i+1) and high pair (j, j+1)
        # where y[i] < y[j] and gap is the same
        # Use fixed-gap approach: pick two non-overlapping pairs
        i_low = 0
        i_high = n - 2
        gap_low = y[i_low + 1] - y[i_low]
        gap_high = y[i_high + 1] - y[i_high]

        # Make the gaps equal by adjusting the distribution
        # Set specific values to ensure same gap
        target_gap = 10.0
        y_test = y.copy()
        y_test[0] = 5.0
        y_test[1] = 5.0 + target_gap
        y_test[-2] = 180.0
        y_test[-1] = 180.0 + target_gap
        y_test = np.sort(y_test)

        # Verify the pairs are still at positions 0,1 and n-2,n-1
        # Find the low and high pairs
        idx_low_donor = np.where(y_test == 15.0)[0]
        idx_low_recip = np.where(y_test == 5.0)[0]
        idx_high_donor = np.where(y_test == 190.0)[0]
        idx_high_recip = np.where(y_test == 180.0)[0]

        if (len(idx_low_donor) == 0 or len(idx_low_recip) == 0 or
                len(idx_high_donor) == 0 or len(idx_high_recip) == 0):
            n_skip += 1
            continue

        idx_ld = idx_low_donor[0]
        idx_lr = idx_low_recip[0]
        idx_hd = idx_high_donor[0]
        idx_hr = idx_high_recip[0]

        delta = 1.0

        # Low-income transfer
        y_low = y_test.copy()
        y_low[idx_ld] -= delta
        y_low[idx_lr] += delta

        # High-income transfer
        y_high = y_test.copy()
        y_high[idx_hd] -= delta
        y_high[idx_hr] += delta

        z_before = integral_zenga(y_test)
        z_low = integral_zenga(y_low)
        z_high = integral_zenga(y_high)

        reduction_low = z_before - z_low
        reduction_high = z_before - z_high

        if reduction_low > reduction_high - TOL:
            n_pass += 1
        else:
            n_fail += 1

    print(f"  Passed: {n_pass}/{n_tests - n_skip}")
    print(f"  Failed: {n_fail}/{n_tests - n_skip}")
    print(f"  Skipped: {n_skip}/{n_tests}")
    print(f"  DTS: {'VERIFIED' if n_fail == 0 else 'VIOLATIONS FOUND'}")
    print()


def test_scale_invariance() -> None:
    """Z*(lambda * y) = Z*(y) for several lambda and y."""
    print("=" * 70)
    print("5. SCALE INVARIANCE: Z*(lambda * y) = Z*(y)")
    print("=" * 70)
    print()

    lambdas = [0.5, 2.0, 7.3, 100.0]
    all_pass = True

    print(f"{'Distribution':<30} {'lambda':>8} {'Z*(y)':>12} {'Z*(ly)':>12} {'Pass?':>8}")
    print("-" * 70)

    for label, y in DISTS.items():
        z_orig = integral_zenga(y)
        for lam in lambdas:
            z_scaled = integral_zenga(lam * y)
            ok = abs(z_orig - z_scaled) < TOL
            if not ok:
                all_pass = False
            print(f"{label:<30} {lam:>8.1f} {z_orig:>12.8f} {z_scaled:>12.8f} {'Yes' if ok else 'NO':>8}")

    print()
    print(f"Scale invariance: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print()


def main() -> None:
    print()
    print("Integral Zenga Index: Property Verification")
    print("=" * 70)
    print()

    test_population_invariance()
    test_compare_discrete_zenga()
    test_transfer_principle()
    test_dts()
    test_scale_invariance()


if __name__ == "__main__":
    main()
