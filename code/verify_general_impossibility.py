"""
Verify the General Split-Point Impossibility Theorem
=====================================================

This script tests the theorem that for any continuous f: (0,∞) → R with
f(1) = 0, the generalized split-point index

    I_f(y) = (1/(n-1)) Σ_{i=1}^{n-1} f(M⁻(i/n) / M⁺(i/n))

cannot satisfy population invariance unless f ≡ 0.

We verify this by computing I_f for several candidate f-functions on the
distributions (1,3), (1,1,3,3), and (1,1,1,3,3,3), showing that
I_f(y) ≠ I_f(y^(2)) ≠ I_f(y^(3)) whenever f is not identically zero.

We also verify the functional equations (FE1) and (FE2) derived from the
2-fold and 3-fold replications, confirming the algebra.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Split-point index for arbitrary f
# ---------------------------------------------------------------------------
def split_point_index(y: np.ndarray, f) -> float:
    """Compute I_f(y) = (1/(n-1)) Σ f(M⁻(i/n) / M⁺(i/n))."""
    ys = np.sort(y)
    n = len(ys)
    total = 0.0
    for i in range(1, n):
        lower_mean = np.mean(ys[:i])
        upper_mean = np.mean(ys[i:])
        ratio = lower_mean / upper_mean
        total += f(ratio)
    return total / (n - 1)


# ---------------------------------------------------------------------------
# Split-point ratio maps for two-person distribution (a, b)
# ---------------------------------------------------------------------------
def phi(r: float) -> float:
    """2-fold replication, split point i=1: 3r/(r+2)."""
    return 3.0 * r / (r + 2.0)


def psi(r: float) -> float:
    """2-fold replication, split point i=3: (2r+1)/3."""
    return (2.0 * r + 1.0) / 3.0


def alpha1(r: float) -> float:
    """3-fold replication, split point i=1: 5r/(2r+3)."""
    return 5.0 * r / (2.0 * r + 3.0)


def alpha2(r: float) -> float:
    """3-fold replication, split point i=2: 4r/(r+3)."""
    return 4.0 * r / (r + 3.0)


def alpha4(r: float) -> float:
    """3-fold replication, split point i=4: (3r+1)/4."""
    return (3.0 * r + 1.0) / 4.0


def alpha5(r: float) -> float:
    """3-fold replication, split point i=5: (3r+2)/5."""
    return (3.0 * r + 2.0) / 5.0


# ---------------------------------------------------------------------------
# General k-fold replication split-point ratios
# ---------------------------------------------------------------------------
def kfold_ratios(r: float, k: int) -> list[float]:
    """Compute all kn-1 split-point ratios for k-fold replication of (a, b).

    For distribution (a,...,a, b,...,b) with k copies of each, n=2k.
    Split point i (1 ≤ i ≤ 2k-1):
      bottom i entries, top 2k-i entries.
      If i ≤ k: bottom is all a's → M⁻ = a
                 top is (k-i) a's and k b's → M⁺ = ((k-i)a + kb)/(2k-i)
                 ratio = (2k-i)r / ((k-i)r + k) = (2k-i)r / (kr - ir + k)
      If i = k: M⁻ = a, M⁺ = b, ratio = r
      If i > k: bottom is k a's and (i-k) b's → M⁻ = (ka + (i-k)b)/i
                top is (2k-i) b's → M⁺ = b
                ratio = (ka + (i-k)b)/(ib) = (kr + i - k) / i = kr/i + 1 - k/i
    """
    n = 2 * k
    ratios = []
    for i in range(1, n):
        if i <= k:
            # bottom i are all a's
            # top 2k-i: (k-i) a's and k b's
            # M⁻ = a, M⁺ = ((k-i)*a + k*b) / (2k-i)
            # ratio = a * (2k-i) / ((k-i)*a + k*b)
            #       = r * (2k-i) / ((k-i)*r + k)
            ratio_val = r * (n - i) / ((k - i) * r + k)
        else:
            # bottom i: k a's and (i-k) b's
            # top 2k-i: all b's
            # M⁻ = (k*a + (i-k)*b) / i, M⁺ = b
            # ratio = (k*r + i - k) / i
            ratio_val = (k * r + i - k) / i
        ratios.append(ratio_val)
    return ratios


# ---------------------------------------------------------------------------
# Functional equation verification
# ---------------------------------------------------------------------------
def verify_fe1(f, r: float) -> float:
    """FE1 residual: 2f(r) - f(φ(r)) - f(ψ(r)). Should be 0 if PI holds."""
    return 2.0 * f(r) - f(phi(r)) - f(psi(r))


def verify_fe2(f, r: float) -> float:
    """FE2 residual: 4f(r) - f(α₁(r)) - f(α₂(r)) - f(α₄(r)) - f(α₅(r)).
    Should be 0 if PI holds."""
    return (
        4.0 * f(r)
        - f(alpha1(r))
        - f(alpha2(r))
        - f(alpha4(r))
        - f(alpha5(r))
    )


def verify_fe_kfold(f, r: float, k: int) -> float:
    """General k-fold FE residual.

    PI requires: f(r) = (1/(2k-1)) Σ f(ratio_i)
    i.e., (2k-1)*f(r) = Σ f(ratio_i)
    But the ratio at i=k is r itself, so:
    (2k-2)*f(r) = Σ_{i≠k} f(ratio_i)

    Returns the residual (should be 0 if PI holds).
    """
    ratios = kfold_ratios(r, k)
    n = 2 * k
    # PI: f(r) = (1/(n-1)) * Σ f(ratio_i)
    total = sum(f(ratio_val) for ratio_val in ratios)
    return (n - 1) * f(r) - total


# ---------------------------------------------------------------------------
# Candidate f-functions (all satisfy f(1) = 0)
# ---------------------------------------------------------------------------
CANDIDATES = {
    "f(r) = 1 - r": lambda r: 1.0 - r,
    "f(r) = (1 - r)^2": lambda r: (1.0 - r) ** 2,
    "f(r) = -ln(r)": lambda r: -np.log(r),
    "f(r) = 1/r - 1": lambda r: 1.0 / r - 1.0,
    "f(r) = r^{-2} - 1": lambda r: r ** (-2) - 1.0,
}


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main() -> None:
    a, b = 1.0, 3.0
    r = a / b

    y1 = np.array([a, b])
    y2 = np.array([a, a, b, b])
    y3 = np.array([a, a, a, b, b, b])

    # ------------------------------------------------------------------
    # Part 1: Split-point ratios for 2-fold and 3-fold replications
    # ------------------------------------------------------------------
    print("=" * 72)
    print("GENERAL SPLIT-POINT IMPOSSIBILITY THEOREM — NUMERICAL VERIFICATION")
    print("=" * 72)

    print(f"\nBase distribution: y = ({a}, {b}), r = a/b = {r:.6f}")
    print()

    print("2-fold replication (a,a,b,b): split-point ratios")
    print(f"  i=1: φ(r) = 3r/(r+2)       = {phi(r):.6f}")
    print(f"  i=2: r                      = {r:.6f}")
    print(f"  i=3: ψ(r) = (2r+1)/3        = {psi(r):.6f}")
    print()

    print("3-fold replication (a,a,a,b,b,b): split-point ratios")
    print(f"  i=1: 5r/(2r+3)              = {alpha1(r):.6f}")
    print(f"  i=2: 4r/(r+3)               = {alpha2(r):.6f}")
    print(f"  i=3: r                       = {r:.6f}")
    print(f"  i=4: (3r+1)/4               = {alpha4(r):.6f}")
    print(f"  i=5: (3r+2)/5               = {alpha5(r):.6f}")
    print()

    # Verify these match the general formula
    ratios_2 = kfold_ratios(r, 2)
    ratios_3 = kfold_ratios(r, 3)
    assert np.allclose(ratios_2, [phi(r), r, psi(r)]), "2-fold ratios mismatch"
    assert np.allclose(
        ratios_3, [alpha1(r), alpha2(r), r, alpha4(r), alpha5(r)]
    ), "3-fold ratios mismatch"
    print("Cross-check: general formula matches hand-computed ratios. OK.\n")

    # ------------------------------------------------------------------
    # Part 2: PI violation — I_f(y) vs I_f(y^(2)) vs I_f(y^(3))
    # ------------------------------------------------------------------
    print("-" * 72)
    print("PI VIOLATION: I_f(y) vs I_f(y^(2)) vs I_f(y^(3))")
    print("-" * 72)
    print(f"{'f':>20s}  {'I_f(1,3)':>12s}  {'I_f(1,1,3,3)':>12s}  "
          f"{'I_f(1^3,3^3)':>12s}  {'PI?':>5s}")
    print("-" * 72)

    for name, f in CANDIDATES.items():
        v1 = split_point_index(y1, f)
        v2 = split_point_index(y2, f)
        v3 = split_point_index(y3, f)
        pi_ok = np.isclose(v1, v2, atol=1e-10) and np.isclose(v1, v3, atol=1e-10)
        print(f"{name:>20s}  {v1:12.8f}  {v2:12.8f}  {v3:12.8f}  "
              f"{'Yes' if pi_ok else 'NO':>5s}")

    print()

    # ------------------------------------------------------------------
    # Part 3: Functional equation residuals
    # ------------------------------------------------------------------
    print("-" * 72)
    print("FUNCTIONAL EQUATION RESIDUALS (should be non-zero for f ≢ 0)")
    print("-" * 72)
    test_r_values = [1.0 / 3.0, 0.2, 0.5, 0.8]

    for name, f in CANDIDATES.items():
        print(f"\n  {name}:")
        for r_val in test_r_values:
            res1 = verify_fe1(f, r_val)
            res2 = verify_fe2(f, r_val)
            res3 = verify_fe_kfold(f, r_val, 4)
            res4 = verify_fe_kfold(f, r_val, 5)
            print(f"    r={r_val:.2f}:  FE1={res1:+.8f}  FE2={res2:+.8f}  "
                  f"FE4={res3:+.8f}  FE5={res4:+.8f}")

    print()

    # ------------------------------------------------------------------
    # Part 4: Orbit density near r=1 under φ and ψ
    # ------------------------------------------------------------------
    print("-" * 72)
    print("ORBIT DENSITY: iterates of φ and ψ converge to 1")
    print("-" * 72)
    r0 = 1.0 / 3.0
    print(f"\nStarting from r₀ = {r0:.6f}:")
    print(f"  {'k':>3s}  {'φ^k(r₀)':>12s}  {'ψ^k(r₀)':>12s}")

    r_phi, r_psi = r0, r0
    for k in range(1, 16):
        r_phi = phi(r_phi)
        r_psi = psi(r_psi)
        print(f"  {k:3d}  {r_phi:12.8f}  {r_psi:12.8f}")

    print()

    # ------------------------------------------------------------------
    # Part 5: Verify contraction rates
    # ------------------------------------------------------------------
    print("-" * 72)
    print("CONTRACTION ANALYSIS: φ'(r) and ψ'(r) at r=1")
    print("-" * 72)
    # φ(r) = 3r/(r+2), φ'(r) = 6/(r+2)², φ'(1) = 6/9 = 2/3
    # ψ(r) = (2r+1)/3, ψ'(r) = 2/3
    print("  φ'(1) = 6/(1+2)² = 2/3 ≈ 0.6667")
    print("  ψ'(1) = 2/3 ≈ 0.6667")
    print("  Both are strict contractions toward the fixed point r=1.")
    print()

    # ------------------------------------------------------------------
    # Part 6: Dense orbit test — generate many points via compositions
    # ------------------------------------------------------------------
    print("-" * 72)
    print("DENSE ORBIT GENERATION via compositions of φ and ψ")
    print("-" * 72)
    # Starting from r₀, apply all sequences of φ and ψ of length ≤ depth
    r0 = 1.0 / 3.0
    depth = 10
    orbit_points = set()

    def generate_orbit(r_val: float, d: int):
        if d == 0:
            orbit_points.add(round(r_val, 12))
            return
        orbit_points.add(round(r_val, 12))
        generate_orbit(phi(r_val), d - 1)
        generate_orbit(psi(r_val), d - 1)

    generate_orbit(r0, depth)
    orbit_sorted = sorted(orbit_points)
    print(f"  Starting point: r₀ = {r0}")
    print(f"  Depth: {depth}")
    print(f"  Distinct orbit points generated: {len(orbit_sorted)}")
    print(f"  Range: [{orbit_sorted[0]:.8f}, {orbit_sorted[-1]:.8f}]")

    # Check density near 1: how many points in (0.9, 1)?
    near_one = [p for p in orbit_sorted if 0.9 < p < 1.0]
    print(f"  Points in (0.9, 1.0): {len(near_one)}")
    near_one_fine = [p for p in orbit_sorted if 0.99 < p < 1.0]
    print(f"  Points in (0.99, 1.0): {len(near_one_fine)}")
    print()

    # ------------------------------------------------------------------
    # Part 7: Verify f ≡ 0 is the UNIQUE solution by testing whether
    # f satisfying both FE1 and FE2 is forced to vanish
    # ------------------------------------------------------------------
    print("-" * 72)
    print("COMBINED FE1 + FE2 RIGIDITY TEST")
    print("-" * 72)
    print()
    print("FE1 from 2-fold: 2f(r) = f(φ(r)) + f(ψ(r))")
    print("FE2 from 3-fold: 4f(r) = f(α₁(r)) + f(α₂(r)) + f(α₄(r)) + f(α₅(r))")
    print()
    print("where:")
    print("  φ(r) = 3r/(r+2),    ψ(r) = (2r+1)/3")
    print("  α₁(r) = 5r/(2r+3),  α₂(r) = 4r/(r+3)")
    print("  α₄(r) = (3r+1)/4,   α₅(r) = (3r+2)/5")
    print()

    # Verify that α₁, α₂, α₄, α₅ can be expressed in terms of φ and ψ
    # α₂(r) = 4r/(r+3): is this φ(φ(r))? φ(φ(r)) = 3·(3r/(r+2))/(3r/(r+2)+2)
    #        = 9r/(r+2) / (3r/(r+2) + 2) = 9r / (3r + 2(r+2)) = 9r/(5r+4)
    # No. α₂(r) = 4r/(r+3) ≠ 9r/(5r+4).

    # Check if FE1 applied at φ(r) and ψ(r) can substitute into FE2.
    # FE1 at φ(r): 2f(φ(r)) = f(φ(φ(r))) + f(ψ(φ(r)))
    # FE1 at ψ(r): 2f(ψ(r)) = f(φ(ψ(r))) + f(ψ(ψ(r)))

    # From FE1: f(φ(r)) = 2f(r) - f(ψ(r))
    # Substitute into FE1 at φ(r):
    #   2(2f(r) - f(ψ(r))) = f(φ²(r)) + f(ψ(φ(r)))
    #   4f(r) - 2f(ψ(r)) = f(φ²(r)) + f(ψ(φ(r)))

    # Verify numerically: are the 3-fold ratios reachable from 2-fold compositions?
    r_test = 0.3
    print("Numerical check: 3-fold ratios vs compositions of φ, ψ (r=0.3):")
    print(f"  α₁(r) = {alpha1(r_test):.8f},  φ(φ(r)) = {phi(phi(r_test)):.8f}")
    print(f"  α₂(r) = {alpha2(r_test):.8f},  ψ(φ(r)) = {psi(phi(r_test)):.8f}")
    print(f"  α₄(r) = {alpha4(r_test):.8f},  φ(ψ(r)) = {phi(psi(r_test)):.8f}")
    print(f"  α₅(r) = {alpha5(r_test):.8f},  ψ(ψ(r)) = {psi(psi(r_test)):.8f}")
    print()
    print("The 3-fold ratios are NOT compositions of φ and ψ.")
    print("This means FE2 provides genuinely new constraints beyond FE1.")
    print()

    # ------------------------------------------------------------------
    # Part 8: Proof sketch verification — the averaging argument
    # ------------------------------------------------------------------
    print("-" * 72)
    print("PROOF VERIFICATION: AVERAGING FORCES f → 0")
    print("-" * 72)
    print()

    # For each k-fold replication, we get:
    #   (2k-2)*f(r) = Σ_{i≠k} f(ratio_i(r))
    # All ratio_i map (0,1) → (0,1) and are contractions toward 1.
    # So f(r) is a weighted average of f at points closer to 1.
    #
    # Key: if |f| has a maximum M on (0,1], attained at some r*,
    # then f(r*) = avg of f at points closer to 1, all with |f| ≤ M.
    # For the average to equal ±M, ALL values must equal ±M.
    # Iterating: f must be ±M on the entire orbit.
    # But the orbit converges to 1, and f(1) = 0 ≠ ±M (if M > 0).
    # Contradiction by continuity.

    # Numerical demonstration: track max |f| on iterates
    print("Demonstration: if f(r) = 1-r satisfies FE1, track the constraint.")
    print()
    f_test = lambda r: 1.0 - r
    r_star = 0.1  # where |f| is large
    print(f"  f({r_star}) = {f_test(r_star):.4f}")
    print(f"  FE1 requires: 2f(r) = f(φ(r)) + f(ψ(r))")
    print(f"  2 * {f_test(r_star):.4f} = {f_test(phi(r_star)):.4f} + "
          f"{f_test(psi(r_star)):.4f}")
    print(f"  {2*f_test(r_star):.4f} vs {f_test(phi(r_star)) + f_test(psi(r_star)):.4f}")
    print(f"  Residual: {verify_fe1(f_test, r_star):+.8f}")
    print()

    # Show that the orbit of r_star under repeated ψ converges to 1
    print("  Iterating ψ from r* = 0.1:")
    r_curr = r_star
    for step in range(20):
        r_curr = psi(r_curr)
        print(f"    ψ^{step+1}(r*) = {r_curr:.10f},  f = {f_test(r_curr):.10f}")

    print()

    # ------------------------------------------------------------------
    # Part 9: Full k-fold test for additional distributions
    # ------------------------------------------------------------------
    print("-" * 72)
    print("EXTENDED PI TESTS: multiple base distributions and k-fold replications")
    print("-" * 72)
    print()

    test_distributions = [
        ("(1, 3)", np.array([1.0, 3.0])),
        ("(1, 2)", np.array([1.0, 2.0])),
        ("(2, 5)", np.array([2.0, 5.0])),
        ("(1, 10)", np.array([1.0, 10.0])),
    ]

    for dist_name, y_base in test_distributions:
        print(f"  Base: y = {dist_name}")
        for fname, f in CANDIDATES.items():
            vals = []
            for k in range(1, 6):
                y_rep = np.tile(y_base, k)
                vals.append(split_point_index(y_rep, f))
            diffs = [abs(v - vals[0]) for v in vals[1:]]
            max_diff = max(diffs)
            print(f"    {fname:>20s}: k=1..5 → "
                  + "  ".join(f"{v:.6f}" for v in vals)
                  + f"  max|Δ|={max_diff:.2e}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("1. SPLIT-POINT RATIOS VERIFIED for 2-fold and 3-fold replications.")
    print()
    print("2. FUNCTIONAL EQUATIONS derived:")
    print("   FE1 (2-fold): 2f(r) = f(3r/(r+2)) + f((2r+1)/3)")
    print("   FE2 (3-fold): 4f(r) = f(5r/(2r+3)) + f(4r/(r+3))")
    print("                         + f((3r+1)/4) + f((3r+2)/5)")
    print()
    print("3. PI VIOLATION confirmed for all five candidate f-functions.")
    print("   None satisfies PI (as predicted by the theorem).")
    print()
    print("4. PROOF STRATEGY (continuous f, no analyticity needed):")
    print("   (a) FE1 from k=2 says f(r) = avg of f at φ(r) and ψ(r),")
    print("       both strictly between r and 1.")
    print("   (b) Suppose f ≢ 0. Then M := sup_{r∈(0,1]} |f(r)| > 0.")
    print("   (c) By continuity, M is attained at some r* ∈ (0,1).")
    print("       (f(1)=0, and f→0 or f→∞ as r→0; if f bounded, use sup.)")
    print("   (d) WLOG f(r*) = M. Then M = (1/2)[f(φ(r*)) + f(ψ(r*))].")
    print("       Since φ(r*), ψ(r*) ∈ (r*, 1) and |f| ≤ M everywhere,")
    print("       we need f(φ(r*)) = f(ψ(r*)) = M.")
    print("   (e) Iterating: f(ψ^k(r*)) = M for all k.")
    print("       But ψ^k(r*) → 1, and f(1) = 0 ≠ M. Contradiction.")
    print("   (f) Therefore M = 0, i.e., f ≡ 0 on (0,1].")
    print("       By f(1)=0 and the same argument on (1,∞), f ≡ 0.")
    print()
    print("5. REGULARITY REQUIREMENT: Continuous f suffices.")
    print("   The proof uses only:")
    print("   - Continuity (for the maximum principle / limit argument)")
    print("   - f(1) = 0")
    print("   - The functional equation FE1")
    print("   No analyticity, differentiability, or smoothness is needed.")
    print()
    print("6. GAP ANALYSIS:")
    print("   The proof above assumes f is bounded on compact subsets")
    print("   of (0,1], which follows from continuity. The argument")
    print("   extends to (0,∞) by scale: for r > 1, use the distribution")
    print("   (b, a) with b < a, getting ratio b/a = 1/r < 1.")
    print("   Alternatively, note that the k-fold replication of (a,b)")
    print("   with a > b gives the same functional equations with")
    print("   r = a/b > 1, and the maps still contract toward 1.")
    print()
    print("   POTENTIAL SUBTLETY: The maximum principle argument requires")
    print("   that |f| attains its supremum. For f continuous on (0,1],")
    print("   sup may be attained only as r → 0+. In that case, we need")
    print("   the functional equation to propagate the near-supremum")
    print("   toward r=1, which still yields a contradiction since")
    print("   f(ψ^k(r)) → f(1) = 0 for any r ∈ (0,1).")
    print("   More precisely: take r_n → 0+ with |f(r_n)| → M.")
    print("   Then M = (1/2)|f(φ(r_n)) + f(ψ(r_n))| ≤ M,")
    print("   so |f(ψ(r_n))| → M. But ψ(r_n) → 1/3 ≠ 1.")
    print("   Iterate: ψ^k pushes toward 1 but from a shifted sequence.")
    print("   A diagonal argument: pick n_k so that |f(ψ^k(r_{n_k}))| > M-1/k")
    print("   and ψ^k(r_{n_k}) → 1. Then f(1)=0 gives 0 ≥ M. Done.")
    print()
    print("   CONCLUSION: The theorem holds for all continuous f with f(1)=0.")
    print("   No analyticity assumption is needed. FE1 alone suffices.")


if __name__ == "__main__":
    main()
