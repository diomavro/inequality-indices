"""Counterexamples behind the corrected claims (run: python3 verify_corrections.py).

Each check is a claim the paper now makes; each assert fails if the claim is wrong.
"""

import math

import numpy as np
from scipy.optimize import brentq

import inequality_indices as ii


def expo(y, kappa):
    """Exponential absolute index E_kappa = mean(exp(kappa(mu - y))) - 1."""
    y = np.asarray(y, float)
    return float(np.mean(np.exp(kappa * (y.mean() - y))) - 1)


def transfer(y, donor, recip, delta):
    y = np.array(y, float)
    y[donor] -= delta
    y[recip] += delta
    return y


# 1. E_kappa (kappa > 0) is translation-invariant, additively decomposable and
#    satisfies DTS, so an absolute index reaches ten properties (Theorem 3 corrected).
y = np.array([5, 15, 50, 100, 190, 200.0])
k = 0.05
assert abs(expo(y, k) - expo(y + 37, k)) < 1e-12  # translation
A, B = y[:3], y[3:]


def w(g):
    return len(g) / len(y) * np.exp(k * (y.mean() - g.mean()))


between = np.r_[np.full(3, A.mean()), np.full(3, B.mean())]
assert (
    abs(expo(y, k) - (w(A) * expo(A, k) + w(B) * expo(B, k) + expo(between, k))) < 1e-12
)
low = expo(y, k) - expo(transfer(y, 1, 0, 1.0), k)  # 15 -> 5
high = expo(y, k) - expo(transfer(y, 5, 4, 1.0), k)  # 200 -> 190
assert low > high > 0, (low, high)
print(f"E_kappa: TI, AD hold; DTS: low-transfer drop {low:.4f} > high {high:.2e}")

# 2. Zenga fails DTS: y = (3,6,7,7,8); 0.5 from 7 to 6 (gap 1, level 6) reduces
#    Z and Z* by LESS than 0.5 from 8 to 7 (gap 1, level 7).
y = np.array([3, 6, 7, 7, 8.0])
lo, hi = transfer(y, 2, 1, 0.5), transfer(y, 4, 3, 0.5)
for f in (ii.zenga, ii.integral_zenga):
    d_lo, d_hi = f(y) - f(lo), f(y) - f(hi)
    assert 0 < d_lo < d_hi, (f.__name__, d_lo, d_hi)
    print(f"{f.__name__}: DTS fails, low {d_lo:.4f} < high {d_hi:.4f}")
for f in (ii.mld, ii.theil):  # DTS indices get it right on the same example
    assert f(y) - f(lo) > f(y) - f(hi)

# 3. Zenga (Z*) fails subgroup consistency: subgroup A's Z* rises at a fixed mean,
#    B unchanged, yet total Z* falls.
A, A2, B = (
    np.array([36, 18, 15.0]),
    np.array([33, 12, 24.0]),
    np.array([28, 23, 2, 3.0]),
)
assert A.mean() == A2.mean()
f = ii.integral_zenga
assert f(A2) > f(A) and f(np.r_[A2, B]) < f(np.r_[A, B])
print(
    f"Z*: subgroup {f(A):.4f}->{f(A2):.4f}, total {f(np.r_[A, B]):.4f}->{f(np.r_[A2, B]):.4f}"
)
assert ii.mld(A2) > ii.mld(A) and ii.mld(np.r_[A2, B]) > ii.mld(
    np.r_[A, B]
)  # MLD consistent
print("all corrected claims verified")


# 4. Simulations quoted in the text (Propositions dts and subgroup).
def dts_failure_rate(n, reps, rng):
    """Share of random (low pair, high pair) comparisons where the high transfer
    reduces Z / Z* by at least as much; log-normal incomes, same gap d."""
    fails = {"Z": 0, "Z*": 0}
    done = 0
    while done < reps:
        y = np.sort(rng.lognormal(0, 0.8, n) * 10)
        i, j = rng.integers(0, n // 2 - 1), rng.integers(n // 2, n - 1)
        d = min(y[i + 1] - y[i], y[j + 1] - y[j])
        y2 = y.copy()
        y2[i + 1], y2[j + 1] = y2[i] + d, y2[j] + d
        if d <= 0 or np.any(np.diff(y2) < 0):
            continue
        done += 1
        lo, hi = transfer(y2, i + 1, i, d / 4), transfer(y2, j + 1, j, d / 4)
        for key, f in (("Z", ii.zenga), ("Z*", ii.integral_zenga)):
            fails[key] += not (f(y2) - f(lo) > f(y2) - f(hi))
    return {k: v / reps for k, v in fails.items()}


def sc_violation_rate(reps, rng):
    """Same-size, same-mean replacement of subgroup A raising its index while
    the total falls (B unchanged); small integer populations."""
    out = {}
    for key, f in (("Z", ii.zenga), ("Z*", ii.integral_zenga), ("MLD", ii.mld)):
        raised = viol = 0
        for _ in range(reps):
            na, nb = rng.integers(2, 5), rng.integers(1, 5)
            A = rng.integers(1, 40, na).astype(float)
            B = rng.integers(1, 40, nb).astype(float)
            A2 = rng.integers(1, 40, na).astype(float)
            A2 *= A.sum() / A2.sum()
            if f(A2) > f(A) + 1e-12:
                raised += 1
                viol += f(np.r_[A2, B]) <= f(np.r_[A, B]) + 1e-12
        out[key] = viol / raised
    return out


rng = np.random.default_rng(2026)
r100 = dts_failure_rate(100, 1000, rng)
print(f"DTS failure rate at n=100: {r100}")
assert r100["Z"] > 0 and r100["Z*"] > 0  # failure persists in large samples
sc = sc_violation_rate(5000, rng)
print(f"Subgroup-consistency violation rate: {sc}")
assert sc["MLD"] == 0 and 0.005 < sc["Z*"] < 0.05 and 0.005 < sc["Z"] < 0.05
print("simulation claims verified")


# 5. Zenga's own finite-population index Z_N: replication-invariant, discontinuous
#    at ties, and it fails DTS and subgroup consistency on the same examples.
zn = ii.zenga_finite
assert abs(zn([1, 3]) - zn([1, 1, 3, 3])) < 1e-12 and abs(zn([1, 3]) - 1 / 3) < 1e-12
assert abs(zn([1, 1 + 1e-9, 3]) - 7 / 18) < 1e-6 and abs(zn([1, 1, 3]) - 4 / 9) < 1e-12
y = np.array([3, 6, 7, 7, 8.0])
assert 0 < zn(y) - zn(transfer(y, 2, 1, 0.5)) < zn(y) - zn(transfer(y, 4, 3, 0.5))
A, A2, B = (
    np.array([36, 18, 15.0]),
    np.array([33, 12, 24.0]),
    np.array([28, 23, 2, 3.0]),
)
assert zn(A2) > zn(A) and zn(np.r_[A2, B]) < zn(np.r_[A, B])
print("Z_N: replication-invariant, jumps 7/18 -> 4/9 at a tie, fails DTS and SC")
# Z_N violates the transfer principle when a transfer creates a tie.
assert zn([1.1, 1.1, 3]) > zn([1, 1.2, 3])
assert ii.integral_zenga([1.1, 1.1, 3]) < ii.integral_zenga([1, 1.2, 3])
assert ii.zenga([1.1, 1.1, 3]) < ii.zenga([1, 1.2, 3])
print(
    f"Z_N transfer-principle failure: {zn([1, 1.2, 3]):.4f} -> {zn([1.1, 1.1, 3]):.4f}"
)


# 6. Additive decomposability failures: same subgroup size, mean and index value,
#    different totals (so no weights can work).


def r9010(y):
    y = np.sort(y)
    n = len(y)
    return y[math.ceil(0.9 * n) - 1] / y[math.ceil(0.1 * n) - 1]


A = np.array([3.0, 9.0])
B, Bp = np.array([2, 4, 6.0]), np.array([2.6, 2.8, 6.6])
assert abs(ii.gini(B) - ii.gini(Bp)) < 1e-12 and B.mean() == Bp.mean()
assert abs(ii.gini(np.r_[A, B]) - ii.gini(np.r_[A, Bp])) > 0.005
A2, B, Bp = np.array([4, 14.0]), np.array([3, 5, 18.0]), np.array([2, 12, 12.0])
assert r9010(B) == r9010(Bp) == 6 and B.sum() == Bp.sum()
assert r9010(np.r_[A2, B]) == 6 and r9010(np.r_[A2, Bp]) == 7
B = np.array([2, 4, 6.0])
for f in (ii.zenga, ii.zenga_finite, ii.integral_zenga):

    def g(x, f=f):
        return f(np.array([x, x + 0.2, 11.8 - 2 * x])) - f(B)

    xs = np.linspace(0.5, 3.9, 400)
    lo = next(a for a, b in zip(xs, xs[1:]) if g(a) * g(b) < 0)
    x = brentq(g, lo, lo + xs[1] - xs[0], xtol=1e-14)
    Bp = np.array([x, x + 0.2, 11.8 - 2 * x])
    assert abs(f(np.r_[A, B]) - f(np.r_[A, Bp])) > 0.005, f.__name__
print("additive-decomposability counterexamples verified (Gini, P90/P10, Z, Z_N, Z*)")
# Z shares the Z* subgroup-consistency counterexample.
A, A2, B = (
    np.array([36, 18, 15.0]),
    np.array([33, 12, 24.0]),
    np.array([28, 23, 2, 3.0]),
)
assert ii.zenga(A2) > ii.zenga(A) and ii.zenga(np.r_[A2, B]) < ii.zenga(np.r_[A, B])
print("Z subgroup-consistency counterexample verified")
