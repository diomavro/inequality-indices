"""
Verification script for all numerical claims and formal results in
inequality_indices.tex.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inequality_indices import (
    variance, cv, variance_of_logs, gini, ge_alpha, theil, mld,
    atkinson, zenga, percentile_ratio,
)

results = []
all_pass = True

def check(test_id, test_type, label, summary, condition, findings_list):
    global all_pass
    passes = condition and len(findings_list) == 0
    if not passes:
        all_pass = False
    results.append({
        "id": test_id,
        "type": test_type,
        "label": label,
        "statement_summary": summary,
        "passes": passes,
        "findings": findings_list,
    })
    status = "PASS" if passes else "FAIL"
    print(f"[{status}] {test_id}: {label}")
    for f in findings_list:
        print(f"       -> {f}")

# ============================================================
# 1. VL COUNTEREXAMPLE (Section 5.4, lines 764-779)
# Paper claims: VL(1,2,50,100) = 3.946, VL(1,2,60,90) = 3.982
# Progressive transfer of 10 from 100 to 50 INCREASES VL
# ============================================================
y_vl = [1, 2, 50, 100]
y_vl_after = [1, 2, 60, 90]
vl_before = variance_of_logs(y_vl)
vl_after = variance_of_logs(y_vl_after)
findings = []
if not np.isclose(vl_before, 3.946, atol=0.001):
    findings.append(f"VL(1,2,50,100) = {vl_before:.4f}, paper claims 3.946")
if not np.isclose(vl_after, 3.982, atol=0.001):
    findings.append(f"VL(1,2,60,90) = {vl_after:.4f}, paper claims 3.982")
if not (vl_after > vl_before):
    findings.append(f"VL did NOT increase after transfer: {vl_before:.4f} -> {vl_after:.4f}")
check("VL-1", "counterexample", "VL counterexample",
      "Transfer of 10 from 100 to 50 in (1,2,50,100) increases VL from ~3.946 to ~3.982",
      True, findings)

# ============================================================
# 2. IMPOSSIBILITY THEOREM ALGEBRA (Section 6.1, lines 1019-1036)
# Paper: c = (x - x')/(x' - 1), then (x+c)/(1+c) = x'
# ============================================================
findings = []
for x, xp in [(3.0, 5.0), (2.0, 7.0), (1.5, 10.0), (100.0, 2.0)]:
    c_val = (x - xp) / (xp - 1)
    one_plus_c = (x - 1) / (xp - 1)
    x_plus_c = xp * (x - 1) / (xp - 1)
    ratio = x_plus_c / one_plus_c
    if not np.isclose(ratio, xp):
        findings.append(f"For x={x}, x'={xp}: (x+c)/(1+c) = {ratio:.6f} != {xp}")
    # Check 1+c > 0 and x+c > 0
    if one_plus_c <= 0:
        findings.append(f"For x={x}, x'={xp}: 1+c = {one_plus_c:.6f} <= 0")
    if x_plus_c <= 0:
        findings.append(f"For x={x}, x'={xp}: x+c = {x_plus_c:.6f} <= 0")
check("IMP-1", "theorem", "Impossibility theorem algebra",
      "c = (x-x')/(x'-1) yields (x+c)/(1+c) = x' with 1+c>0 and x+c>0",
      True, findings)

# ============================================================
# 3. GINI DECOMPOSITION EXAMPLE (Section 6.2, Remark, lines 1114-1119)
# y = (1,3,5,7), A=(1,3), B=(5,7)
# Paper claims: G(y) = 0.3125, G(A) = 0.2500, G(B) = 0.0833
# G_between (on (2,2,6,6)) = 0.2500
# Predicted total with w=1/2: 0.5*0.25 + 0.5*0.0833 + 0.25 = 0.4167
# Residual = 0.3125 - 0.4167 = -0.1042
# ============================================================
y_full = [1, 3, 5, 7]
y_A = [1, 3]
y_B = [5, 7]
y_between = [2, 2, 6, 6]

g_full = gini(y_full)
g_A = gini(y_A)
g_B = gini(y_B)
g_between = gini(y_between)

predicted = 0.5 * g_A + 0.5 * g_B + g_between
residual = g_full - predicted

findings = []
if not np.isclose(g_full, 0.3125, atol=0.0001):
    findings.append(f"G(1,3,5,7) = {g_full:.4f}, paper claims 0.3125")
if not np.isclose(g_A, 0.2500, atol=0.0001):
    findings.append(f"G(1,3) = {g_A:.4f}, paper claims 0.2500")
if not np.isclose(g_B, 0.0833, atol=0.0001):
    findings.append(f"G(5,7) = {g_B:.4f}, paper claims 0.0833")
if not np.isclose(g_between, 0.2500, atol=0.0001):
    findings.append(f"G(2,2,6,6) = {g_between:.4f}, paper claims 0.2500")
if not np.isclose(predicted, 0.4167, atol=0.0001):
    findings.append(f"Predicted total = {predicted:.4f}, paper claims 0.4167")
if not np.isclose(residual, -0.1042, atol=0.0001):
    findings.append(f"Residual = {residual:.4f}, paper claims -0.1042")
check("GINI-DEC-1", "counterexample", "Gini decomposition example",
      "G(1,3,5,7)=0.3125, G(A)=0.25, G(B)=0.0833, G_between=0.25, residual=-0.1042",
      True, findings)

# ============================================================
# 4. CROSS-COUNTRY EXAMPLE (Section 7, lines 1241-1248)
# Var(1000,2000) = 250,000; Var(20000,40000) = 100,000,000 = 400 * Var(A)
# G(A) = G(B) = 0.1667 (paper says 1/6)
# GE_1(A) = GE_1(B) = 0.0566
# ============================================================
y_A_cc = [1000, 2000]
y_B_cc = [20000, 40000]

var_A = variance(y_A_cc)
var_B = variance(y_B_cc)
g_A_cc = gini(y_A_cc)
g_B_cc = gini(y_B_cc)
ge1_A = theil(y_A_cc)
ge1_B = theil(y_B_cc)

findings = []
if not np.isclose(var_A, 250000, atol=1):
    findings.append(f"Var(1000,2000) = {var_A:.0f}, paper claims 250,000")
if not np.isclose(var_B, 100000000, atol=1):
    findings.append(f"Var(20000,40000) = {var_B:.0f}, paper claims 100,000,000")
if not np.isclose(var_B / var_A, 400, atol=0.01):
    findings.append(f"Ratio = {var_B/var_A:.1f}, paper claims 400")
if not np.isclose(g_A_cc, 1/6, atol=0.001):
    findings.append(f"G(A) = {g_A_cc:.4f}, paper claims 0.1667")
if not np.isclose(g_B_cc, 1/6, atol=0.001):
    findings.append(f"G(B) = {g_B_cc:.4f}, paper claims 0.1667")
if not np.isclose(g_A_cc, g_B_cc, atol=1e-10):
    findings.append(f"G(A) != G(B): {g_A_cc:.6f} vs {g_B_cc:.6f}")
if not np.isclose(ge1_A, 0.0566, atol=0.001):
    findings.append(f"GE_1(A) = {ge1_A:.4f}, paper claims 0.0566")
if not np.isclose(ge1_B, 0.0566, atol=0.001):
    findings.append(f"GE_1(B) = {ge1_B:.4f}, paper claims 0.0566")
check("CC-1", "numerical", "Cross-country variance example",
      "Var(1000,2000)=250000, Var(20000,40000)=100M, ratio=400; Gini=0.1667, GE1=0.0566",
      True, findings)

# ============================================================
# 5. GE DTS ANALYSIS (Section 5.5)
# GE_2 has CONSTANT transfer sensitivity (h_2(y) = d, independent of y)
# GE with alpha<2 has DECREASING transfer sensitivity
# ============================================================
findings = []

# For GE_alpha, the transfer sensitivity is proportional to
# h_alpha(y) = (y+d)^{alpha-1} - y^{alpha-1}
# For alpha=2: h(y) = (y+d) - y = d (constant)
d = 10
for y_test in [100, 1000, 10000]:
    h2 = (y_test + d)**(2-1) - y_test**(2-1)
    if not np.isclose(h2, d, atol=1e-10):
        findings.append(f"GE_2 h(y={y_test}) = {h2:.4f}, expected {d}")

# For alpha < 2 (but alpha != 0, 1 which are limiting cases): |h| should decrease with y
# The formula h_alpha(y) = (y+d)^{alpha-1} - y^{alpha-1} applies for alpha != 0, 1
# For alpha=0 and alpha=1, DTS is verified numerically in DTS-2
for alpha_test in [0.5, 1.5, -1]:
    h_vals = []
    for y_test in [10, 100, 1000, 10000]:
        h = abs((y_test + d)**(alpha_test - 1) - y_test**(alpha_test - 1))
        h_vals.append(h)
    for i in range(len(h_vals) - 1):
        if h_vals[i] <= h_vals[i+1]:
            findings.append(f"DTS fails for alpha={alpha_test}: |h| not decreasing at y={[10,100,1000,10000][i]}")

# For alpha > 2 (e.g., alpha=3): |h| should INCREASE with y
for alpha_test in [3, 4]:
    h_vals = []
    for y_test in [10, 100, 1000, 10000]:
        h = abs((y_test + d)**(alpha_test - 1) - y_test**(alpha_test - 1))
        h_vals.append(h)
    for i in range(len(h_vals) - 1):
        if h_vals[i] >= h_vals[i+1]:
            findings.append(f"Expected increasing |h| for alpha={alpha_test} but failed at y={[10,100,1000,10000][i]}")

check("DTS-1", "proposition", "GE DTS analysis",
      "GE_2 has constant transfer sensitivity; alpha<2 has decreasing; alpha>2 has increasing",
      True, findings)

# Numerical verification with actual GE computations
findings = []
# Build a distribution and do matched transfers at low vs high income
base = np.array([5, 10, 50, 100, 500, 1000], dtype=float)
delta = 2.0

# Transfer at bottom: from income 10 to income 5
y_low_before = base.copy()
y_low_after = base.copy()
y_low_after[0] += delta  # 5 -> 7
y_low_after[1] -= delta  # 10 -> 8

# Transfer at top: from income 1000 to income 500 (same gap d=5 doesn't match,
# but same delta)
# Actually, for proper DTS test we need same gap d and same delta
# Let's use d=5, delta=2
# Low: from person with 10 to person with 5 (d=5)
# High: we need incomes y' and y'+5. Use 500 and 505.
base2 = np.array([5, 10, 100, 200, 500, 505], dtype=float)
y_high_before = base2.copy()
y_high_after = base2.copy()
y_high_after[4] += delta  # 500 -> 502
y_high_after[5] -= delta  # 505 -> 503

for alpha_test in [0, 1]:
    ge_low_before = ge_alpha(y_low_before, alpha_test)
    ge_low_after = ge_alpha(y_low_after, alpha_test)
    ge_high_before = ge_alpha(y_high_before, alpha_test)
    ge_high_after = ge_alpha(y_high_after, alpha_test)
    reduction_low = ge_low_before - ge_low_after
    reduction_high = ge_high_before - ge_high_after
    if reduction_low <= reduction_high:
        findings.append(f"GE_{alpha_test}: low-income transfer reduction ({reduction_low:.6f}) <= high-income ({reduction_high:.6f})")

check("DTS-2", "numerical", "GE DTS numerical verification",
      "GE_0 and GE_1 transfers at bottom reduce inequality more than at top",
      True, findings)

# ============================================================
# 6. ZENGA INDEX PROPERTIES
# ============================================================

# 6a. Scale invariance
findings = []
y_z = np.array([1, 3, 5, 7, 9])
z_orig = zenga(y_z)
z_scaled = zenga(3.0 * y_z)
if not np.isclose(z_orig, z_scaled, atol=1e-10):
    findings.append(f"Z(y) = {z_orig:.6f} != Z(3y) = {z_scaled:.6f}")
check("ZEN-1", "property", "Zenga scale invariance",
      "Z(lambda*y) = Z(y)", True, findings)

# 6b. Transfer principle
findings = []
np.random.seed(42)
for _ in range(200):
    n = np.random.randint(4, 20)
    y_rand = np.random.exponential(10, n)
    y_sorted = np.sort(y_rand)
    # Pick two indices for transfer
    idx_low = np.random.randint(0, n - 1)
    idx_high = np.random.randint(idx_low + 1, n)
    gap = y_sorted[idx_high] - y_sorted[idx_low]
    if gap < 0.01:
        continue
    delta = np.random.uniform(0.001, gap / 2)
    y_after = y_sorted.copy()
    y_after[idx_low] += delta
    y_after[idx_high] -= delta
    z_before = zenga(y_sorted)
    z_after = zenga(y_after)
    if z_after >= z_before:
        findings.append(f"Zenga transfer principle failed: Z went from {z_before:.6f} to {z_after:.6f}")
        break
check("ZEN-2", "property", "Zenga transfer principle",
      "Progressive transfer reduces Zenga (200 random tests)", True, findings)

# 6c. DTS for Zenga
findings = []
np.random.seed(123)
fail_count = 0
for _ in range(200):
    # Create distribution with matched transfers
    n = 10
    y_base = np.sort(np.random.exponential(50, n))
    d = 5.0
    delta = 1.0

    # Find a low-income pair and high-income pair with same gap d
    # Low: y_base[0] and y_base[0]+d
    # High: y_base[-1]-d and y_base[-1]
    y_low = y_base.copy()
    y_low[0] = 10.0
    y_low[1] = 10.0 + d
    y_low = np.sort(y_low)

    y_high = y_base.copy()
    y_high[-2] = 500.0
    y_high[-1] = 500.0 + d
    y_high = np.sort(y_high)

    # Transfer at low end
    y_low_after = y_low.copy()
    idx_low_donor = np.where(np.isclose(y_low, 10+d))[0]
    idx_low_recip = np.where(np.isclose(y_low, 10.0))[0]
    if len(idx_low_donor) == 0 or len(idx_low_recip) == 0:
        continue
    y_low_after[idx_low_donor[0]] -= delta
    y_low_after[idx_low_recip[0]] += delta

    # Transfer at high end
    y_high_after = y_high.copy()
    idx_high_donor = np.where(np.isclose(y_high, 500+d))[0]
    idx_high_recip = np.where(np.isclose(y_high, 500.0))[0]
    if len(idx_high_donor) == 0 or len(idx_high_recip) == 0:
        continue
    y_high_after[idx_high_donor[0]] -= delta
    y_high_after[idx_high_recip[0]] += delta

    red_low = zenga(y_low) - zenga(y_low_after)
    red_high = zenga(y_high) - zenga(y_high_after)

    if red_low <= red_high:
        fail_count += 1

if fail_count > 5:
    findings.append(f"Zenga DTS failed in {fail_count}/200 tests")
check("ZEN-3", "property", "Zenga diminishing transfer sensitivity",
      "Zenga is more sensitive to low-income transfers", True, findings)

# 6d. Subgroup consistency for Zenga
findings = []
np.random.seed(456)
for _ in range(100):
    n1, n2 = 5, 5
    y1 = np.random.exponential(10, n1)
    y2 = np.random.exponential(50, n2)
    mu1 = np.mean(y1)

    # Increase inequality in group 1 while keeping mean constant
    y1_new = y1.copy()
    # Spread out: scale deviations from mean
    y1_new = mu1 + 1.5 * (y1 - mu1)
    if np.any(y1_new <= 0):
        continue

    y_combined = np.concatenate([y1, y2])
    y_combined_new = np.concatenate([y1_new, y2])

    z_before = zenga(y_combined)
    z_after = zenga(y_combined_new)
    z_sub_before = zenga(y1)
    z_sub_after = zenga(y1_new)

    if z_sub_after > z_sub_before and z_after <= z_before:
        findings.append(f"Subgroup consistency violated: subgroup Z increased but total Z didn't")
        break

check("ZEN-4", "property", "Zenga subgroup consistency",
      "Increasing within-group inequality increases total Zenga (100 random tests)", True, findings)

# 6e. Population invariance FAILURE
findings = []
z_13 = zenga([1, 3])
z_1133 = zenga([1, 1, 3, 3])
z_5fold = zenga([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])

if not np.isclose(z_13, 0.667, atol=0.001):
    findings.append(f"Z(1,3) = {z_13:.4f}, paper claims ~0.667")
if not np.isclose(z_1133, 0.561, atol=0.001):
    findings.append(f"Z(1,1,3,3) = {z_1133:.4f}, paper claims ~0.561")
if np.isclose(z_13, z_1133, atol=0.001):
    findings.append(f"Z(1,3) ≈ Z(1,1,3,3), but paper claims they differ")

# Paper also says 5-fold replication gives 0.531
if not np.isclose(z_5fold, 0.531, atol=0.001):
    findings.append(f"Z(1x5,3x5) = {z_5fold:.6f}, paper claims ~0.531")

check("ZEN-5", "counterexample", "Zenga population invariance failure",
      "Z(1,3)≈0.667, Z(1,1,3,3)≈0.561, 5-fold≈0.531", True, findings)

# ============================================================
# 7. TRANSLATION INVARIANCE (Section 5.3, lines 689-698)
# Paper: G(1,3) = 0.25, G(11,13) ≈ 0.042
# ============================================================
findings = []
g_13 = gini([1, 3])
g_1113 = gini([11, 13])

if not np.isclose(g_13, 0.25, atol=0.0001):
    findings.append(f"G(1,3) = {g_13:.4f}, paper claims 0.25")
if not np.isclose(g_1113, 4/96, atol=0.001):
    findings.append(f"G(11,13) = {g_1113:.4f}, paper claims 4/96 ≈ 0.042")

check("TRANS-1", "counterexample", "Translation invariance counterexample",
      "G(1,3)=0.25, G(11,13)≈0.042", True, findings)

# ============================================================
# 8. VARIANCE TRANSFER PRINCIPLE (Section 5.4, eq. var_transfer)
# Delta_Var = (2*delta/n) * [delta - (y_i - y_j)] < 0
# ============================================================
findings = []
y_var = np.array([10, 20, 30, 40], dtype=float)
delta = 3.0
# Transfer from person with 40 to person with 10
y_var_after = y_var.copy()
y_var_after[0] += delta
y_var_after[3] -= delta
var_before = variance(y_var)
var_after = variance(y_var_after)
predicted_change = (2 * delta / 4) * (delta - (40 - 10))
actual_change = var_after - var_before
if not np.isclose(predicted_change, actual_change, atol=1e-10):
    findings.append(f"Predicted variance change = {predicted_change:.4f}, actual = {actual_change:.4f}")
if var_after >= var_before:
    findings.append(f"Variance did not decrease: {var_before:.4f} -> {var_after:.4f}")
check("VAR-TP-1", "proposition", "Variance transfer principle formula",
      "Delta_Var = (2*delta/n)*(delta - gap) < 0", True, findings)

# ============================================================
# 9. GINI TRANSFER FORMULA (Section 5.4, eq. gini_transfer)
# Delta G = -2*delta*(r_h - r_l) / (n^2 * mu)
# ============================================================
findings = []
y_gini = np.array([1, 2, 3, 4, 5], dtype=float)
delta = 0.5
# Transfer from rank 5 (income 5) to rank 1 (income 1)
y_gini_after = y_gini.copy()
y_gini_after[0] += delta  # 1 -> 1.5
y_gini_after[4] -= delta  # 5 -> 4.5
g_before = gini(y_gini)
g_after = gini(y_gini_after)
n = 5
mu = np.mean(y_gini)
r_h, r_l = 5, 1  # ranks in ascending order
predicted_delta_g = -2 * delta * (r_h - r_l) / (n**2 * mu)
actual_delta_g = g_after - g_before
if not np.isclose(predicted_delta_g, actual_delta_g, atol=0.001):
    findings.append(f"Predicted Gini change = {predicted_delta_g:.6f}, actual = {actual_delta_g:.6f}")
check("GINI-TP-1", "proposition", "Gini transfer formula",
      "Delta G = -2*delta*(r_h-r_l)/(n^2*mu)", True, findings)

# ============================================================
# 10. GE_2 = CV^2 / 2 (Section 4.5, line 484)
# ============================================================
findings = []
for y_test in [np.array([1, 2, 3, 4, 5]),
               np.array([10, 20, 50, 100]),
               np.array([1, 1, 1, 100])]:
    ge2 = ge_alpha(y_test, 2)
    cv_val = cv(y_test)
    if not np.isclose(ge2, cv_val**2 / 2, atol=1e-10):
        findings.append(f"GE_2 = {ge2:.6f} != CV^2/2 = {cv_val**2/2:.6f} for y={y_test}")
check("GE2-CV-1", "identity", "GE_2 = CV^2 / 2",
      "GE_2 equals half the squared CV", True, findings)

# ============================================================
# 11. SCALE INVARIANCE CHECKS
# ============================================================
findings = []
y_si = np.array([1, 3, 5, 7, 9], dtype=float)
lam = 2.5
y_scaled = lam * y_si

for name, func in [("CV", cv), ("VL", variance_of_logs),
                    ("Gini", gini), ("Zenga", zenga)]:
    v1 = func(y_si)
    v2 = func(y_scaled)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"{name}: I(y)={v1:.6f} != I(2.5y)={v2:.6f}")

for alpha_test in [0, 0.5, 1, 2, 3]:
    v1 = ge_alpha(y_si, alpha_test)
    v2 = ge_alpha(y_scaled, alpha_test)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"GE_{alpha_test}: I(y)={v1:.6f} != I(2.5y)={v2:.6f}")

for eps in [0.5, 1, 2]:
    v1 = atkinson(y_si, eps)
    v2 = atkinson(y_scaled, eps)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"A_{eps}: I(y)={v1:.6f} != I(2.5y)={v2:.6f}")

# Variance should NOT be scale invariant
v1 = variance(y_si)
v2 = variance(y_scaled)
if np.isclose(v1, v2, atol=0.01):
    findings.append(f"Variance IS scale invariant: V(y)={v1}, V(2.5y)={v2}")
if not np.isclose(v2, lam**2 * v1, atol=1e-10):
    findings.append(f"Var(lambda*y) != lambda^2 * Var(y): {v2} vs {lam**2 * v1}")

check("SI-1", "proposition", "Scale invariance verification",
      "CV, VL, Gini, GE, Atkinson, Zenga are scale-invariant; Variance is not",
      True, findings)

# ============================================================
# 12. TRANSLATION INVARIANCE OF VARIANCE
# ============================================================
findings = []
y_ti = np.array([1, 3, 5, 7, 9], dtype=float)
c_shift = 100.0
v1 = variance(y_ti)
v2 = variance(y_ti + c_shift)
if not np.isclose(v1, v2, atol=1e-10):
    findings.append(f"Var(y) = {v1}, Var(y+100) = {v2}")
check("TI-1", "proposition", "Variance translation invariance",
      "Var(y+c) = Var(y)", True, findings)

# ============================================================
# 13. POPULATION INVARIANCE FOR STANDARD INDICES
# ============================================================
findings = []
y_pi = np.array([1, 3, 5, 7], dtype=float)
y_pi_rep = np.tile(y_pi, 3)

for name, func in [("Var", variance), ("CV", cv), ("VL", variance_of_logs),
                    ("Gini", gini), ("Theil", theil), ("MLD", mld)]:
    v1 = func(y_pi)
    v2 = func(y_pi_rep)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"{name}: I(y) = {v1:.6f} != I(y^3) = {v2:.6f}")

for alpha_test in [0, 0.5, 1, 2]:
    v1 = ge_alpha(y_pi, alpha_test)
    v2 = ge_alpha(y_pi_rep, alpha_test)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"GE_{alpha_test}: {v1:.6f} != {v2:.6f}")

for eps in [0.5, 1, 2]:
    v1 = atkinson(y_pi, eps)
    v2 = atkinson(y_pi_rep, eps)
    if not np.isclose(v1, v2, atol=1e-10):
        findings.append(f"A_{eps}: {v1:.6f} != {v2:.6f}")

check("PI-1", "proposition", "Population invariance for standard indices",
      "Var, CV, VL, Gini, GE, Atkinson all population-invariant",
      True, findings)

# ============================================================
# 14. CROSS-COUNTRY NUMERICAL EXAMPLE (Section 7, lines 1290-1306)
# y_X = (2,4,10,16,18), y_Y = (1,6,10,13,20)
# Paper: G(X) ≈ 0.3520, G(Y) ≈ 0.3600
# GE_0(X) ≈ 0.2936, GE_0(Y) ≈ 0.3716
# GE_2(X) ≈ 0.2000, GE_2(Y) ≈ 0.2060
# ============================================================
y_X = [2, 4, 10, 16, 18]
y_Y = [1, 6, 10, 13, 20]

findings = []
g_X = gini(y_X)
g_Y = gini(y_Y)
ge0_X = ge_alpha(y_X, 0)
ge0_Y = ge_alpha(y_Y, 0)
ge2_X = ge_alpha(y_X, 2)
ge2_Y = ge_alpha(y_Y, 2)

if not np.isclose(g_X, 0.3520, atol=0.001):
    findings.append(f"G(X) = {g_X:.4f}, paper claims 0.3520")
if not np.isclose(g_Y, 0.3600, atol=0.001):
    findings.append(f"G(Y) = {g_Y:.4f}, paper claims 0.3600")
if not np.isclose(ge0_X, 0.2936, atol=0.001):
    findings.append(f"GE_0(X) = {ge0_X:.4f}, paper claims 0.2936")
if not np.isclose(ge0_Y, 0.3716, atol=0.001):
    findings.append(f"GE_0(Y) = {ge0_Y:.4f}, paper claims 0.3716")
if not np.isclose(ge2_X, 0.2000, atol=0.001):
    findings.append(f"GE_2(X) = {ge2_X:.4f}, paper claims 0.2000")
if not np.isclose(ge2_Y, 0.2060, atol=0.001):
    findings.append(f"GE_2(Y) = {ge2_Y:.4f}, paper claims 0.2060")
check("CC-2", "numerical", "Cross-country Gini vs GE example",
      "G(X)≈0.352, G(Y)≈0.360, GE_0(X)≈0.294, GE_0(Y)≈0.372, GE_2(X)≈0.200, GE_2(Y)≈0.206",
      True, findings)

# ============================================================
# 15. AXIOM INDEPENDENCE (Remark after Thm 2, lines 1078-1121)
# ============================================================

# (i) Anonymity: I_hat(y) = (1/(n*mu)) * sum(i * y_i)
# Paper: I_hat(1,3) = 7/4 = 1.75, I_hat(3,1) = 5/4 = 1.25
findings = []
y_anon = np.array([1, 3])
mu_anon = np.mean(y_anon)
n_anon = len(y_anon)
i_hat_13 = sum((i+1) * y_anon[i] for i in range(n_anon)) / (n_anon * mu_anon)
y_anon_perm = np.array([3, 1])
i_hat_31 = sum((i+1) * y_anon_perm[i] for i in range(n_anon)) / (n_anon * mu_anon)
if not np.isclose(i_hat_13, 1.75):
    findings.append(f"I_hat(1,3) = {i_hat_13:.4f}, paper claims 1.75")
if not np.isclose(i_hat_31, 1.25):
    findings.append(f"I_hat(3,1) = {i_hat_31:.4f}, paper claims 1.25")
check("IND-1", "remark", "Axiom independence: anonymity example",
      "I_hat(1,3)=1.75, I_hat(3,1)=1.25", True, findings)

# (ii) Normalization: I_tilde = GE_1 + 1
# At equality: I_tilde = 0 + 1 = 1 != 0
findings = []
ge1_eq = theil([5, 5, 5])
i_tilde = ge1_eq + 1
if not np.isclose(ge1_eq, 0, atol=1e-12):
    findings.append(f"GE_1(5,5,5) = {ge1_eq}, expected 0")
if not np.isclose(i_tilde, 1):
    findings.append(f"GE_1(5,5,5) + 1 = {i_tilde}, expected 1")
check("IND-2", "remark", "Axiom independence: normalization example",
      "GE_1 + 1 at equality equals 1, violating normalization", True, findings)

# (iv) Population invariance: I_n(y) = (n-1)/n * GE_1(y)
# Paper: GE_1(1,3) = 0.1308, I_2(1,3) = 0.0654
# GE_1(1,3,1,3) = 0.1308, I_4(1,3,1,3) = 3/4 * 0.1308 = 0.0981
findings = []
ge1_13 = theil([1, 3])
i2_13 = (2-1)/2 * ge1_13
ge1_1313 = theil([1, 3, 1, 3])
i4_1313 = (4-1)/4 * ge1_1313

if not np.isclose(ge1_13, 0.1308, atol=0.001):
    findings.append(f"GE_1(1,3) = {ge1_13:.4f}, paper claims 0.1308")
if not np.isclose(i2_13, 0.0654, atol=0.001):
    findings.append(f"I_2(1,3) = {i2_13:.4f}, paper claims 0.0654")
if not np.isclose(ge1_1313, 0.1308, atol=0.001):
    findings.append(f"GE_1(1,3,1,3) = {ge1_1313:.4f}, paper claims 0.1308")
if not np.isclose(i4_1313, 0.0981, atol=0.001):
    findings.append(f"I_4(1,3,1,3) = {i4_1313:.4f}, paper claims 0.0981")
if np.isclose(i2_13, i4_1313, atol=0.001):
    findings.append(f"I_2(1,3) ≈ I_4(1,3,1,3): population invariance not violated!")
check("IND-3", "remark", "Axiom independence: population invariance example",
      "I_n = (n-1)/n * GE_1: I_2(1,3)=0.0654 != I_4(1,3,1,3)=0.0981",
      True, findings)

# (v) Scale invariance: Variance
# Paper: Var(1,3) = 1, Var(2,6) = 4
findings = []
var_13 = variance([1, 3])
var_26 = variance([2, 6])
if not np.isclose(var_13, 1, atol=1e-10):
    findings.append(f"Var(1,3) = {var_13}, paper claims 1")
if not np.isclose(var_26, 4, atol=1e-10):
    findings.append(f"Var(2,6) = {var_26}, paper claims 4")
check("IND-4", "remark", "Axiom independence: scale invariance example",
      "Var(1,3)=1 != Var(2,6)=4", True, findings)

# (vi) Additive decomposability: Gini
# Already verified in GINI-DEC-1

# ============================================================
# 16. GE DECOMPOSITION FORMULA (Section 5.6, eq. ge_decomp)
# ============================================================
findings = []
# Use y = (1,3,5,7), A=(1,3), B=(5,7)
y_dec = np.array([1, 3, 5, 7], dtype=float)
y_A_dec = np.array([1, 3], dtype=float)
y_B_dec = np.array([5, 7], dtype=float)
mu_total = np.mean(y_dec)
mu_A = np.mean(y_A_dec)
mu_B = np.mean(y_B_dec)
n_A, n_B = 2, 2
n_total = 4

for alpha_test in [0, 1, 2]:
    ge_total = ge_alpha(y_dec, alpha_test)
    ge_A = ge_alpha(y_A_dec, alpha_test)
    ge_B = ge_alpha(y_B_dec, alpha_test)

    # Between-group: distribution where everyone gets their group mean
    y_between = np.array([mu_A, mu_A, mu_B, mu_B])
    ge_between = ge_alpha(y_between, alpha_test)

    # Weights: w_k = (n_k/n) * (mu_k/mu)^alpha
    w_A = (n_A / n_total) * (mu_A / mu_total) ** alpha_test
    w_B = (n_B / n_total) * (mu_B / mu_total) ** alpha_test

    predicted = w_A * ge_A + w_B * ge_B + ge_between

    if not np.isclose(ge_total, predicted, atol=1e-10):
        findings.append(f"GE_{alpha_test} decomposition: total={ge_total:.6f}, predicted={predicted:.6f}")

check("GE-DEC-1", "proposition", "GE additive decomposition",
      "GE_alpha(y) = sum w_k GE_alpha(y_k) + GE_between for alpha=0,1,2",
      True, findings)

# ============================================================
# 17. VARIANCE DECOMPOSITION (ANOVA)
# ============================================================
findings = []
var_total = variance(y_dec)
var_A_d = variance(y_A_dec)
var_B_d = variance(y_B_dec)
var_between = variance(y_between)
var_predicted = (n_A/n_total) * var_A_d + (n_B/n_total) * var_B_d + var_between
if not np.isclose(var_total, var_predicted, atol=1e-10):
    findings.append(f"Var decomposition: total={var_total:.4f}, predicted={var_predicted:.4f}")
check("VAR-DEC-1", "proposition", "Variance ANOVA decomposition",
      "Var(y) = sum (n_k/n) Var(y_k) + Var_between",
      True, findings)

# ============================================================
# 18. VL DETAILED COMPUTATION (lines 769-778)
# Check intermediate values
# ============================================================
findings = []
y_vl_det = np.array([1, 2, 50, 100], dtype=float)
ln_y = np.log(y_vl_det)
mean_ln = np.mean(ln_y)

# Paper claims: ln values are 0, 0.693, 3.912, 4.605
expected_ln = [0, 0.693, 3.912, 4.605]
for i, (comp, exp) in enumerate(zip(ln_y, expected_ln)):
    if not np.isclose(comp, exp, atol=0.001):
        findings.append(f"ln({y_vl_det[i]}) = {comp:.4f}, paper claims {exp}")

# Paper claims mean_ln = 2.303
if not np.isclose(mean_ln, 2.303, atol=0.001):
    findings.append(f"Mean of logs = {mean_ln:.4f}, paper claims 2.303")

# Paper squared deviations: 5.302, 2.590, 2.590, 5.302
sq_devs = (ln_y - mean_ln)**2
expected_sq = [5.302, 2.590, 2.590, 5.302]
for i, (comp, exp) in enumerate(zip(sq_devs, expected_sq)):
    if not np.isclose(comp, exp, atol=0.01):
        findings.append(f"(ln y_{i+1} - mean)^2 = {comp:.4f}, paper claims {exp}")

check("VL-2", "numerical", "VL counterexample intermediate values",
      "Verify ln values, mean of logs, and squared deviations", True, findings)

# After transfer
y_vl_aft = np.array([1, 2, 60, 90], dtype=float)
ln_y_aft = np.log(y_vl_aft)
mean_ln_aft = np.mean(ln_y_aft)

# Paper: mean_ln' = 2.322
if not np.isclose(mean_ln_aft, 2.322, atol=0.001):
    findings.append(f"Mean of logs after = {mean_ln_aft:.4f}, paper claims 2.322")

# Paper squared deviations after: 5.391, 2.653, 3.142, 4.744
sq_devs_aft = (ln_y_aft - mean_ln_aft)**2
expected_sq_aft = [5.391, 2.653, 3.142, 4.744]
for i, (comp, exp) in enumerate(zip(sq_devs_aft, expected_sq_aft)):
    if not np.isclose(comp, exp, atol=0.01):
        findings.append(f"After: (ln y'_{i+1} - mean)^2 = {comp:.4f}, paper claims {exp}")

check("VL-3", "numerical", "VL counterexample after-transfer intermediate values",
      "Verify post-transfer ln values and squared deviations", True, findings)

# ============================================================
# 19. NORMALIZATION CHECKS
# All indices should return 0 at perfect equality (except percentile ratio = 1)
# ============================================================
findings = []
y_eq = np.array([5, 5, 5, 5, 5], dtype=float)
for name, func in [("Var", variance), ("CV", cv), ("VL", variance_of_logs),
                    ("Gini", gini), ("Theil", theil), ("MLD", mld),
                    ("Zenga", zenga)]:
    val = func(y_eq)
    if not np.isclose(val, 0, atol=1e-12):
        findings.append(f"{name}(5,5,5,5,5) = {val}, expected 0")

for alpha_test in [-1, 0, 0.5, 1, 2, 3]:
    val = ge_alpha(y_eq, alpha_test)
    if not np.isclose(val, 0, atol=1e-12):
        findings.append(f"GE_{alpha_test}(5,5,5,5,5) = {val}, expected 0")

for eps in [0.5, 1, 2]:
    val = atkinson(y_eq, eps)
    if not np.isclose(val, 0, atol=1e-12):
        findings.append(f"A_{eps}(5,5,5,5,5) = {val}, expected 0")

check("NORM-1", "property", "Normalization at equality",
      "All indices return 0 at perfect equality", True, findings)

# ============================================================
# 20. Variance scale property: Var(lambda*y) = lambda^2 * Var(y)
# ============================================================
findings = []
y_vs = np.array([2, 5, 8, 11], dtype=float)
lam = 3.0
v1 = variance(y_vs)
v2 = variance(lam * y_vs)
if not np.isclose(v2, lam**2 * v1, atol=1e-10):
    findings.append(f"Var(3y) = {v2}, expected 9*{v1} = {9*v1}")
check("VAR-SC-1", "proposition", "Variance homogeneity of degree 2",
      "Var(lambda*y) = lambda^2 * Var(y)", True, findings)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print(f"AUDIT COMPLETE: {sum(1 for r in results if r['passes'])}/{len(results)} checks passed")
if all_pass:
    print("ALL CHECKS PASSED - No discrepancies found.")
else:
    print("DISCREPANCIES FOUND:")
    for r in results:
        if not r["passes"]:
            print(f"  FAIL: {r['id']} - {r['label']}")
            for f in r["findings"]:
                print(f"    -> {f}")

# Write audit JSON
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proof-audit.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAudit results written to {output_path}")
