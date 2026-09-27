# AGENT_TODO

The weekly agent's queue (see `docs/AGENT_MISSION.md`). Work the first unchecked
item that needs no account, key, or author decision. Close items in place with
`**Done YYYY-MM-DD**: <what shipped>` or `**Moot YYYY-MM-DD**: <why>`.

## Robustness of the empirical section (2026-09-26)

- [x] **Done 2026-09-27**: Add a tolerance band to the Lorenz-crossing classification.
      Crossings are now counted with a 1e-9 tolerance on 99 percentile
      ordinates, and many sit in the tails where the curves nearly touch
      (62% cross on percentiles, 36% on deciles). Add macros for the crossing
      share and the seven-index agreement rate when ordinate gaps below
      0.001 and 0.002 are treated as ties, and one sentence reporting them in
      the empirical section (`sec:empirics`). The dominance-direction assertion must keep passing.
      Added `PipShareCrossTie001/002` and `PipTpAgreeCrossTie001/002` macros
      (crossing share falls to 29%/16%, and among those coarser crossings the
      seven indices agree in only 27%/20% — near-tangencies were not driving
      the disagreement rate). The strict dominance-direction assertion (TOL=1e-9)
      is untouched and still passes.
- [ ] **Report the headline rates separately for income and consumption surveys.**
      `pip_pairs.csv` has `welfare_type`. Add macros for the crossing share,
      `PipTpAgreeCross` and `PipRelAbsOppMaterialDom` by welfare type, and one
      sentence saying whether the pattern holds in both.
- [ ] **Add an exponential-index (E_kappa) robustness row.**
      The Axiomatic Optimality theorem uses E_kappa, the empirics use the absolute Gini. Compute
      E_kappa with kappa scaled to each pair's first-survey mean (kappa = c/mu,
      c in {0.5, 1}) and report its agreement with the absolute Gini as macros,
      alongside `PipKolmAgree`.

## Code hygiene

- [ ] **Bring `code/verify_properties.py` under the lint gate.**
      Fix its ruff findings without changing what it prints, add it to `LINTED`
      in the Makefile. (Makefile is a guarded path: expect a human merge.)
- [ ] **Retire or fix the superseded DTS / subgroup tests in `verify_zenga.py`
      and `verify_integral_zenga.py`.**
      Both report PASS for properties the Zenga indices fail, because they
      sample only favourable configurations (see the NOTE at the top of each).
      Either delete those two test blocks or rewrite them to sample the full
      domain so they fail as they should; do not leave a misleading PASS.
- [ ] **Make `percentile_ratio` in `code/inequality_indices.py` follow eq. (pq).**
      The paper defines R_{p/q} by order statistics y_(ceil(pn)) / y_(ceil(qn));
      the library interpolates with `np.percentile`. Add an `order_statistic`
      keyword defaulting to the paper's definition, update callers, and check
      `verify_properties.py` output is unchanged or explain the change in the PR.
