# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Academic paper: "What Should We Ask of an Inequality Measure? Axioms, Proofs, and How Often They Bind in Survey Data" by Diomides and Noemie Mavroyiannis. Evaluates eight inequality indices (Var, CV, VL, Gini, GE, Atkinson, percentile ratios, three Zenga versions Z / Z_N / Z*) against eleven properties, then tests how often the axioms bind on World Bank PIP percentile data (1,955 comparable survey pairs, 143 countries).

**Corrections made 2026-09-26 (do not regress):** the old "translation-invariant indices satisfy at most nine properties" theorem was FALSE (the exponential absolute index E_kappa satisfies ten); Zenga fails DTS and subgroup consistency (old "satisfies"/"conjectured" claims were false); Zenga's own finite index Z_N is replication-invariant but discontinuous at ties and violates the transfer principle; Lorenz consistency is defined strictly. Every counterexample is asserted in `code/verify_corrections.py`. The old `verify_zenga.py` / `verify_integral_zenga.py` DTS/SC "PASS" results are superseded (they sample only favourable configurations).

## Build Commands

### Compile the paper
```bash
cd paper && pdflatex inequality_indices.tex && bibtex inequality_indices && pdflatex inequality_indices.tex && pdflatex inequality_indices.tex
```

### Generate all figures (outputs to `figures/` as PDFs)
```bash
cd code && python3 generate_figures.py
```

### Run property verification (prints Table 1 — axiomatic properties matrix)
```bash
cd code && python3 verify_properties.py
```

### Install Python dependencies
```bash
pip install -r code/requirements.txt
```

### Rebuild the empirical section (PIP data; every number in Section 7 is a macro)
```bash
cd code && python3 pip_panel.py && python3 pip_disagreement.py && python3 make_pip_figures.py
python3 ../figures/audit_figures.py   # figure overlap/leak audit (runs both figure scripts)
python3 verify_corrections.py         # asserts every counterexample the paper states
```
Raw inputs: `data/raw/pip_world_100bin.csv` (World Bank PIP percentiles, 2017 PPP) and `data/raw/pip_summary_ppp2017.csv` (PIP API, `ppp_version=2017`; the default API vintage is 2021 PPP and does NOT match the bins). `pip_disagreement.py` writes `paper/pip_macros.tex` and fails loudly if a transfer-principle index moves against a Lorenz-dominance ranking or ordinal twins disagree.

## Architecture

- **`code/inequality_indices.py`** — Core library. Implements all inequality index functions (`gini`, `theil`, `mld`, `ge_alpha`, `atkinson`, `cv`, `variance`, `variance_of_logs`, `percentile_ratio`). Every function takes a 1-D numpy array and returns a float. This is imported by both scripts below.
- **`code/generate_figures.py`** — Produces three publication-quality PDF figures: Lorenz curves, transfer sensitivity comparison, and VL counter-example. Outputs to `figures/`. Imports from `inequality_indices.py` using a bare import (must run from `code/` directory).
- **`code/verify_properties.py`** — Numerically tests each index against the properties and prints a formatted results table. Uses concrete distributions (`Y`, `Y_LARGE`, `Y_DTS`) as test cases.
- **`paper/inequality_indices.tex`** — Main manuscript. Uses `plainnat` bibliography style, `natbib` citations, `booktabs` tables, `amsthm` environments. Custom commands: `\bfy`, `\bfz`, `\R`, `\N`, `\D`.
- **`paper/references.bib`** — BibTeX references.
- **`figures/`** — Generated PDF figures tracked in git (not gitignored).

## Important Notes

- Python scripts use **bare imports** (`from inequality_indices import ...`), so they must be run from inside `code/`.
- Dependencies are pinned in `code/requirements.txt` (numpy, pandas, scipy, matplotlib, ruff).
- `make check` is the whole gate (CI runs it too). Backlogs: `AGENT_TODO.md` (weekly agent) and `HUMAN_TODO.md` (authors); invariants in `docs/AGENT_MISSION.md`.
- Simulated referee reports, `pipeline-state.json`, `proof-audit.json` and the stale `submission/` package are kept locally but gitignored (not published).
