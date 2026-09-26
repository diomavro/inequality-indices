# What Should We Ask of an Inequality Measure?
### Axioms, Proofs, and How Often They Bind in Survey Data

**Authors:** Diomides Mavroyiannis and Noemie Mavroyiannis
**Status:** working paper, not yet submitted. Please cite the paper if you use it.

[![CI](https://github.com/diomavro/inequality-indices/actions/workflows/ci.yml/badge.svg)](https://github.com/diomavro/inequality-indices/actions/workflows/ci.yml)

We evaluate eight inequality indices against eleven axiomatic properties, with a
proof or a counterexample for every index–property pair, and then ask how often
the properties decide anything in data. On the theory side, the Zenga index fails
diminishing transfer sensitivity and subgroup consistency, and the exponential
absolute index satisfies ten of the eleven properties, as the Generalized Entropy
family does. On the data side, across 1,955 consecutive comparable surveys from
143 countries in the World Bank's Poverty and Inequality Platform, Lorenz curves
compared at every percentile cross in 62% of pairs (36% on deciles), relative
indices then often disagree about small changes,
and Lorenz dominance, which settles every relative index that satisfies the
transfer principle, does not settle whether *absolute* inequality rose.

The compiled paper is [`paper/inequality_indices.pdf`](paper/inequality_indices.pdf).

## Reproduce everything

Requires Python 3.12 and TeX Live (`pdflatex`, `latexmk`).

```bash
make setup     # pinned Python dependencies
make check     # the full gate CI runs
```

`make check` runs, in order:

| Target | What it guarantees |
|---|---|
| `lint` | ruff on the analysis pipeline |
| `verify` | every counterexample and simulation result stated in the paper is asserted (`code/verify_corrections.py`) |
| `reproduce` | rebuilds the PIP panel from `data/raw/` and fails if any number quoted in the paper would change (`paper/pip_macros.tex`) |
| `figures` | regenerates all figures and fails on overlapping or clipped labels |
| `paper` | compiles the manuscript and fails on undefined references |

The data pipeline also asserts two theorems against the data: no index satisfying
the transfer principle moves against a Lorenz-dominance ranking, and indices that
rank distributions identically always move together.

## Layout

```
paper/     manuscript (.tex, .bib), generated macros (pip_macros.tex), compiled PDF
code/      index library, PIP pipeline, figure scripts, verification scripts
data/raw/  World Bank PIP inputs (see data/README.md)
data/processed/  panel and survey pairs built by code/pip_panel.py, pip_disagreement.py
figures/   generated figures and the figure audit
docs/      AGENT_MISSION.md: invariants for automated contributions
```

## Automation

CI runs `make check` on every pull request and every push to `main`. A weekly agent
(`.github/workflows/agent.yml`) takes the top item from `AGENT_TODO.md`, opens a
pull request, and an automated reviewer checks it against
`docs/AGENT_MISSION.md`. Any PR that changes code, figures, or the
manuscript is merged by an author; only backlog and documentation updates merge
automatically. The authors' own queue is `HUMAN_TODO.md`.

## Licence

Code: MIT (see `LICENSE`). Data: World Bank, CC BY 4.0 (see `data/README.md`).
Manuscript text and figures: CC BY 4.0 (see `paper/LICENSE`).

JEL: D63, D31, I32.
