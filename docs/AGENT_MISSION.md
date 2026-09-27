# Agent mission

This repository holds an academic paper, *What Should We Ask of an Inequality
Measure? Axioms, Proofs, and How Often They Bind in Survey Data* (Diomides and
Noemie Mavroyiannis), and the code that produces every number in it. The weekly
agent (`.github/workflows/agent.yml`) and the PR reviewer (`ci.yml`,
`agent-review`) both read this file first.

## What the agent is for

Small, verifiable improvements that make the paper more correct and the results
more reproducible: robustness checks, code hygiene, tighter assertions, and
fixes to anything a check shows to be wrong. One item per run, smallest diff
that does it. Proposing nothing is a valid outcome.

## Invariants (a PR that breaks one is a DEFECT)

1. **No hand-typed data numbers.** Every number in the manuscript that comes
   from data is a macro in `paper/pip_macros.tex`, written by
   `code/pip_disagreement.py`. To change a number, change the code.
   `make reproduce` fails if the committed macros differ from a fresh run.
2. **Every mathematical claim is backed.** A property that holds needs a proof
   in the paper; a property that fails needs a counterexample in the paper *and*
   an `assert` in `code/verify_corrections.py`. A random "verification" must
   draw from the full domain the property quantifies over, add a targeted search
   (small integer vectors, ties, boundaries), and run the same test on an index
   known to pass and one known to fail as controls. A test that samples only
   favourable cases is not evidence (the Zenga DTS claim was "verified" that way
   and was false).
3. **Theorems are checked against the data.** `pip_disagreement.py` asserts that
   no transfer-principle index moves against a Lorenz-dominance ranking and that
   ordinal twins (CV/GE2, A1/MLD, A2/GE-1, A0.5/GE0.5) move identically. Never
   weaken these assertions to make a run pass.
4. **`data/raw/` is immutable.** It is the World Bank PIP percentile file and
   survey table in 2017 PPP (World Bank, CC BY 4.0). The API's default vintage is 2021 PPP
   and does not match the bins.
5. **`make check` is green before any PR.** CI runs the same target.
6. **The manuscript reads as a paper.** No process language ("referee",
   "reviewer", "this round", "earlier draft", "as requested"), no sentence that
   reads as a reply to a reviewer.
7. **Code quality.** No `Any` annotations; no bare or blanket `except`; new
   Python files join the `LINTED` list in the Makefile.
8. **No AI co-author in git history.** Commits and PR bodies carry no
   `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" line;
   the authors of this paper are the two people named above.

## Reserved for the authors (never decide these; add to HUMAN_TODO.md instead)

The paper's claims of novelty, its title and abstract framing, its structure
(what goes to an appendix), the target journal, licensing, co-author sign-off,
and anything that needs an account, a key, or new data sources.

## The constitution (human-curated)

As in tail-lab, agent PRs merge automatically once the whole CI run is green:
`make check`, the co-author trailer check, `constitution-guard`, and the
adversarial reviewer (a DEFECT verdict blocks). `constitution-guard` fails an
agent PR that edits the constitution: `.github/`, this file, `README.md`,
`data/raw/`, any makefile, latexmkrc, ruff or pyproject config, or
`code/requirements.txt`, or that removes an `assert` from
`code/verify_corrections.py` or `code/pip_disagreement.py`. Such changes need a
human. Everything else, including the manuscript, code, and figures, is the
agent's to change within the invariants above.

## Backlogs

- `AGENT_TODO.md`: the agent's queue. Work the first unchecked item it can
  finish alone. Close items with `**Done YYYY-MM-DD**: <what shipped>` or
  `**Moot YYYY-MM-DD**: <why>`, keeping the original text.
- `HUMAN_TODO.md`: the authors' queue. The agent only appends to it and never
  edits, reorders, or ticks existing items.
