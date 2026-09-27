# HUMAN_TODO

The authors' queue (Dio, Noemie). The agent only appends here; it never edits,
reorders, or ticks items. Close items as `**Done YYYY-MM-DD (who)**: ...`.

## GitHub Actions and billing (2026-09-26)

- [x] **Done 2026-09-26 (Claude)**: CI runs now that the repo is public (run
      36268777116, `check` green in 1m38s on GitHub's runners). If a job ever dies
      within seconds again, check Settings → Billing & plans first.
## Before making the repository public (2026-09-26)

- [x] **Done 2026-09-26 (Dio, Noemie)**: both authors agreed to publish the
      working paper and code.
- [x] **Done 2026-09-26 (Claude, at Dio's request)**: history squashed into a
      single root commit; the old history (stale submission PDFs, simulated
      referee reports, pipeline state) was never published and is kept only in
      the local tag `backup/pre-public-history`.
- [x] **Done 2026-09-26 (Claude, at Dio's request)**: paper text and figures
      licensed CC BY 4.0 (`paper/LICENSE`); code stays MIT.
- [x] **Done 2026-09-26 (Claude)**: data licence confirmed as CC BY 4.0 on the
      World Bank Data Catalog page for dataset 0063646; stated in `data/README.md`.
- [x] **Done 2026-09-26 (Claude, at Dio's request)**: repository made public
      with a description.
## Turn on the automation (2026-09-26)

- [ ] **Install the Claude GitHub App on this repository** (github.com/apps/claude,
      "Only select repositories"). Agent PRs opened by the app trigger CI; PRs
      opened with the default workflow token would not.
- [ ] **Add the `CLAUDE_CODE_OAUTH_TOKEN` secret.** Run `claude setup-token`
      locally, then `gh secret set CLAUDE_CODE_OAUTH_TOKEN --repo diomavro/inequality-indices`
      and paste it. (tail-lab and quizkit use the same kind of token.)
- [x] **Done 2026-09-26 (Claude)**: Actions workflow permissions set to read,
      and Actions may create and approve pull requests.
- [x] **Done 2026-09-26 (Claude)**: ruleset `protect-main` on `main` (pull
      request required, `check` status required, no force pushes or deletion);
      secret scanning with push protection and private vulnerability reporting
      enabled. Changes to `main` now go through pull requests.
- [ ] **Decide how agent work appears in the history.** The authors' policy is
      no AI co-author trailer, and CI rejects one, but squash-merged agent PRs are
      still *authored* by `claude[bot]`. Options: accept that (the history then
      says truthfully which commits a bot wrote), or keep the agent to opening
      PRs and merge them yourself, re-authoring as you see fit. Automerge only lands backlog and
      documentation changes; everything touching code, figures, or the paper
      waits for you.
- [ ] **Run the agent once by hand and read its PR before trusting the schedule:**
      `gh workflow run agent.yml --repo diomavro/inequality-indices`. The
      schedule is Mondays 07:00 UTC.
- [ ] **Set GitHub usage alerts** (Billing → spending limits) so a runaway
      workflow cannot eat the Actions minutes shared with tail-lab.

## Research decisions (2026-09-26, from internal review)

- [x] **Done 2026-09-26 (Claude, at Dio's request)**: restructured in PR #1:
      property-by-property proofs moved to Appendix A, long Section 6 proofs to
      Appendix B, main text 36 → 28 pages, no content removed.
- [x] **Done 2026-09-26 (Claude, at Dio's request)**: target journal chosen:
      Journal of Economic Inequality (no submission fee, no length cap,
      single-anonymous review, preprints allowed); fallback Review of Income
      and Wealth (abstract ≤150 words, blind manuscript, recheck its fee).
- [ ] **Statistical inference.** The main weakness is the absence of standard
      errors. Formal Lorenz-dominance tests need microdata (LIS registration).
- [x] **Done 2026-09-26 (Claude)**: JEI package built locally in `submission/`
      (not in the repo): manuscript PDF, LaTeX source zip, cover letter. The
      paper now has a Statements and Declarations section.

## Submitting to the Journal of Economic Inequality (2026-09-26)

- [x] **Done 2026-09-27 (Claude, from Dio)**: Noemie's affiliation (Corvinus
      University of Budapest) added to the author block.
- [ ] **Decide the AI-use statement.** Springer asks authors to document use of
      AI tools beyond copy-editing (in the methods section). Much of this
      revision, the data pipeline, and the checks were produced with Claude; a
      sentence in the declarations or the introduction is likely required.
      Suggested: "The authors used an AI assistant (Claude, Anthropic) to help
      write analysis code, check proofs numerically, and edit the text; the
      authors verified all results and take full responsibility for the paper."
- [ ] **Archive a release on Zenodo** (zenodo.org → GitHub → enable the repo,
      then publish a GitHub release) and add the DOI to the data-availability
      statement. JEI requires a repository deposit on acceptance; a Zenodo DOI
      is the safe choice.
- [ ] **Submit** through JEI's Springer submission system with
      `submission/main_submission.pdf`, `submission/source.zip`, and
      `submission/cover_letter.pdf`; declare funding/competing interests in the
      form as in the paper.
