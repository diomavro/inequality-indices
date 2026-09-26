# One gate, run identically by `make check` locally and by CI.
# PY strips PYTHONPATH so a user-level site-packages leak cannot shadow the
# pinned versions (see code/requirements.txt).
PY ?= env -u PYTHONPATH python3
# Fixed PDF metadata timestamps (matplotlib and pdfTeX honour it) so rebuilding
# figures and the paper does not dirty the tree (the manuscript's date is fixed
# text, not \today, for the same reason).
export SOURCE_DATE_EPOCH := 1767225600
# Files held to the lint gate. The older verify_*.py scripts are not yet
# clean; AGENT_TODO.md tracks bringing them in one at a time.
LINTED := code/pip_panel.py code/pip_disagreement.py code/make_pip_figures.py code/verify_corrections.py

.PHONY: help setup lint verify reproduce figures paper conflicts check

help:  ## List targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## Install pinned Python dependencies
	$(PY) -m pip install -r code/requirements.txt

lint:  ## ruff check + format check on the gated files
	$(PY) -m ruff check $(LINTED)
	$(PY) -m ruff format --check $(LINTED)

verify:  ## Assert every counterexample and simulation claim the paper states
	cd code && $(PY) -W ignore verify_corrections.py

reproduce:  ## Rebuild the PIP panel and fail if any number quoted in the paper changes
	cd code && $(PY) -W ignore pip_panel.py && $(PY) -W ignore pip_disagreement.py > /dev/null
	git diff --exit-code -- paper/pip_macros.tex

figures:  ## Regenerate all figures and run the overlap/leak audit (exits 1 on issues)
	$(PY) -W ignore figures/audit_figures.py

paper:  ## Compile the manuscript; fail on undefined references or citations
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error inequality_indices.tex > /dev/null
	test -f paper/inequality_indices.log
	! grep -E "undefined|Rerun to get" paper/inequality_indices.log

conflicts:  ## Fail on leftover merge-conflict markers
	! git grep -nE '^(<<<<<<<|>>>>>>>)( |$$)' -- . ':!*.pdf' ':!data/'

check: conflicts lint verify reproduce figures paper  ## Everything CI runs
