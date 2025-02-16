NUM_TEST_RETRY=4

#
# Installation
#

.PHONY: setup
setup:
	pip install -U uv

.PHONY: install
install:
	uv sync --all-extras

#
# linter/formatter/typecheck
#

.PHONY: lint
lint: install
	uv run ruff check --output-format=github .
	uv run nbqa ruff notebooks

.PHONY: format
format: install
	uv run ruff format --check --diff .

.PHONY: typecheck
typecheck: install
	uv run mypy .jupytext
	uv run nbqa mypy notebooks

.PHONY: apply-formatter
apply-formatter: install
	uv run ruff check --select I --fix notebooks
	uv run ruff format

#
# jupytext
#
.PHONY: sync-script-to-notebook
sync-script-to-notebook: install
	uv run jupytext --sync scripts/*.py

.PHONY: sync-notebook-to-script
sync-notebook-to-script: install
	uv run jupytext --sync notebooks/*.ipynb

#
# LaTeX code generation
#
.PHONY: generate-latex-code
generate-latex-code: install
	uv run scripts/convert_cell_to_latex.py --notebooks-dir notebooks --output-dir latex/code

#
# Testing
#

.PHONY: test-notebooks
test-notebooks: install
	uv run pytest --force-flaky --max-runs=$(NUM_TEST_RETRY) --nbmake notebooks/*.ipynb
