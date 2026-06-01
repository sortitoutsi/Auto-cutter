.PHONY: install install-dev lint format typecheck check run benchmark benchmark-update benchmark-generate clean

PYTHON ?= python3.13
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt

install-dev: install
	$(PIP) install --quiet -r requirements-dev.txt
	$(PY) -m pre_commit install

lint:
	$(VENV)/bin/ruff check scripts/

format:
	$(VENV)/bin/ruff format scripts/
	$(VENV)/bin/ruff check --fix scripts/

typecheck:
	$(VENV)/bin/mypy scripts/

check: lint typecheck

run:
	./pipeline.sh --skip-download

# --- benchmark ---
# CI check: regenerate synthetic goldens, compare to committed baseline.json
benchmark:
	$(PY) benchmarks/create_golden.py
	$(PY) scripts/benchmark.py benchmarks/golden/ --compare benchmarks/baseline.json

# After running the full pipeline on real images, promote results as new baseline.
# Usage: make benchmark-update INPUT=output/final/
benchmark-update:
	$(PY) scripts/benchmark.py $(INPUT) --update-baseline benchmarks/baseline.json

# Regenerate committed synthetic goldens + baseline from scratch (run after
# changing create_golden.py, then commit both benchmarks/golden/ and baseline.json).
benchmark-generate:
	$(PY) benchmarks/create_golden.py

clean:
	rm -rf tmp_pipeline/ output/
