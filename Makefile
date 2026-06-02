.PHONY: install install-dev lint format typecheck test check run gui benchmark benchmark-update benchmark-generate clean

PYTHON ?= python3.13
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet .

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"
	$(PY) -m pre_commit install

lint:
	$(VENV)/bin/ruff check src/

format:
	$(VENV)/bin/ruff format src/
	$(VENV)/bin/ruff check --fix src/

typecheck:
	$(VENV)/bin/mypy src/

test:
	$(VENV)/bin/pytest -q

check: lint typecheck test

run:
	./pipeline.sh --skip-download

gui:
	$(VENV)/bin/image-cropper

# --- benchmark ---
# CI check: regenerate synthetic goldens, compare to committed baseline.json
benchmark:
	$(PY) benchmarks/create_golden.py
	$(PY) -m image_cropper.pipeline.benchmark benchmarks/golden/ --compare benchmarks/baseline.json

# After running the full pipeline on real images, promote results as new baseline.
# Usage: make benchmark-update INPUT=output/final/
benchmark-update:
	$(PY) -m image_cropper.pipeline.benchmark $(INPUT) --update-baseline benchmarks/baseline.json

# Regenerate committed synthetic goldens + baseline from scratch.
benchmark-generate:
	$(PY) benchmarks/create_golden.py

clean:
	rm -rf tmp_pipeline/ output/ build/ dist/ *.egg-info src/*.egg-info
