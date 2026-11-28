PYTHON ?= python
CONFIG ?= configs/default.yaml
ARGS = --config $(CONFIG)

.PHONY: install install-all synthetic-data prepare features baselines tune mlp plots pipeline test lint format clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,dl,notebook]"

synthetic-data:
	$(PYTHON) tools/make_synthetic_workbook.py

prepare:
	$(PYTHON) scripts/data/prepare_data.py $(ARGS)

features:
	$(PYTHON) scripts/features/build_features.py $(ARGS)

baselines:
	$(PYTHON) scripts/models/train_baselines.py $(ARGS)

tune:
	$(PYTHON) scripts/models/tune_models.py $(ARGS)

mlp:
	$(PYTHON) scripts/models/train_mlp.py $(ARGS)

plots:
	$(PYTHON) scripts/visualization/make_plots.py $(ARGS)

pipeline:
	$(PYTHON) scripts/run_pipeline.py $(ARGS)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

clean:
	rm -rf data/interim data/processed models reports
