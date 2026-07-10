.PHONY: download data prepare dvc-init dvc-repro dvc-push test

PYTHON ?= python

download:
	$(PYTHON) scripts/download_data.py

prepare:
	$(PYTHON) src/data/prepare_data.py

dvc-init:
	dvc init -f
	dvc add data/raw/train.csv data/raw/test.csv data/raw/data_description.txt
	dvc remote add -d localstorage .dvc-storage

dvc-repro:
	dvc repro

dvc-push:
	dvc push

test:
	pytest tests/ -v
