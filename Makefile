.PHONY: download data prepare dvc-init dvc-repro dvc-push test

PYTHON ?= python

download:
	$(PYTHON) scripts/download_data.py

prepare:
	$(PYTHON) -m src.data.prepare_data

featurize:
	$(PYTHON) -m src.features.build_features

baseline:
	$(PYTHON) -m src.models.baseline

train:
	$(PYTHON) -m src.models.train

mlflow-ui:
	mlflow ui --port 5000

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
