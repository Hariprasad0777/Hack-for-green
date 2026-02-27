# OffroadTerrain AI - Professional Task Runner
# Usage: make <target>

.PHONY: help setup train train-balanced eval eval-calibrated stream stream-simulator clean test lint

help:
	@echo "OffroadTerrain AI - Makefile"
	@echo "Available targets:"
	@echo "  setup            Install all dependencies"
	@echo "  train            Start the model training pipeline"
	@echo "  train-balanced   Start training with class-balanced weights"
	@echo "  eval             Evaluate the model on validation set"
	@echo "  eval-calibrated  Evaluate the model using logit bias calibration"
	@echo "  stream           Launch the Pathway reactive engine"
	@echo "  stream-simulator Simulate a live data feed"
	@echo "  test             Run the unit test suite"
	@echo "  lint             Perform code quality and style checks"
	@echo "  clean            Remove temporary files and logs"

setup:
	pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

train:
	offroad-train

eval:
	offroad-eval

eval-calibrated:
	offroad-eval --bias-file weights/optimal_biases.npy

stream:
	offroad-stream

stream-simulator:
	python src/offroad_ai/pipeline/stream_simulator.py --interval 1.5

test:
	pytest tests/ --cov=offroad_ai --cov-report=term-missing

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	mypy src/

clean:
	powershell -Command "Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Remove-Item .pytest_cache -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -Command "Remove-Item .coverage -Force -ErrorAction SilentlyContinue"
	powershell -Command "Remove-Item reports/*.log -Force -ErrorAction SilentlyContinue"
