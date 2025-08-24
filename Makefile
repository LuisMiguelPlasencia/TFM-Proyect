.PHONY: help install install-dev test lint format clean data notebooks
.DEFAULT_GOAL := help

PYTHON := python3
PIP := pip3
VENV := venv

help: ## Show this help message
	@echo "Real Estate Forecasting Spain - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment Management
install: ## Install project dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

setup-env: ## Setup complete development environment
	$(PYTHON) -m venv $(VENV)
	source $(VENV)/bin/activate && $(MAKE) install-dev

# Code Quality
lint: ## Run linting checks
	black --check src notebooks
	isort --check-only src notebooks
	flake8 src

format: ## Format code with black and isort
	black src notebooks
	isort src notebooks

test: ## Run tests
	pytest tests/ -v

# Data Pipeline
data-download: ## Download raw datasets
	dvc pull

data-process: ## Process raw data
	$(PYTHON) -m src.data.make_dataset

# Notebooks
notebooks-run: ## Execute all notebooks
	jupyter nbconvert --execute --inplace notebooks/*.ipynb

notebooks-clean: ## Clean notebook outputs
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

notebooks-html: ## Convert notebooks to HTML
	mkdir -p reports/notebooks
	jupyter nbconvert --to html notebooks/*.ipynb --output-dir reports/notebooks

# MLflow and Experiments
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

train: ## Train models
	$(PYTHON) -m src.models.train_model

evaluate: ## Evaluate models
	$(PYTHON) -m src.models.evaluate_model

# Utilities
clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/

# DVC Operations  
dvc-add: ## Add files to DVC tracking
	dvc add data/raw/
	dvc add data/processed/
	dvc add models/

dvc-push: ## Push data to remote storage
	dvc push

dvc-status: ## Check DVC status
	dvc status