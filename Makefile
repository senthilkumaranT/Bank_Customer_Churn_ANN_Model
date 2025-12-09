.PHONY: install install-dev test lint format clean run help

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code with Black"
	@echo "  make run          - Run Streamlit app"
	@echo "  make clean        - Clean cache files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black src/ tests/

format-check:
	black --check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

run:
	streamlit run src/app.py

verify-imports:
	python -c "import sys; sys.path.insert(0, 'src'); import app; print('✓ Imports successful')"

check-models:
	@echo "Checking model files..."
	@test -f models/model.h5 || (echo "Error: model.h5 not found" && exit 1)
	@test -f models/scaler.pkl || (echo "Error: scaler.pkl not found" && exit 1)
	@test -f models/label_encoder_gender.pkl || (echo "Error: label_encoder_gender.pkl not found" && exit 1)
	@test -f models/onehot_encoder_geo.pkl || (echo "Error: onehot_encoder_geo.pkl not found" && exit 1)
	@echo "✓ All model files present"

ci: install-dev format-check lint test check-models verify-imports
	@echo "✓ CI checks passed"

