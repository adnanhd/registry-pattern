#!/bin/bash


# Install dependencies
pip install pytest pytest-cov black flake8 pyright

# Run tests
pytest -vv --cov

# Run type checking
pyright

# Check code format
black --check .

# Lint code
flake8 .
