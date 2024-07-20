# Registry Pattern

[![Build Status](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml/badge.svg)](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/adnanhd/registry-pattern/badge.svg)](https://coveralls.io/github/adnanhd/registry-pattern)



## Continuous Integration

This project uses GitHub Actions for continuous integration.
The CI pipeline is configured to run the following checks on every push and pull request to the `main` branch:

- **Tests**: Runs tests using `pytest` to ensure all tests pass.
- **Type Checking**: Uses `pyright` for static type checking.
- **Code Formatting**: Checks code formatting with `black`.
- **Linting**: Uses `flake8` to lint the code.

## Running Checks Locally

To run the checks locally, you can use the following commands:

```bash
# Install dependencies
pip install pytest black flake8 pyright

# Run tests
pytest -vv --cov

# Run type checking
pyright

# Check code format
black --check .

# Lint code
flake8 .
```
