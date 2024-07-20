# Registry Pattern

![Python 3.12](https://github.com/adnanhd/registry-pattern/actions/workflows/ci.yml/badge.svg?branch=main&event=push&name=build&matrix.python-version=3.12)
![Python 3.11](https://github.com/adnanhd/registry-pattern/actions/workflows/ci.yml/badge.svg?branch=main&event=push&name=build&matrix.python-version=3.11)
![Python 3.10](https://github.com/adnanhd/registry-pattern/actions/workflows/ci.yml/badge.svg?branch=main&event=push&name=build&matrix.python-version=3.10)
![Python 3.9](https://github.com/adnanhd/registry-pattern/actions/workflows/ci.yml/badge.svg?branch=main&event=push&name=build&matrix.python-version=3.9)
![Python 3.8](https://github.com/adnanhd/registry-pattern/actions/workflows/ci.yml/badge.svg?branch=main&event=push&name=build&matrix.python-version=3.8)

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
pytest

# Run type checking
pyright

# Check code format
black --check .

# Lint code
flake8 .
```
