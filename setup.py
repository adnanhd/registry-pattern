#!/usr/bin/env python
"""Backwards-compatible setup.py for editable installs.

The main configuration is in pyproject.toml. This file exists for:
- Editable installs: pip install -e .
- Legacy tools that don't support PEP 517/518
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
