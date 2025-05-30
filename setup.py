import os, re
from setuptools import setup, find_packages

# Read the README file
with open("README.md") as f:
    registry_readme = f.read()

def read_file(filepath: str) -> str:
    """Read and return the content of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_dependencies() -> list:
    """Retrieve dependencies from the requirements file."""
    depfile = "requirements.txt"
    if os.path.exists(depfile):
        return [
            line.strip() for line in read_file(depfile).splitlines() 
            if line.strip() and not line.startswith("#")
        ]
    return []

def get_package_name() -> str:
    """Retrieve the package name from the project directory structure."""
    packages = find_packages()
    if packages:
        return packages[0]
    raise RuntimeError(
        "No package found. Ensure your project contains a valid Python package."
    )

def get_version() -> str:
    """Retrieve the package version from the version file."""
    package = get_package_name()
    versionfile = os.path.join(package, "_version.py")

    if os.path.exists(versionfile):
        verstrline = read_file(versionfile)
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string in '_version.py'.")

    raise FileNotFoundError("Version file '_version.py' not found.")

# Define optional dependencies for plugins
extras_require = {
    # Pydantic validation plugin
    "pydantic-plugin": [
        "pydantic>=2.8.0",
        "pydantic-core>=2.20.0",
    ],
    
    # Observability plugin for enhanced logging and monitoring
    "observability-plugin": [
        "structlog>=23.0.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "prometheus-client>=0.19.0",
    ],
    
    # Serialization plugin for JSON/YAML/pickle support
    "serialization-plugin": [
        "orjson>=3.9.0",  # Fast JSON serialization
        "pyyaml>=6.0.1",  # YAML support
        "msgpack>=1.0.7",  # MessagePack binary serialization
    ],
    
    # Remote registry plugin for distributed registries
    "remote-plugin": [
        "redis>=5.0.0",
        "httpx>=0.25.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
    ],
    
    # CLI plugin for command-line registry management
    "cli-plugin": [
        "click>=8.1.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
    ],
    
    # Caching plugin for advanced caching strategies
    "caching-plugin": [
        "cachetools>=5.3.0",
        "diskcache>=5.6.0",
    ],
    
    # Development dependencies
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
        "hypothesis>=6.88.0",  # Property-based testing
        "black>=23.0.0",
        "flake8>=6.0.0",
        "pyright>=1.1.0",
        "pre-commit>=3.4.0",
    ],
    
    # Testing dependencies
    "test": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
        "hypothesis>=6.88.0",
        "pytest-benchmark>=4.0.0",
        "pytest-mock>=3.11.0",
    ],
    
    # Performance testing
    "performance": [
        "pytest-benchmark>=4.0.0",
        "memory-profiler>=0.61.0",
        "psutil>=5.9.0",
    ],
}

# Add 'all' option to install all plugins
extras_require["all"] = [
    dep for deps in extras_require.values() 
    for dep in deps if isinstance(deps, list)
]

# Setup the package
if __name__ == '__main__':
    setup(
        name="registry",
        version=get_version(),
        description="A comprehensive registry pattern for Pydantic models and PyTorch modules with plugin support.",
        long_description=registry_readme,
        long_description_content_type="text/markdown",
        author="adnanharundogan",
        author_email="adnanharundogan@gmail.com",
        url="https://github.com/adnanhd/registry-pattern",
        license="MIT",
        packages=find_packages(),
        install_requires=get_dependencies(),
        extras_require=extras_require,
        python_requires=">=3.8",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Utilities",
        ],
        keywords="registry pattern validation pydantic pytorch",
        project_urls={
            "Bug Reports": "https://github.com/adnanhd/registry-pattern/issues",
            "Source": "https://github.com/adnanhd/registry-pattern",
            "Documentation": "https://registry-pattern.readthedocs.io/",
        },
        entry_points={
            "console_scripts": [
                "registry-cli=registry.plugins.cli:main [cli-plugin]",
            ],
        },
    )
