import os, re
from setuptools import setup, find_packages

# Read the README file
with open("README.md") as f:
    torchutils_readme = f.read()

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    torchutils_requirements = f.read().strip().split("\n")


def read_file(filepath: str) -> str:
    """Read and return the content of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_dependencies() -> list:
    """Retrieve dependencies from the requirements file."""
    depfile = "requirements.txt"
    if os.path.exists(depfile):
        return [
            line.strip() for line in read_file(depfile).splitlines() if line.strip()
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


# Setup the package
setup(
    name="registry",
    version=get_version(),
    description="A registry pattern for Pydantic models and PyTorch modules.",
    long_description=torchutils_readme,
    long_description_content_type="text/markdown",
    author="adnanharundogan",
    author_email="adnanharundogan@gmail.com",
    license="MIT",
    install_requires=torchutils_requirements,
    packages=["registry"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
