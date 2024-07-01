from setuptools import setup

# Read the README file
with open('README.md') as f:
    torchutils_readme = f.read()

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    torchutils_requirements = f.read().strip().split('\n')


# Setup the package
setup(
    name='pydantic_pytorch',
    version='0.0.1',
    description='A collection of utilities for PyTorch',
    long_description=torchutils_readme,
    long_description_content_type='text/markdown',
    author='adnanharundogan',
    author_email='adnanharundogan@gmail.com',
    license='MIT',
    install_requires=torchutils_requirements,
    packages=['pydantic_pytorch'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
