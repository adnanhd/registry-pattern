from __future__ import annotations

import os
import re
import subprocess
import platform
import sys
from pathlib import Path
if sys.version_info >= (3, 9):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

VERSION = '0.0.1'


def _safe_regex_search(pattern, text, index=1):
    result = re.search(pattern, text)
    if isinstance(result, re.Match):
        return result.group(index)
    else:
        return 'not-found'


def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.exists(os.path.join(dir, '.git'))


def have_git() -> bool:
    """Can we run the git executable?"""
    try:
        subprocess.check_output(['git', '--help'])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def git_most_recent_commit(dir: str, short: bool = False) -> str:
    """Get the SHA-1 of the HEAD of a git repository."""
    if short:
        command = ['git', 'rev-parse', '--short', 'HEAD']
    else:
        command = ['git', 'rev-parse', 'HEAD']

    if is_git_repo(dir) and have_git():
        return subprocess.check_output(command, cwd=dir).decode('utf-8').strip()
    else:
        return 'unknown'


def get_torch_version() -> str:
    try:
        import torch
    except ImportError:
        return 'torch not found'

    TORCH_CONFIG = torch.__config__.show()

    def search_on_torch_config(pattern):
        return _safe_regex_search(pattern, TORCH_CONFIG, 1)

    TORCH_VERSION = torch.__version__
    CUDA_VERSION = search_on_torch_config(r'CUDA Runtime (\d+\.\d+)')
    CUDNN_VERSION = search_on_torch_config(r'CuDNN (\d+\.\d+)')
    GCC_VERSION = search_on_torch_config(r'GCC (\d+\.\d+)')
    MAGMA_VERSION = search_on_torch_config(r'Magma (\d+\.\d+)')

    return f'{TORCH_VERSION} (CUDA {CUDA_VERSION}, CuDNN {CUDNN_VERSION}, GCC {GCC_VERSION}, Magma {MAGMA_VERSION})'


def list_gpu_devices() -> str:
    from collections import Counter

    command = 'nvidia-smi --query-gpu=name --format=csv,noheader'
    try:
        query: bytes = subprocess.check_output(command.split(' '))
    except subprocess.CalledProcessError:
        return 'GPU not found'
    else:
        devices: list[str] = query.decode('utf-8').strip().split('\n')
        return " + ".join(
            f'{device_count} x {gpu_device}'
            for gpu_device, device_count in Counter(devices).items()
        )


def get_cuda_runtime_version() -> str:
    try:
        query = subprocess.check_output(['nvcc', '--version'])
    except subprocess.CalledProcessError:
        return 'CUDA-Runtime not found'
    else:
        nvcc_output = query.decode('utf-8').strip()
        BUILD_INFO = _safe_regex_search(r'Build\s+(.+)', nvcc_output)
        VERSION = _safe_regex_search(r'release (\d+\.\d+)', nvcc_output)
        return f'{VERSION} ({BUILD_INFO})'


def get_cuda_driver_version() -> str:
    try:
        query = subprocess.check_output(['nvidia-smi'])
    except subprocess.CalledProcessError:
        return 'CUDA-Driver not found'
    else:
        output = query.decode('utf-8').strip()
        CUDA_RT = _safe_regex_search(r'CUDA Version: (\d+\.\d+)', output)
        DRIVER = _safe_regex_search(r'Driver Version: (\d+\.\d+)', output)
        return f'{CUDA_RT} (NVIDIA {DRIVER})'


def get_related_packages(package_names) -> list[str]:
    return [
        f'{dist.metadata["Name"]}-{dist.version}'
        for dist in importlib_metadata.distributions()
        if dist.metadata['Name'] in package_names
    ]


def version_info() -> str:
    """Return complete version information for Pydantic and its dependencies."""

    pydantic_pytorch_dir = os.path.abspath(
        os.path.dirname(os.path.dirname(__file__)))

    # get data about packages that are closely related to pytorch,
    # use pytorch or often conflict with pydantic_pytorch
    pytorch_related_packages = get_related_packages({
        'torch',
        'torchvision',
        'numpy',
        'torcheval',
        'torcfunc',
        'torchinfo',
        'torchtext',
    })

    # get data about packages that are closely related to pydantic,
    # use pydantic or often conflict with pydantic_pytorch
    pydantic_related_packages = get_related_packages({
        'pydantic',
        'pydantic_core',
        'email-validator',
        'fastapi',
        'mypy',
        'pydantic-extra-types',
        'pydantic-settings',
        'pyright',
        'typing_extensions',
    })

    info = {
        f'pydantic-pytorch version': VERSION,
        'install path': Path(__file__).resolve().parent,
        'Git Hash': git_most_recent_commit(pydantic_pytorch_dir, short=False),
        'python version': sys.version,
        'NVIDIA GPU devices': list_gpu_devices(),
        'NVIDIA CUDA Runtime': get_cuda_runtime_version(),
        'NVIDIA CUDA Driver': get_cuda_driver_version(),
        'platform': platform.platform(),
        'pytorch version': get_torch_version(),
        'pytorch related packages': ' '.join(pytorch_related_packages),
        'pydantic related packages': ' '.join(pydantic_related_packages),
    }
    # + '\n' + torch.__config__.show()
    return '\n'.join('{:>30} {}'.format(k + ':', str(v).replace('\n', ' ')) for k, v in info.items())
