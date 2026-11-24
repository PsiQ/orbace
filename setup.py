# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
from setuptools import setup, find_packages

setup(
    name="orbace",
    version="0.3.0",
    description="OrbACE: Equivariant Message-Passing Potentials with Quantum-Chemistry Features",
    author="PsiQuantum Corp.",
    url="https://github.com/PsiQ/orbace",
    license="ASL-1.0 (Academic Software License; non-commercial)",
    packages=find_packages(include=["orbace", "orbace.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12",
        "e3nn==0.4.4",
        "mace-torch==0.3.15",
        "numpy",
        "scipy",
        "ase",
        "pyscf",
        "pyyaml",
    ],
    extras_require={
        "xtb": ["xtb"],
        "wandb": ["wandb"],
        "dev": ["pytest", "wandb"],
    },
)
