"""Installation script for the 'cyberdog2_rl_lab' python package."""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "psutil",
    "pyyaml",
]

setup(
    name="cyberdog2_rl_lab",
    packages=find_packages(),
    package_data={
        "cyberdog2_rl_lab": [
            "assets/data/cyberdog2/meshes/*.dae",
            "assets/data/cyberdog2/urdf/*.urdf",
        ]
    },
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
    entry_points={
        "isaaclab_tasks": [
            "cyberdog2 = cyberdog2_rl_lab.tasks",
        ],
    },
)
