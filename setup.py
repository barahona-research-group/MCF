"""Setup."""

from setuptools import find_packages
from setuptools import setup

__version__ = "0.0.2"

setup(
    name="MCF",
    version=__version__,
    author="Dominik Schindler",
    install_requires=["numpy>=1.18.1", "gudhi>=3.7.1"],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
)
