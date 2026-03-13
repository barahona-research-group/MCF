"""Setup."""

from setuptools import find_namespace_packages
from setuptools import setup

__version__ = "0.0.9"


setup(
    name="MCF",
    version=__version__,
    author="Juni Schindler",
    install_requires=[
        "matplotlib",
        "numpy",
        "gudhi",    
        "tqdm",
        "pandas",
    ],
    zip_safe=False,
    packages=find_namespace_packages("src"),
    include_package_data=True,
    package_dir={"": "src"},
)
