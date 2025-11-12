"""Setup."""

from setuptools import find_packages
from setuptools import setup

__version__ = "0.0.8"


setup(
    name="MCF",
    version=__version__,
    author="Juni Schindler",
    install_requires=[
        "matplotlib",
        "numpy",
        "gudhi",    
        "tqdm"
    ],
    zip_safe=False,
    extras_require={
        "rivet": ['pyrivet @ git+https://github.com/juni-schindler/rivet-python.git'],
        "sankey": ['omics_sankey @ git+https://github.com/juni-schindler/OmicsSankey.git']
    },
    packages=find_packages(),
    include_package_data=True,
)
