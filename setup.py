# pylint: disable=missing-module-docstring
from setuptools import find_packages, setup

setup(
    name="decision_rules",
    description=(
        "Package implementing decision rules. Includes tools for calculations of various measures "
        "and indicators, as well as algorithms for filtering rulesets."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version="1.9.0",
    author="Cezary Maszczyk, Dawid Macha, Adam Grzelak, Bartosz Piguła",
    author_email="cezary.maszczyk@emag.lukasiewicz.gov.pl",
    readme="README.md",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=1.5",
        "pydantic>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.1",
        "imbalanced-learn>=0.10",
        "typeguard>=4.3",
        "packaging>=14.1",
    ],
    extras_require={
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.13.2",
            "pyvis>=0.2.2",
            "plotly>=5.0.0",
        ],
    },
    python_requires=">=3.9",
)
