# pylint: disable=missing-module-docstring
from setuptools import find_packages
from setuptools import setup

setup(
    name='decision_rules',
    description=(
        "Package implementing decision rules. Includes tools for calculations of various measures "
        "and indicators, as well as algorithms for filtering rulesets."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version='1.1.0',
    author='Cezary Maszczyk, Dawid Macha, Adam Grzelak',
    author_email='cezary.maszczyk@emag.lukasiewicz.gov.pl',
    readme='README.md',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.0.3',
        'pydantic==2.9.2',
        'scipy==1.13.1',
        'scikit-learn==1.5.2',
        'imbalanced-learn==0.12.3',
        'typeguard==4.3.0',
    ],
    extras_require={
        "ruleset_factories": [
            "rulekit>=2.1.18.0"
        ]
    },
    python_requires='>=3.9',
)
