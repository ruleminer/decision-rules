# pylint: disable=missing-module-docstring
from setuptools import find_packages
from setuptools import setup

setup(
    name='decision_rules',
    description='''
    Package implementing decision rules. Includes tools for calculations of various measures and indicators,
    as well as algorithms for filtering rulesets.
    ''',
    version='1.0.0',
    author='Cezary Maszczyk, Dawid Macha, Adam Grzelak',
    author_email='cezary.maszczyk@emag.lukasiewicz.gov.pl',
    packages=find_packages(),
    install_requires=[
        'numpy~=1.24.2',
        'pandas~=1.5.3',
        'pydantic >= 2.0.0, < 2.6.0',
        'scipy==1.11.1',
        'scikit-learn~=1.1.3',
        'imbalanced-learn~=0.10.1',
        'typeguard~=4.3.0',
    ],
)
