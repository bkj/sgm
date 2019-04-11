#!/usr/bin/env/python

from setuptools import setup

setup(
    name="sgm",
    author="...",
    classifiers=[],
    packages=['sgm'],
    install_requires=[
        'cython==0.29.3',
        'tqdm',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'git+https://github.com/gatagat/lap.git',
        'git+https://github.com/src-d/lapjv',
    ],
    version="1.0"
)