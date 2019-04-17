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
        'lap @ git+https://github.com/nowfred/lap.git@v0.5.0#egg=lap',
        'lapjv @ git+https://github.com/nowfred/lapjv.git@v1.4.1#egg=lapjv'
    ],
    dependency_links=[

    ],
    version="1.0"
)