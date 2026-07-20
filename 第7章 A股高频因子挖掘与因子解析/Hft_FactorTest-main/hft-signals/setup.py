#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

required = [
        'numba',
        'numpy',
        'pandas',
        'loguru',
        'matplotlib',
        ]

setup(
    name='hft_signals',
    version='0.0.2',
    packages=['hft', 
              'hft.signal',
              'hft.fast_signal',
              'hft.utils',
              'hft.utils.combine',
              'hft.utils.format',
              'hft.utils.target',
              'hft.utils.validate',
              'hft.utils.wrapper',],
    install_requires=required,
)