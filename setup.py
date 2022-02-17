#!/usr/bin/env python3

from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup

setup(name='pyLODE',
      version='0.0.0',
      description='Python Implementation of LODE',
      packages=find_packages(),
      package_dir={'': '.'},
      py_modules=[splitext(basename(i))[0] for i in glob("*.py")],
      )
