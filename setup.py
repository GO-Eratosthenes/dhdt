#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit dhdt/__version__.py
version = {}
with open(os.path.join(here, 'dhdt', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = [line.strip('\n') for line in requirements_file.readlines()]

setup(
    name='dhdt',
    version=version['__version__'],
    description='Extracting topography from mountain glaciers, through the '
                'use of shadow casted by surrounding mountains. imagery from '
                'optical satellite systems are used, over all mountain ranges '
                'on Earth.',
    long_description=readme + '\n\n',
    author='Bas Altena',
    author_email='b.altena@uu.nl',
    url='https://github.com/GO-Eratosthenes/eratosthenes',
    packages=find_packages(exclude=['*tests*']),
    include_package_data=True,
    license='Apache Software License 2.0',
    zip_safe=False,
    keywords='dhdt',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests',
    install_requires=requirements,
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pycodestyle'],
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
