|GitHub Badge| |License Badge| |Python Build| |Documentation Status| |OpenSSF Best Practices|

.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/GO-Eratosthenes/eratosthenes
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/GO-Eratosthenes/start-code
   :target: https://github.com/GO-Eratosthenes/start-code
   :alt: License Badge

.. |Python Build| image:: https://github.com/GO-Eratosthenes/start-code/workflows/Build/badge.svg
   :target: https://github.com/GO-Eratosthenes/start-code/actions?query=workflow%3A%22build.yml%22
   :alt: Python Build
   
.. |Documentation Status| image:: https://readthedocs.org/projects/dhdt/badge/?version=latest
   :target: https://dhdt.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |OpenSSF Best Practices| image:: https://bestpractices.coreinfrastructure.org/projects/7641/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/7641
   :alt: OpenSSF Best Practices

############
dhdt
############

Extracting topography from mountain glaciers, through the use of shadow casted by surrounding mountains. Imagery from optical satellite systems are used, over all mountain ranges on Earth.


Installation
************

Download and access the package folder using `git`:

.. code-block:: console

  git clone https://github.com/GO-Eratosthenes/dhdt.git
  cd dhdt


The dependencies are most easily installed with `conda` from the `conda-forge` channel (see `Miniforge installers`_ for a minimal Conda installation).
Create and activate a virtual environment with all the required dependencies:

.. code-block:: console

  conda env create -n dhdt -f environment.yml
  conda activate dhdt


Install `dhdt` using `pip` (add the `-e` option to install in development mode):

.. code-block:: console

  pip install .

.. _Miniforge installers : https://github.com/conda-forge/miniforge/releases

Contributing
************

If you want to contribute to the development of this package,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2023, Netherlands eScience Center, Utrecht University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Credits
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
