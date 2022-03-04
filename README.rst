|GitHub Badge| |License Badge| |Python Build| 

.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/GO-Eratosthenes/eratosthenes
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/GO-Eratosthenes/start-code
   :target: https://github.com/GO-Eratosthenes/start-code
   :alt: License Badge

.. |Python Build| image:: https://github.com/GO-Eratosthenes/start-code/workflows/Build/badge.svg
   :target: https://github.com/GO-Eratosthenes/start-code/actions?query=workflow%3A%22build.yml%22
   :alt: Python Build

############
Eratosthenes
############

Extracting topography from mountain glaciers, through the use of shadow casted by surrounding mountains. Imagery from optical satellite systems are used, over all mountain ranges on Earth.


Installation
************

This package requires the [GDAL](https://gdal.org) library, which is most 
easily installed through `conda` from the `conda-forge` channel:

.. code-block:: console

   conda install gdal -c conda-forge

This package can then be downloaded and installed using `git` and `pip`:

.. code-block:: console

  git clone https://github.com/GO-Eratosthenes/start-code.git
  cd start-code
  pip install .


Contributing
************

If you want to contribute to the development of this package,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2020, Netherlands eScience Center, Utrecht University

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
