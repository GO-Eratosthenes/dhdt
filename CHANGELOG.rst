###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

In order to release a new package version:

* Make sure the current file (`<CHANGELOG.rst>`_) is updated.
* Set the new version number in `<dhdt/__version__.py>`_ .
* Update the `<CITATION.cff>`_ file (at lease the ``version`` and ``date-released`` fields).
* Update the `<.zenodo.json>`_ file from  `<CITATION.cff>`_ (using ``cffconvert``):

.. code-block::
    cffconvert --validate
    cffconvert -f zenodo -o .zenodo.json

* Create Github release.
* Verify that the code is released on Zenodo and PyPI.

[Unreleased]
************

Added
-----

* Empty Python project directory structure
