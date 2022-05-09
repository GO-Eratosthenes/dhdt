Minimal documentation for dhdt, a photohypsometric Python library
=================================================================

**dhdt** is a modular geodetic imaging framework written in Python.
It uses data from imaging satellites to extract (above) surface kinematics and
elevation (change). This is done via image matching techniques and geometric
principles.

This library is generic, but has a preference towards open satellite data, as
in recent years many space agencies have adopted an open data policy.
Consequently, functions for the Sentinel-2 satellites of the Copernicus system
are closest to completion.

Methodology
-----------
This library has two aspects

* photohypsometry : extracting elevation change from changing shadow cast
* photogrammetry : extracting displacement or disparity through changes in observation time and of angle

Application domains
-------------------
This library can be used for a multitude of purposes, though the main focused
lies towards the following products:

* glacier elevation change
* glacier velocity
* ocean circulation
* natural mass movements
* earthquake displacements


Guide
^^^^^

.. toctree::
   :maxdepth: 2

   getting_started
   tutorials
   modules
   theory
   help
   contribute

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
