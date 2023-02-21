input
-----

read Sentinel-2 (meta)data
~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentinel-2 is a high-resolution (10-60 m) multi-spectral satellite constellation
from European Space Agency and the European Union. The following functions make
is easier to read such data.

.. automodule:: dhdt.input.read_sentinel2
    :members:

read Landsat (meta)data
~~~~~~~~~~~~~~~~~~~~~~~~
Landsat 8 & 9 are the latest fleet of satellites that are part of a legacy. Both
satellites are build by NASA and data provision is done through USGS. The
following function makes reading such data easier.

.. automodule:: dhdt.input.read_landsat
    :members:

read Terra ASTER data
~~~~~~~~~~~~~~~~~~~~~
The ASTER instrument onboard of the Terra satellite has been collecting data
since 1999, but will be decommisioned soon. Nonetheless, its archive is very
valuable and the following functions make reading such data easier.

.. automodule:: dhdt.input.read_aster
    :members:

read RapidEye data
~~~~~~~~~~~~~~~~~~~~~
The RapidEye constellation composed of five commercial mini-satellites,
image collection started in 2008 and halted in 2020. The following functions
make reading such data easier.

.. automodule:: dhdt.input.read_rapideye
    :members:

read PlanetScope data
~~~~~~~~~~~~~~~~~~~~~
The PlanetScope constellation is a commercial fleet of micro-satellites. The
following function makes reading such data easier.

.. automodule:: dhdt.input.read_planetscope
    :members:

read VENµS data
~~~~~~~~~~~~~~~~~~~~~
The VENµS satellite is a demonstration satellite, that acquires over specific
pre-defined regions. It has a high repeat rate in the order of one or two days,
if cloud cover permits. The following functions make reading of such data
easier.

.. automodule:: dhdt.input.read_venus
    :members:
