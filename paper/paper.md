---
title: 'dhdt: A photohypsometric Python library to estimate glacier elevation
        change via optical remote sensing imagery'
tags:
  - Python
  - glacier elevation change
  - optical remote sensing
  - geodetic imaging
  - feature tracking
authors:
  - name: Bas Altena
    orcid: 0000-0001-9025-0326
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Francesco Nattino
    orcid: 0000-0003-3286-0139
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
affiliations:
 - name: Institute for Marine and Atmospheric research (IMAU),
         Department of Physics, Faculty of Science, Utrecht University,
         Utrecht, the Netherlands
   index: 1
 - name: Space Accountants, Utrecht, the Netherlands
   index: 2
 - name: Netherlands eScience Center, Amsterdam, the Netherlands
   index: 3
date: 14 November 2022
bibliography: paper.bib

---

# Summary
Extracting historical and present day measurements of mountain glacier change is
challenging. Their locations are remote and their population is vast[^1].
Consequently, satellite Earth observation are an ideal means to collect
information about their geometry and surface characteristics. Such information
can be used for contemporary climate reconstruction and improve future
projection of fresh water availability or their contribution to sea-level.

The archive of satellite missions with geometric mapping capabilities is very
limited in space and time. While optical missions with a single telescope are
more abundant. Hence, methodologies that are able to get at least some
information out are worth exploring.

`dhdt` is Python library that has the functionality to automatically extract
glacier elevation change from large collections of satellite imagery. The
different functions are organised in modules, and each function is documented in
a [readthedocs](https://dhdt.readthedocs.io/).

[^1]: current estimates of glaciers on Earth come to almost 200'000

# Statement of need
Regional or global scale surface mass-balance modeling is mainly forced through
atmospheric circulation models [@lenaerts2019observing; @zekollari2022ice].
Where in-situ stake measurements are used for calibration, or validation. While
remote sensing data is able to provide additional observations of the
(sub)-surface.

Such remote sensing observations can be for longer time periods, where the
length changes from glacier outlines, can sometimes date back up a century or
more. While at decadal timescales the use of elevation models can help to
constrain a time-spans, as it delivers an estimate of volume changes (a.k.a.
geodetic mass balance) [@hugonnet2021accelerated]. While, velocity fields can
be used to infer current ice thickness [@millan2022ice]. At annual time scales,
the transient snowline can be linked to the surface mass balance
[@davaze2018monitoring; @barandun2021hot]. Enhancing the ability to glacier
specific mass-balance functions.

Extraction of elevation data from shadow cast has been around for some time in
the computer vision community. But recently this has gone into the wild,
exploiting real world data. Here we have dubbed this technique photohypsometry,
but other terms like shape-from-shadowing [@daum1998threed] and heliometric
stereo [@abrams2012heliometric] have also been used for this methodology.
This methodology has gotten some interest in the cryospheric research community
[@rada2022high; ], though up to now it has stayed at the level of a proof of concept.

Here, we present a Pyhton library that includes a complete and highly automated
processing pipeline. The procedures of such a photohypsometic pipeline are
illustrated in \autoref{fig:pipeline}.

![Schematic of the photohypsometric pipeline.\label{fig:pipeline}](fig/generalworkflow.pdf){ width=100% }


# Design and data sources
The data structure of the `dhdt` library follows a data processing pipeline,
where general functions call more detailed and refactured components. The
top level structure is as follows:

```
dhdt
├── input
├── preprocessing
├── processing
├── postprocessing
├── presentation
├── generic
├── auxilary
└── testing
```

When a specific function is based upon a given methodology, the literature that
is at its root is given in the doc-string. Hence the references in this work are
by far not comprehensive.


## [input](https://dhdt.readthedocs.io/en/latest/modules.html#input)

A suit of high resolution[^2] optical satellite systems are currently in space,
of which many have adopted an open access policy. Hence, a large selection of
these satellite data is supported by the `dhdt` library. Though an emphasis is
given to the Senintel-2 system ([``read_sentinel2``](https://dhdt.readthedocs.io/en/latest/modules.html#read-sentinel-2-meta-data)),
which is in operation since XXXX.

Though support is also given to the Landsat ([``read_landsat8``](https://dhdt.readthedocs.io/en/latest/modules.html#read-landsat8-meta-data)),
SPOT legacy () and ASTER ([``read_aster``](https://dhdt.readthedocs.io/en/latest/modules.html#read-terra-aster-data)),
as well as, commercial satellite systems ([``read_planetscope``](https://dhdt.readthedocs.io/en/latest/modules.html#read-planetscope-data),
[``read_rapideye``](https://dhdt.readthedocs.io/en/latest/modules.html#read-rapideye-data))
and demonstration missions like VENuS [``read_venus``](https://dhdt.readthedocs.io/en/latest/modules.html#read-venus-data)).

Typical functions include loading of the imagery, and associated meta-data.
These are functions about the instrument specifics or the acquisition situation.

```
input/
├── read_aster.py
├── read_hyperion.py
├── read_landsat.py
├── read_planetscope.py
├── read_rapideye.py
├── read_sentinel2.py
├── read_spot4.py
├── read_spot5.py
└── read_venus.py
```

[^2]: here high resolution denotes imagery with a pixel spacing in the order
      of 5 to 30 meters

## [preprocessing](https://dhdt.readthedocs.io/en/latest/modules.html#preprocessing)

Prior to the coupling of imagery and the extraction of elevation change data,
the satellite data needs to be made ready. Typically, shadows are not the main
interest for such imagery, hence the functions in this sub-directory are
taylored towards enhancing the shadow imagery and getting the proper geometric
information needed. The structure of the folder is as follows,

```
preprocessing/
├── acquisition_geometry.py
├── aquatic_transforms.py
├── atmospheric_geometry.py
├── color_transforms.py
├── handler_multispec.py
├── image_transforms.py
├── shadow_filters.py
├── shadow_geometry.py
├── shadow_matting.py
├── shadow_transforms.py
└── snow_transforms.py
```

Most optical remote sensing instruments record multiple spectral intervals at
(nearly-)simultaneously. Such spectral information can de used to separate out
the surface reflectivity (albedo) from the illumination component. This
illumination component is a combination of shading and shadowing. A multitude of
methods have been developed [@tsai2006comparative],[@tsai2006comparative],
[@tsai2006comparative] and these and more are implemented in
[``shadow_transforms``](https://dhdt.readthedocs.io/en/latest/modules.html#shadow-transforms).

An even richer collection can be implemented when analysis in the temporal
domain are included. Though such methods enforce a stable acquisition setup,
which is not always guaranteed for spaceborne acquisitions. However, apart from
the spectral methods mentioned above, the spatial neighbourhood can also be used
to enhance the performance of the shadow localisation. Such methods are included
within [``shadow_filters``](https://dhdt.readthedocs.io/en/latest/modules.html#shadow-filters)
Where local or zonal neighbourhood filters can reduce speckle noise in the
classification, see also an example in \autoref{fig:red-shadow}.

![Example of the different preprocessing functions.\label{fig:red-shadow}](fig/red-shadow.pdf){ width=100% }

Apart from the signal separation to enhance shadows, a suit of functions is
present in `dhdt` to deal with parsing of geometric information. The metadata of
the instrument needs to be combined with terrain properties. Hence in [``acquisition_geometry``](https://dhdt.readthedocs.io/en/latest/modules.html#acquisition-geometry)
coupling of the acquisition geometry of the satellite scenes are coupled to
different terrain angles.

![Acquisition geometry with an emphasis on refraction.\label{fig:red-shadow}](fig/spacecamera_hypsometry.pdf){ width=30% }

The atmosphere is also an element within the photohypsometric chain to take into
account, since the sun traces are bended through the atmosphere. This refraction
is typically not present in the meta data of the satellite, and only the
geometric line of sight is given, see also \autoref{fig:red-shadow}. However,
correct sun angles are essential, especially for scenes with low sun angles
(such situations occur especially often at high latitudes, where many glaciers
are situated). Therfore, ``atmospheric_geometry`` has functions to correct for
such angles given specific wavelengths and atmospheric compositions.

## [processing](https://dhdt.readthedocs.io/en/latest/modules.html#processing)

An important building block of `dhdt` is image correspondence, this is an
ill-posed problem. Hence, a suit of methodologies are implemented in this
library.

Image correspondence

Generic .

```
processing/
├── coregistration.py
├── coupling_tools.py
├── geometric_correction_measures.py
├── geometric_image_describtion.py
├── geometric_precision_describtion.py
├── gis_tools.py
├── matching_tools.py
├── matching_tools_binairy_boundaries.py
├── matching_tools_differential.py
├── matching_tools_frequency_affine.py
├── matching_tools_frequency_correlators.py
├── matching_tools_frequency_differential.py
├── matching_tools_frequency_filters.py
├── matching_tools_frequency_metrics.py
├── matching_tools_frequency_spectra.py
├── matching_tools_frequency_subpixel.py
├── matching_tools_geometric_temporal.py
├── matching_tools_harmonic_functions.py
├── matching_tools_organization.py
├── matching_tools_spatial_correlators.py
├── matching_tools_spatial_metrics.py
├── matching_tools_spatial_subpixel.py
├── network_tools.py
├── photoclinometric_functions.py
└── photohypsometric_image_refinement.py
```

In the `dhdt` library the different approaches to implement image
correspondence [^3] is subdivided into three domains. The first major
subdivision, are implementation based on Fourier based methods. Here, imagery
are transformed into the frequency domain, which is computational efficient.
The second subdivision are methods formulated in the spatial domain, which are
mostly based upon convolution. The last subdivision is also formulated in the
spatial domain, but is based upon differential methods. These optical flow
methods work best, when displacements are within sub-pixel level.

[^3]: also known as, pattern matching, feature tracking, image velocimetry

![Illustration of finding image correspondence via the frequency domain.\label{fig:freq-match}](fig/frequency_matching.pdf){ width=100% }

The general procedure of frequency based image correspondence estimation is
illustrated in \autoref{fig:freq-match}. After the image templates are
transformed to the frequency domain, the phase imagery can be adjusted via
different filters ([``matching_tools_frequency_filters``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_frequency_filters)).
Similarly, the calculation of the cross-power spectrum can be done via different
means ([``matching_tools_frequency_correlators``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_frequency_correlators)).
The resulting phase plane can be used to estimate the subpixel displacement
([``matching_tools_frequency_subpixel``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_frequency_subpixel)).
While the quality of match can be deduced from the cross-power spectrum
([``matching_tools_frequency_metrics``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_frequency_metrics)).

![Illustration of finding image correspondence via the spatial domain.\label{fig:spat-match}](fig/spatial_matching.pdf){ width=100% }

Image correspondence via the spatial domain is very similar, see also
\autoref{fig:spat-match}. Though here a smaller image subset is used and is
slided over the other image subset. Again specific image operators can be used
to put emphasis on certain image structures, but these functions are already
present in the [preprocessing](#preprocessing) folder. While
different correspondence metrics can be found in [``matching_tools_spatial_correlators``](https://dhdt.readthedocs.io/en/latest/modules.html#spatial-correlators).
This results in a two dimensional correlation function, where its peak can be
localised by different methods ([``matching_tools_spatial_subpixel``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_spatial_subpixel)),
as well as its metrics ([``matching_tools_spatial_metrics``](https://dhdt.readthedocs.io/en/latest/modules.html#module-dhdt.processing.matching_tools_correlation_metrics)). Other descriptions of the
matching quality are implemented in ``geometric_precision_describtion``,
where functions are situated that give indicators for the precision of an
individual match.

![Schematic of the processing steps within a geodetic imaging pipeline, using a frequency based matching approach.\label{fig:freq-scheme}](fig/matching_schematic_fourier.pdf){ width=100% }

While image correspondence is an essential component of a typical geodetic
imaging pipeline, many functions are needed to support this building block. See
for a schematic of such a pipeline also \autoref{fig:freq-scheme}.
For example, functions in ``matching_tools`` and ``coupling_tools`` are generic
functions to create for example a sampling setup. While ``netwerk_tools`` deals
with large organization of imagery over time, similarly multi-temporal functions
can be found in ``matching_tools_geometric_temporal``. Lastly, most of the
matching methods implemented have been extended to also cope with multi-spectral
data such that stacking of cross-power correlation spectra [@altena2022improved]
is possible.

## [postprocessing](https://dhdt.readthedocs.io/en/latest/modules.html#postprocessing)

The functions within this folder are mostly concerned with product creation for
specific application domains. It is composed of the following file structure:

```
postprocessing
├── adjustment_geometric_temporal.py
├── atmospheric_tools.py
├── displacement_filters.py
├── displacement_tools.py
├── glacier_tools.py
├── group_statistics.py
├── mapping_io.py
├── photohypsometric_tools.py
├── snow_tools.py
├── solar_surface.py
├── solar_tools.py
└── terrain_tools.py
```

The products generated from image matching are typically noisy. The error
distribution has elements of normally distributed noise, but a large part of the
sample can also have outliers. Typically sampling of the neighbourhood is used
to clean such data, and these function can be found in [``displacement_filters``](https://dhdt.readthedocs.io/en/latest/modules.html#displacement-filters)
and more generic in ``group\_statistics.py``.
Multiple displacement and velocity products over time create redundancy and make
it possible to apply inversion. This results in harmonised and evenly sampled
data. Such functions can be found in ``adjustment_geometric_temporal``.

The spatialtemporal coupling of can be found in ``photohypsometric\_tools``

Further specific application domain functions, that relate to glaciology,
hydrology, meteorology can be found in the other packages.

## [presentation](https://dhdt.readthedocs.io/en/latest/modules.html#presentation)

Functions within the presentation directory of the library are associated to
specific data representation, typically with a spatial component. The file
structure of this directory is as follows:

```
presentation
├── displacement_tools.py
├── image_io.py
├── terrain_tools.py
└── velocity_tools.py
```

Generic functions are present in [``image_io``](https://dhdt.readthedocs.io/en/latest/modules.html).
to create (geo-referenced) imagery of the (intermediate) results presented in
the former directories. Since elevation data is needed for the photohypsometric
pipeline, a suit of functions is present in the [``terrain_tools``](https://dhdt.readthedocs.io/en/latest/modules.html).
Where mountain specific shading functions are incorporated.
Other map making and data visualization functions specificly for surface
kinematics can be found in [``displacement_tools``](https://dhdt.readthedocs.io/en/latest/modules.html)
and [``glacier_tools``](https://dhdt.readthedocs.io/en/latest/modules.html).
Think of displacement vectors and their associated precision or strain rate maps.
While fucntions within ['velocity_tools'](https://dhdt.readthedocs.io/en/latest/modules.html))
are focussed on animations, of moving particles and flow paths.

## [generic](https://dhdt.readthedocs.io/en/latest/modules.html#generic)

Many spatial functions are used in several .
Especially terrain tools based on the elevation model used.

![Example of shadow refinement (red) starting from an initial shadowing (purple) that is based on an elevation model.](fig/red-snake.pdf){ width=100% }


## [auxilary](https://dhdt.readthedocs.io/en/latest/modules.html#auxilary)

CopernicusDEM, Randolph Glacier Inventory, ERA-5

Localization of the glaciers is needed.

The XXX

For the estimation of the refraction, the state of the atmosphere needs to be
known. Hence, atmospheric variables such as temperature and humidity are XX

```
auxilary/
├── handler_coastal.py
├── handler_copernicusdem.py
├── handler_era5.py
├── handler_mgrs.py
└── handler_randolph.py
```

# Functionality

##### Surface displacement estimation
nice glacier velocity plot...?

##### Spatial temporal elevation change
geodetic mass balance, maybe get some CryoSAT data?

# Other software
Dissemination of satellite imagery is typically provided at different processing
levels. The nomenclature is not standardised, but mostly level-0 data is raw
sensor readings. Level-1 is georeferenced, while Level-2 is transformed to a
physical measure. Lastly, Level-3 data are spatiotemporal consistent datasets.
Since `dhdt` is mostly interested in the geometrical aspect, the lowest level is
of most importance. Hence, for Sentinel-2 Level-1T is used. This is not the most
common level, and repositories such as [SentinelSat](https://github.com/sentinelsat/sentinelsat)
have routines for downloading Sentinel-2 Level-2 imagery.

A generic remote sensing library is [Orfeo Toolbox](https://www.orfeo-toolbox.org)
[@grizonnet2017orfeo] this toolbox has a rich code base which has its emphasis
on typical remote sensing methods, such as classification.

[MicMac](https://www.micmac.ign.fr)[@rupnik2017micmac] is another open-source
software library, which stems from photogrammetry, where displacements are
directly related to disparity, resulting in a topographic reconstruction. Though
with some adjustments of the pipeline, it is also possible to extract surface
displacements.

More glacier specific software can be found in Open Global Glacier Model
([OGGM](https://www.oggm.org)) [@maussion2019open]. While this initiative
started out as a modular glacier flowline model, it now entails a large
ecosystem with highly automated access to atmospheric and geospatial datasets.
This modelling infrastructure has an approach based upon individual glaciers,
which is also adopted in `dhdt`. Hence integration with this framework is
foreseen in the future. Lastly, Quantum Geographic Information System
([QGIS](https://www.qgis.org/en/site/)) is a general system to combine and
catalogue geographical datasets of all sorts.

# Other datasets and repositories
Photogrammetry[^4] is another technique that is able to get elevation products over
time. However, an additional off-nadir telescope is needed to be able to create
a systematic mapping configuration. It is also possible to stare towards a
target region, but such systems are typically commercial and have limited
coverage. Nonetheless, photogrammetry is able to create a large spatial coverage
of a region. Similar to `dhdt`, the postprocessing procedures for glacier
specific workflows are also captured by a repository such as [xDEM](https://github.com/GlacioHack/xdem).
Since elevation models have specific systematic errors within [@hugonnet2022uncertainty].

[^4]: also known as, structure from motion, shape from motion

Apart from elevation products from optical remote sensing instruments, is it
possible to estimate the geodetic mass balance from scattered laser altimetry
data [@kaab2008glacier]. Such systems are relatively limited in space, but their
detailed footprint in the order of tens of meters and their repeated
overflights make the generated products spatially sparse but consistent.
Currently, ICESAT-2 is operational and tools for data processing can be found in
the [icepyx](https://github.com/icesat2py/icepyx) repository.

In the same realm one can use of microwave altimetry can be used for geodetic
mass balance. Such altimeters have a footprint in the order of kilometers, thus
are less useful for rough mountain terrain. Though interferometric altimeters,
such as CryoSAT, are able to cope with complex mountain terrain [@gourmelen2018cryosat].
Products from such satellites are ideal for flatter upper regions of glaciers
and ice caps. Data from such satellites are currently available via the
[CryoTEMPO-EOLIS](https://cryotempo-eolis.org/point-product/) project.


# Description of software

`dhdt` has other important dependencies, namely

# Acknowledgements

We acknowledge contributions from Ou Ku, and Meiert Grootes, and support from
Bert Wouters, Yifat Dzigan and Michiel van den Broeke during the genesis of this
project. This project received funding from the Dutch research council (NWO)
and the Netherlands Space Office (NSO) under their joint research progamme GO
(grant agreement No. ALWGO.2018.044 - Eratosthenes).

# References
