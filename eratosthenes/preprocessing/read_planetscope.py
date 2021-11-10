# generic libraries
import os
import glob

import numpy as np

from ..generic.mapping_io import read_geo_image

def read_band_ps(dir_path, fname):
    """
    This function takes as input the PlanetScope file name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    dir_path : string
        path of the folder
    fname : string
        PlanetScope image name

    Returns
    -------
    data : np.array, size=(_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    See Also
    --------
    list_central_wavelength_ps

    """
    fname = os.path.join(dir_path, fname)
    assert len(glob.glob(fname))!=0, ('file does not seem to be present')

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])
    return data, spatialRef, geoTransform, targetprj