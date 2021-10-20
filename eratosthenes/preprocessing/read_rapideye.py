import glob
import os

import numpy as np

from osgeo import gdal, osr

from xml.etree import ElementTree

def read_band_re(band, path):
    """ read specific band of RapidEye image
    
    This function takes as input the RapidEye band number and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    band : string
        RapidEye band number.
    path : string
        path to folder with imagery.

    Returns
    -------
    data : np.array, size=(m,n), dtype=integer
        data of one RadidEye band.
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)    
    """
    fname = os.path.join(path, '*_Analytic.tif')
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(band).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_sun_angles_re(path):
    """ read the sun angles, corresponding to the RapidEye acquisition

    Parameters
    ----------
    path : string
        path where xml-file of RapidEye is situated.

    Returns
    -------
    Zn : np.array, size=(m,n), dtype=float
        array of the Zenith angles
    Az : np.array, size=(m,n), dtype=float
        array of the Azimtuh angles

    Notes
    ----- 
    The azimuth angle declared in the following coordinate frame: 
        
        .. code-block:: text
        
                 ^ North & y
                 |  
            - <--|--> +
                 |
                 +----> East & x

    The angles related to the sun are as follows:
        
        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          |-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          +---- surface         +------


    """
    fname = os.path.join(path, '_metadata.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    mI = int(root[4][0][0][0][5].text)
    nI = int(root[4][0][0][0][6].text)
    
    Azimuth = float(root[2][0][3][0][2].text)
    Zenith = 90-float(root[2][0][3][0][3].text)
    
    Zn = Zenith*np.ones((mI,nI))
    Az = Azimuth*np.ones((mI,nI))
    
    return Zn, Az