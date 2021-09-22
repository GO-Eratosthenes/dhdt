import glob
import os

import numpy as np

from osgeo import gdal, osr

from xml.etree import ElementTree

def read_band_re(band, path):  # pre-processing
    """
    This function takes as input the RapidEye band number and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   band           string            RapidEye band name
             path           string            path of the folder
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
    """
    fname = os.path.join(path, '*_Analytic.tif')
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(band).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_sun_angles_re(path):  # pre-processing
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    input:   path           string            path where xml-file of
                                              Sentinel-2 is situated
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimtuh angles

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