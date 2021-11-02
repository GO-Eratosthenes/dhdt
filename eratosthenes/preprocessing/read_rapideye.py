import glob
import os

import numpy as np
import pandas as pd

from osgeo import gdal, osr

from xml.etree import ElementTree

def list_central_wavelength_re():
    """ create dataframe with metadata about RapidEye

    Returns
    -------
    df : datafram
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2, having the following collumns:

            * wavelength : central wavelength of the band
            * bandwidth : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * name : general name of the band, if applicable
            * irradiance : exo-atmospheric radiance
                unit=W/m**2 μm
            * detectortime : relative sampling time difference
                unit=np.timedelta64(XX, 'ns')

    References
    ----------
    .. [1] Krauß, et al. "Traffic flow estimation from single
       satellite images." Archives of the ISPRS, vol.XL-1/WL3, 2013.

    Notes
    -----
    The detector arrays of the satellite are configured as follows:

        .. code-block:: text
                78 mm
          <-------------->
          +--------------+    #*# satellite
          |      red     |     |
          +--------------+     | flight
          |   red edge   |     | direction
          +--------------+     |
          |near infrared |     v
          +--------------+
          |              |
          |              |
          |              |
          |              |
          +--------------+  ^
          |  not in use  |  | 6.5 μm
          +--------------+  v
          |   green      |
          +--------------+
          |   blue       |
          +--------------+

    Example
    -------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue']
    >>> re_df = list_central_wavelength_re()
    >>> re_df = re_df[re_df['name'].isin(boi)]
    >>> re_df
         wavelength  bandwidth  resolution   name  bandid  irradiance
    B01         475         70           5   blue       0      1997.8
    B02         555         70           5  green       1      1863.5
    B03         657         55           5    red       2      1560.4

    similarly you can also select by bandwidth:

    >>> re_df = list_central_wavelength_re()
    >>> sb_df = re_df[re_df['bandwidth']<=60]
    >>> sb_df.index
    Index(['B03', 'B04'], dtype='object')

    """
    wavelength = {"B01": 475, "B02": 555, "B03": 657,
                  "B04": 710, "B05": 805,
                  }
    bandwidth = {"B01": 70, "B02": 70, "B03": 55, "B04": 40, "B05": 90,
                 }
    bandid = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,
                  }
    resolution = {"B01": 5, "B02": 5, "B03": 5, "B04": 5, "B05": 5,
                  }
    name = {"B01" : 'blue',         "B02" : 'green',
            "B03" : 'red',          "B04" : 'red edge',
            "B05" : 'near infrared',
            }
    irradiance = {"B01": 1997.8, "B02": 1863.5,
                  "B03": 1560.4, "B04": 1395.0,
                  "B05": 1124.4,
                  }
    detectortime = {"B01": np.timedelta64(000, 'ms'),
                    "B02": np.timedelta64(410, 'ms'),
                    "B03": np.timedelta64(820, 'ms'),
                    "B04": np.timedelta64(2650, 'ms'),
                    "B05": np.timedelta64(3060, 'ms'),
                    } # estimates, see [1]
    d = {
         "wavelength": pd.Series(wavelength),
         "bandwidth": pd.Series(bandwidth),
         "resolution": pd.Series(resolution),
         "name": pd.Series(name),
         "bandid": pd.Series(bandid),
         "irradiance": pd.Series(irradiance),
         "detectortime": pd.Series(detectortime)
         }
    df = pd.DataFrame(d)
    return df

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
