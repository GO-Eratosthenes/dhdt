import glob
import os

import numpy as np
import pandas as pd

from xml.etree import ElementTree

from ..generic.mapping_io import read_geo_image


def list_platform_metadata_re1():
    re1_dict = {
        'COSPAR': '2008-040C',
        'NORAD': 33314,
        'instruments': {'REIS'},  # RapidEye Earth Imaging System
        'constellation': 'rapideye',
        'launch': '2008-08-29',
        'orbit': 'sso'
    }
    return re1_dict


def list_platform_metadata_re2():
    re2_dict = {
        'COSPAR': '2008-040A',
        'NORAD': 33312,
        'instruments': {'REIS'},
        'constellation': 'rapideye',
        'launch': '2008-08-29',
        'orbit': 'sso'
    }
    return re2_dict


def list_platform_metadata_re3():
    re3_dict = {
        'COSPAR': '2008-040D',
        'NORAD': 33315,
        'instruments': {'REIS'},
        'constellation': 'rapideye',
        'launch': '2008-08-29',
        'orbit': 'sso'
    }
    return re3_dict


def list_platform_metadata_re4():
    re4_dict = {
        'COSPAR': '2008-040E',
        'NORAD': 33316,
        'instruments': {'REIS'},  # RapidEye Earth Imaging System
        'constellation': 'rapideye',
        'launch': '2008-08-29',
        'orbit': 'sso'
    }
    return re4_dict


def list_platform_metadata_re5():
    re5_dict = {
        'COSPAR': '2008-040B',
        'NORAD': 33313,
        'instruments': {'REIS'},  # RapidEye Earth Imaging System
        'constellation': 'rapideye',
        'launch': '2008-08-29',
        'orbit': 'sso'
    }
    return re5_dict


def list_central_wavelength_re():
    """ create dataframe with metadata about RapidEye

    Returns
    -------
    df : pandas.dataframe
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2, having the following collumns:

            * center_wavelength, unit=µm : central wavelength of the band
            * full_width_half_max, unit=µm : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution, unit=m : spatial resolution of a pixel
            * common_name : general name of the band, if applicable
            * irradiance, unit=W m-2 μm-1 : exo-atmospheric radiance
            * relative_timing, unit=ms : relative sampling time difference

    Notes
    -----
    The detector arrays of the satellite are configured as follows[Kr13]_:

        .. code-block:: text

                78 mm
          <-------------->
          ┌--------------┐    #*# satellite
          |      red     |     |
          ├--------------┤     | flight
          |   red edge   |     | direction
          ├--------------┤     |
          |near infrared |     v
          ├--------------┤
          |              |
          |              |
          |              |
          |              |
          ├--------------┤  ^
          |  not in use  |  | 6.5 μm
          ├--------------┤  v
          |   green      |
          ├--------------┤
          |   blue       |
          └--------------┘

    References
    ----------
    .. [Kr13] Krauß, et al. "Traffic flow estimation from single satellite
              images" Archives of the ISPRS, vol.XL-1/WL3, 2013.

    Examples
    --------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue']
    >>> re_df = list_central_wavelength_re()
    >>> re_df = re_df[re_df['common_name'].isin(boi)]
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
    center_wavelength = {
        "B01": 475,
        "B02": 555,
        "B03": 657,
        "B04": 710,
        "B05": 805,
    }
    # convert from nm to µm
    center_wavelength = {k: v / 1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {
        "B01": 70,
        "B02": 70,
        "B03": 55,
        "B04": 40,
        "B05": 90,
    }
    full_width_half_max = {k: v / 1E3 for k, v in full_width_half_max.items()}

    bandid = {
        "B01": 0,
        "B02": 1,
        "B03": 2,
        "B04": 3,
        "B05": 4,
    }
    resolution = {
        "B01": 5.,
        "B02": 5.,
        "B03": 5.,
        "B04": 5.,
        "B05": 5.,
    }
    name = {
        "B01": 'blue',
        "B02": 'green',
        "B03": 'red',
        "B04": 'red edge',
        "B05": 'near infrared',
    }
    irradiance = {
        "B01": 1997.8,
        "B02": 1863.5,
        "B03": 1560.4,
        "B04": 1395.0,
        "B05": 1124.4,
    }
    relative_timing = {
        "B01": np.timedelta64(000, 'ms'),
        "B02": np.timedelta64(410, 'ms'),
        "B03": np.timedelta64(820, 'ms'),
        "B04": np.timedelta64(2650, 'ms'),
        "B05": np.timedelta64(3060, 'ms'),
    }  # estimates, see [1]__
    d = {
        "center_wavelength": pd.Series(center_wavelength),
        "full_width_half_max": pd.Series(full_width_half_max),
        "resolution": pd.Series(resolution),
        "common_name": pd.Series(name),
        "bandid": pd.Series(bandid),
        "irradiance": pd.Series(irradiance),
        "relative_timing": pd.Series(relative_timing)
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
    fname = os.path.join(path, '*_Analytic*tif')

    bnd_id = int(band) - 1
    assert 0 <= bnd_id <= 4, (
        'please provide correct band number for RapidEye data')

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])
    data = np.squeeze(data[..., bnd_id])
    return data, spatialRef, geoTransform, targetprj


def read_stack_re(path):
    """ read specific band of RapidEye image

    This function takes as input the RapidEye band number and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    path : string
        path to folder with imagery.

    Returns
    -------
    data : np.array, size=(m,n,_), dtype=integer
        data of one RadidEye band.
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    """
    fname = os.path.join(path, '*_Analytic*tif')
    ffull = glob.glob(fname)[0]

    assert os.path.isfile(ffull), ('file does not seem to be present')
    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(ffull)
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
            - <--┼--> +
                 |
                 +----> East & x

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          ├-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          +---- surface         +--┴---


    """
    fname = os.path.join(path, '_metadata.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    mI = int(root[4][0][0][0][5].text)
    nI = int(root[4][0][0][0][6].text)

    Azimuth = float(root[2][0][3][0][2].text)
    Zenith = 90 - float(root[2][0][3][0][3].text)

    Zn = Zenith * np.ones((mI, nI))
    Az = Azimuth * np.ones((mI, nI))

    return Zn, Az
