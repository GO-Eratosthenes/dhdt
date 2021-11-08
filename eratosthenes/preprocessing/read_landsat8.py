import glob
import os

import numpy as np
import pandas as pd

from ..generic.mapping_io import read_geo_image

def list_central_wavelength_l8():
    """ create dataframe with metadata about Landsat8 or Landsat9

    Returns
    -------
    df : datafram
        metadata and general multispectral information about the OLI
        instrument that is onboard Landsat8, having the following collumns:

            * wavelength : central wavelength of the band
            * bandwidth : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * name : general name of the band, if applicable

    Example
    -------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue', 'near infrared']
    >>> l8_df = list_central_wavelength_l8()
    >>> l8_df = l8_df[l8_df['name'].isin(boi)]
    >>> l8_df
             wavelength  bandwidth  resolution           name  bandid
    B02         482         60          30           blue       2
    B03         561         57          30          green       3
    B04         655         37          30            red       4
    B05         865         28          30  near infrared       5

    similarly you can also select by pixel resolution:

    >>> l8_df = list_central_wavelength_l8()
    >>> pan_df = l8_df[l8_df['resolution']==15]
    >>> pan_df.index
    Index(['B08'], dtype='object')

    """
    wavelength = {"B1": 443, "B2": 482, "B3": 561, "B4": 655,
                  "B5": 865, "B6":1609, "B7":2201, "B8": 590,
                  "B9":1373, "B10":10896,"B11":12001,
                  }
    bandwidth = {"B1": 16, "B2": 60, "B3": 57, "B4": 37,
                 "B5": 28, "B6": 85, "B7":187, "B8":172,
                 "B9": 20, "B10":588, "B11":1011,
                 }
    bandid = {"B1": 1, "B2": 2,  "B3": 3, "B4": 4,
              "B5": 5, "B6": 6,  "B7": 7, "B8": 8,
              "B9": 9, "B10":10, "B11":11,
                  }
    resolution = {"B1": 30, "B2": 30,  "B3": 30, "B4": 30,
                  "B5": 30, "B6": 30,  "B7": 30, "B8": 15,
                  "B9": 30, "B10":100, "B11":100,
                  }
    name = {"B1": 'aerosol',       "B2" : 'blue',
            "B3" : 'green',        "B4" : 'red',
            "B5" : 'near infrared',"B6" : 'shortwave infrared',
            "B7" : 'shortwave infrared',
            "B8" : 'panchromatic',
            "B9" : 'cirrus',       "B10": 'thermal',
            "B11": 'thermal',
            }
    d = {
         "wavelength": pd.Series(wavelength),
         "bandwidth": pd.Series(bandwidth),
         "resolution": pd.Series(resolution),
         "name": pd.Series(name),
         "bandid": pd.Series(bandid)
         }
    df = pd.DataFrame(d)
    return df

def read_band_l8(path, band='B8'):
    """
    This function takes as input the Landsat8 band name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    path : string
        path of the folder, or full path with filename as well
    band : string, optional
        Landsat8 band name, for example '4', 'B8'.

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
    list_central_wavelength_l8

    Example
    -------
    >>> path = '/LC08_L1TP_018060_20200904_20200918_02_T1/'
    >>> band = '2'
    >>> _,spatialRef,geoTransform,targetprj = read_band_l8(path, band)

    >>> spatialRef
   'PROJCS["WGS 84 / UTM zone 15N",GEOGCS["WGS 84",DATUM["WGS_1984", ...
    >>> geoTransform
    (626385.0, 30.0, 0.0, 115815.0, 0.0, -30.0)
    >>> targetprj
    <osgeo.osr.SpatialReference; proxy of <Swig Object of type 'OSRSpatial ...
    """

    if band!='0':
        if len(band)==2: #when band : 'BX'
            fname = os.path.join(path, '*' + band + '.TIF')
        else: # when band: '0X'
            fname = os.path.join(path, '*' + band + '.TIF')
    else:
        fname = path
    assert len(glob.glob(fname))!=0, ('file does not seem to be present')

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])
    return data, spatialRef, geoTransform, targetprj

def read_stack_l8(path, l8_df):

    for idx, val in enumerate(l8_df.index):
        if idx==0:
            im_stack, spatialRef, geoTransform, targetprj = read_band_l8(path, band=val)
        else: # stack others bands
            if im_stack.ndim==2:
                im_stack = im_stack[:,:,np.newaxis]
            band,_,_,_ = read_band_l8(path, band=val)
            band = band[:,:,np.newaxis]
            im_stack = np.concatenate((im_stack, band), axis=2)
    return im_stack, spatialRef, geoTransform, targetprj

def read_view_angles_l8(path):

    # ANGLE_SENSOR_AZIMUTH_BAND_4
    fname = os.path.join(path, '*VAA.TIF')
    assert len(glob.glob(fname))!=0, ('Azimuth file does not seem to be present')

    Az, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])

    # ANGLE_SENSOR_ZENITH_BAND_4
    fname = os.path.join(path, '*VZA.TIF')
    assert len(glob.glob(fname))!=0, ('Zenith file does not seem to be present')

    Zn, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])
    return Zn, Az

def read_sun_angles_l8(path):

    # ANGLE_SOLAR_AZIMUTH_BAND_4
    fname = os.path.join(path, '*SAA.TIF')
    assert len(glob.glob(fname))!=0, ('Azimuth file does not seem to be present')

    Az,_,_,_ = read_geo_image(glob.glob(fname)[0])

    # ANGLE_SOLAR_ZENITH_BAND_4
    fname = os.path.join(path, '*SZA.TIF')
    assert len(glob.glob(fname))!=0, ('Zenith file does not seem to be present')

    Zn,_,_,_ = read_geo_image(glob.glob(fname)[0])
    return Zn, Az

# 14 sensors
# detector samping time: 4.2 msec
# total staring time:    1.2 sec
#def
