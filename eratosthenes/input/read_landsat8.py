import glob
import os

import numpy as np
import pandas as pd

from ..generic.mapping_io import read_geo_image

def list_platform_metadata_l7():
    l7_dict = {
        'COSPAR': '1999-020A',
        'NORAD': 25682,
        'instruments': {'ETM+'},
        'launch': '1999-04-15',
        'constellation': 'landsat',
        'orbit': 'sso'}
    return l7_dict

def list_platform_metadata_l8():
    # following https://github.com/stac-extensions/sat
    l8_dict = {
        'COSPAR': '2013-008A', # sat:platform_international_designator
        'NORAD': 39084,
        'instruments': {'OLI','TIRS'},
        'constellation': 'landsat',
        'launch': '2013-02-11',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:11'}
    return l8_dict

def list_platform_metadata_l9():
    l9_dict = {
        'COSPAR': '2021-088A',
        'NORAD': 49260,
        'instruments': {'OLI','TIRS'},
        'constellation': 'landsat',
        'launch': '2021-09-27',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:00'}
    return l9_dict

def list_central_wavelength_oli():
    """ create dataframe with metadata about OLI on Landsat8 or Landsat9

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
    >>> l8_df = list_central_wavelength_oli()
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
# https://schemas.stacspec.org/v1.0.0/item-spec/json-schema/instrument.json
# following the naming of pystac.extensions.eo???
# https://stac-extensions.github.io/eo/v1.0.0/schema.json
    # center_wavelength = (min_wavelength + max_wavelength) / 2
    center_wavelength = {"B1": .443, "B2": .482, "B3": .561, "B4": .655,
                         "B5": .865, "B6":1.609, "B7":2.201, "B8": .590,
                         "B9":1.373, "B10":10.896,"B11":12.001,
                        }
    # full_width_half_max = max_wavelength - min_wavelength
    full_width_half_max = {"B1": 0.016, "B2": 0.060, "B3": 0.057, "B4": 0.037,
                 "B5": 0.028, "B6": 0.085, "B7": 0.187, "B8": 0.172,
                 "B9": 0.020, "B10": .588, "B11":1.011,
                 }
    bandid = {"B1": 1, "B2": 2,  "B3": 3, "B4": 4,
              "B5": 5, "B6": 6,  "B7": 7, "B8": 8,
              "B9": 9, "B10":10, "B11":11,
                  }
    gsd = {"B1": 30, "B2": 30,  "B3": 30, "B4": 30,
           "B5": 30, "B6": 30,  "B7": 30, "B8": 15,
           "B9": 30, "B10":100, "B11":100,
          }
    field_of_view = {"B1": 15., "B2" : 15., "B3" : 15., "B4": 15.,
                     "B5": 15., "B6" : 15., "B7" : 15., "B8": 15.,
                     "B9": 15., "B10": 15., "B11": 15.}
# "common_name": {"coastal", "blue", "green", "red", "rededge", "yellow",
#            "pan", "nir", "nir08", "nir09", "cirrus", "swir16", "swir22",
#            "lwir", "lwir11", "lwir12" }
    common_name = {"B1": 'coastal',       "B2" : 'blue',
            "B3" : 'green',        "B4" : 'red',
            "B5" : 'nir08',        "B6" : 'swir16',
            "B7" : 'swir22',       "B8" : 'pan',
            "B9" : 'cirrus',       "B10": 'lwir11',
            "B11": 'lwir12',
            }
    d = {
         "center_wavelength": pd.Series(center_wavelength),
         "full_width_half_max": pd.Series(full_width_half_max),
         "gsd": pd.Series(gsd),
         "common_name": pd.Series(common_name),
         "field_of_view": pd.Series(field_of_view),
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

def read_metadata_l8(dir_path, suffix='MTL.txt'):
    """ Convert Landsat MTL file to dictionary of metadata values """
    full_path = os.path.join(dir_path, suffix)
    with open('readme.txt') as f:
        mtl_text = f.readlines()

    mtl = {}
    for line in mtl_text.split('\n'):
        meta = line.replace('\"', "").strip().split('=')
        if len(meta) > 1:
            key = meta[0].strip()
            item = meta[1].strip()
            if key != "GROUP" and key != "END_GROUP":
                mtl[key] = item
    return mtl


# 14 sensors
# detector samping time: 4.2 msec
# total staring time:    1.2 sec
#def
