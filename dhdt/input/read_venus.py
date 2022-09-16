# function to get the basic information from the venµs imagery data

# generic libraries
import glob
import os

from xml.etree import ElementTree

import numpy as np
import pandas as pd

from ..generic.mapping_io import read_geo_image

def list_platform_metadata_vn():
    vn_dict = {
        'COSPAR': '2017-044B',
        'NORAD': 42901,
        'full_name': 'Vegetation and Environment monitoring on a New Micro-Satellite',
        'instruments': {'VSSC'}, # VENµS SuperSpectral Camera
        'constellation': 'VENµS',
        'launch': '2017-08-02',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:30'}
    return vn_dict

def list_central_wavelength_vssc():
    center_wavelength = {"B1": 420, "B2" : 443, "B3" : 490, "B4" : 555,
                         "B5": 620, "B6" : 620, "B7" : 667, "B8" : 702,
                         "B9": 742, "B10": 782, "B11": 865, "B12": 910,
                        }
    full_width_half_max = {"B1": 40, "B2" : 40, "B3" : 40, "B4" : 40,
                           "B5": 40, "B6" : 40, "B7" : 30, "B8" : 24,
                           "B9": 16, "B10": 16, "B11": 40, "B12": 20,
                          }
    gsd = {"B1" : 5., "B2" : 5., "B3" : 5., "B4" : 5., "B5": 5., "B6" : 5.,
           "B7" : 5., "B8" : 5., "B9": 5., "B10": 5., "B11": 5., "B12": 5.,
           }
    bandid = {"B1": 'B1', "B2" : 'B2', "B3" : 'B3', "B4" : 'B4',
              "B5": 'B5', "B6" : 'B6', "B7" : 'B7', "B8" : 'B8',
              "B9": 'B9', "B10":'B10', "B11":'B11', "B12":'B12',
              }
    # along_track_view_angle =
    common_name = {"B1": 'coastal',    "B2" : 'coastal',
                   "B3" : 'blue',      "B4" : 'green',
                   "B5" : 'stereo',    "B6" : 'stereo',
                   "B7" : 'red',       "B8" : 'rededge',
                   "B9" : 'rededge',   "B10": 'rededge',
                   "B11": 'nir08',     "B12": 'nir09',
                  }
    d = {
         "center_wavelength": pd.Series(center_wavelength),
         "full_width_half_max": pd.Series(full_width_half_max),
         "gsd": pd.Series(gsd),
         "common_name": pd.Series(common_name),
         "bandid": pd.Series(bandid)
         }
    df = pd.DataFrame(d)
    return df

def read_band_vn(path, band='00'):

    if band!='00':
        fname = os.path.join(path, '*SRE_'+band+'.tif')
    else:
        fname = path
    f_full = glob.glob(fname)[0]
    assert os.path.exists(f_full)

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(f_full)

    return data, spatialRef, geoTransform, targetprj
    
# def read_sun_angles_vn(path)
def read_view_angles_vn(path):
    assert isinstance(path, str)
    fname = os.path.join(path, 'DATA', 'VENUS*UII_ALL.xml')
    f_full = glob.glob(fname)[0]
    assert os.path.exists(f_full)

    dom = ElementTree.parse(f_full)
    root = dom.getroot()
    
    ul_xy = np.array([float(root[0][1][2][0][2].text),
                      float(root[0][1][2][0][3].text)])
    ur_xy = np.array([float(root[0][1][2][1][2].text),
                      float(root[0][1][2][1][3].text)])
    lr_xy = np.array([float(root[0][1][2][2][2].text),
                      float(root[0][1][2][2][3].text)])
    ll_xy = np.array([float(root[0][1][2][3][2].text),
                      float(root[0][1][2][3][3].text)])
    # hard coded for detector_id="01"
    ul_za = np.array([float(root[0][2][0][1][0][0].text),
                      float(root[0][2][0][1][0][1].text)])
    ur_za = np.array([float(root[0][2][0][1][1][0].text),
                      float(root[0][2][0][1][1][1].text)])
    lr_za = np.array([float(root[0][2][0][1][2][0].text),
                      float(root[0][2][0][1][2][1].text)])
    ll_za = np.array([float(root[0][2][0][1][3][0].text),
                      float(root[0][2][0][1][3][1].text)])
        
    
    Az, Zn = 0,0
    return Az, Zn

def read_mean_sun_angles_vn(path):
    assert isinstance(path, str)
    fname = os.path.join(path, 'VENUS*MTD_ALL.xml')
    f_full = glob.glob(fname)[0]
    assert os.path.exists(f_full)

    dom = ElementTree.parse(f_full)
    root = dom.getroot()

    Zn = float(root[5][0][0][0].text)
    Az = float(root[5][0][0][1].text)
    return Zn, Az

def read_mean_view_angle_vn(path):
    assert isinstance(path, str)
    fname = os.path.join(path, 'DATA', 'VENUS*UII_ALL.xml')
    f_full = glob.glob(fname)[0]
    assert os.path.exists(f_full)

    dom = ElementTree.parse(f_full)
    root = dom.getroot()
    
    # hard coded for detector_id="01"
    Zn = float(root[0][2][0][1][4][0].text)
    Az = float(root[0][2][0][1][4][1].text)
    return Az, Zn

