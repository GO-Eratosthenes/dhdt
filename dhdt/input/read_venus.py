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
    """ create dataframe with metadata about VENµS

    Returns
    -------
    df : pandas.dataframe
        metadata and general multispectral information about the  SuperSpectral
        Camera that is onboard VENµS, having the following collumns:

            * center_wavelength, unit=µm : central wavelength of the band
            * full_width_half_max, unit=µm : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * gsd, unit=m : spatial resolution of a pixel
            * common_name : general name of the band, if applicable

    See Also
    --------
    dhdt.input.read_sentinel2.list_central_wavelength_msi :
        for the instrument data of MSI on Sentinel-2
    dhdt.input.read_landsat.list_central_wavelength_oli :
        for the instrument data of Landsat 8 or 9
    dhdt.input.read_aster.list_central_wavelength_as :
        for the instrument data of ASTER on the Terra satellite

    Notes
    -----
    The sensor is composed of four detectors, which each have three arrays.
    These multi-spectral bands (BXX) are arranged as follows:

        .. code-block:: text

          ┌-----B06------┐                      #*# satellite
          ├-----B04------┤                      | flight
          └-----B03------┘ detector id: IV      | direction
          ┌-----B09------┐                      v
          ├-----B08------┤
          └-----B07------┘ detector id: III
          ┌-----B11------┐
          ├-----B12------┤
          └-----B10------┘ detector id: II
          ┌-----B01------┐
          ├-----B02------┤
          └-----B05------┘ detector id: I

    References
    -------
    .. [1] Dick, et al. "VENμS: mission characteristics, final evaluation of the
           first phase and data production" Remote sensing vol.14(14) pp.3281,
           2022.
    """
    center_wavelength = {"B1": 420, "B2" : 443, "B3" : 490, "B4" : 555,
                         "B5": 620, "B6" : 620, "B7" : 667, "B8" : 702,
                         "B9": 742, "B10": 782, "B11": 865, "B12": 910,
                        }
    # convert from nm to µm
    center_wavelength = {k: v/1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {"B1": 40, "B2" : 40, "B3" : 40, "B4" : 40,
                           "B5": 40, "B6" : 40, "B7" : 30, "B8" : 24,
                           "B9": 16, "B10": 16, "B11": 40, "B12": 20,
                          }
    # convert from nm to µm
    full_width_half_max = {k: v/1E3 for k, v in full_width_half_max.items()}
    gsd = {"B1" : 5., "B2" : 5., "B3" : 5., "B4" : 5., "B5": 5., "B6" : 5.,
           "B7" : 5., "B8" : 5., "B9": 5., "B10": 5., "B11": 5., "B12": 5.,
           }
    bandid = {"B1": 'B1', "B2" : 'B2', "B3" : 'B3', "B4" : 'B4',
              "B5": 'B5', "B6" : 'B6', "B7" : 'B7', "B8" : 'B8',
              "B9": 'B9', "B10":'B10', "B11":'B11', "B12":'B12',
              }
    detector_id = {"B1": 1, "B2" : 1, "B3" : 4, "B4" : 4,
                   "B5": 1, "B6" : 4, "B7" : 3, "B8" : 3,
                   "B9": 3, "B10": 2, "B11": 2, "B12": 2,
                  }
    acquisition_order = {"B1":10, "B2" :11, "B3" : 3, "B4" : 2,
                         "B5":12, "B6" : 1, "B7" : 6, "B8" : 5,
                         "B9": 4, "B10": 9, "B11": 7, "B12": 8,
                        }
    common_name = {"B1": 'coastal',    "B2" : 'coastal',
                   "B3" : 'blue',      "B4" : 'green',
                   "B5" : 'stereo',    "B6" : 'stereo',
                   "B7" : 'red',       "B8" : 'rededge',
                   "B9" : 'rededge',   "B10": 'rededge',
                   "B11": 'nir08',     "B12": 'nir09',
                  }
    d = {
         "center_wavelength": pd.Series(center_wavelength,
                                        dtype=np.dtype('float')),
         "full_width_half_max": pd.Series(full_width_half_max,
                                          dtype=np.dtype('float')),
         "gsd": pd.Series(gsd,
                          dtype=np.dtype('float')),
         "common_name": pd.Series(common_name,
                                  dtype=np.dtype('str')),
         "bandid": pd.Series(bandid,
                             dtype=np.dtype('str')),
         "detector_id": pd.Series(detector_id,
                                  dtype=np.dtype('int64')),
         "acquisition_order": pd.Series(acquisition_order,
                                        dtype=np.dtype('int64')),
         }
    df = pd.DataFrame(d)
    return df

def read_band_vn(path, band=None):

    if isinstance(band, str):
        fname = os.path.join(path, '*SRE_'+band+'.tif')
    elif isinstance(band,int):
        fname = os.path.join(path, '*SRE_'+ str(band).zfill(2) + '.tif')
    else:
        fname = path
    f_full = glob.glob(fname)[0]
    assert os.path.exists(f_full)

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(f_full)

    return data, spatialRef, geoTransform, targetprj

def read_stack_vn(vn_df):
    """
    read imagery data of interest into an three dimensional np.array

    Parameters
    ----------
    vn_df : pandas.Dataframe
        metadata and general multispectral information about the VSSC
        instrument that is onboard VENµS

    Returns
    -------
    im_stack : numpy.array, size=(_,_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple, size=(8,1)
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    """
    assert isinstance(vn_df, pd.DataFrame), ('please provide a dataframe')
    assert 'imagepath' in vn_df, ('please first run "get_vn_image_locations"' +
                                 ' to find the proper file locations')

    # start with the highest resolution
    for val, idx in enumerate(vn_df.sort_values('bandid').index):
        full_path = vn_df['imagepath'][idx]
        if val == 0:
            im_stack, spatialRef, geoTransform, targetprj = read_band_vn(full_path)
        else:  # stack others bands
            if im_stack.ndim == 2:
                im_stack = np.atleast_3d(im_stack)
            band = np.atleast_3d(read_band_vn(full_path)[0])
            im_stack = np.concatenate((im_stack, band), axis=2)

    # in the meta data files there is no mention of its size, hence include
    # it to the geoTransform
    if len(geoTransform) == 6:
        geoTransform = geoTransform + (im_stack.shape[0], im_stack.shape[1])
    return im_stack, spatialRef, geoTransform, targetprj

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

