# generic libraries
import glob

import numpy as np
import pandas as pd

# geospatial libaries
from osgeo import gdal, osr

def list_platform_metadata_tr():
    tr_dict = {
        'COSPAR': '1999-068A',
        'NORAD': 25994,
        'instruments': {'ASTER','MODIS','MISR'},
        'launch': '1999-12-18',
        'constellation': 'landsat',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:30'}
    return tr_dict

def list_central_wavelength_as():
    """ create dataframe with metadata about Terra's ASTER

    Returns
    -------
    df : pandas.DataFrame
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2, having the following collumns:

            * wavelength : central wavelength of the band
            * bandwidth : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * name : general name of the band, if applicable
            * irradiance :

    Example
    -------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue']
    >>> as_df = list_central_wavelength_as()
    >>> as_df = as_df[as_df['name'].isin(boi)]
    >>> as_df
         wavelength  bandwidth  resolution   name  bandid  irradiance
    B01         560         80          15   blue       0      1848.0
    B02         660         60          15  green       1      1549.0
    B3B         810        100          15    red       2      1114.0
    B3N         810        100          15    red       3      1114.0

    similarly you can also select by pixel resolution:

    >>> as_df = list_central_wavelength_as()
    >>> sw_df = as_df[as_df['resolution']==30]
    >>> sw_df.index
    Index(['B04', 'B05', 'B06', 'B07', 'B08', 'B09'], dtype='object')

    """
    center_wavelength = {"B01": 560, "B02": 660, "B3B": 810, "B3N": 810,
                         "B04":1650, "B05":2165, "B06":2205, "B07":2360,
                         "B08":2330, "B09":2395, "B10":8300, "B11":8650,
                         "B12":9100, "B13":10600,"B14":11300,}
    full_width_half_max = {"B01": 80, "B02": 60, "B3B":100, "B3N":100,
                           "B04":100, "B05": 40, "B06": 40, "B07": 50,
                           "B08": 70, "B09": 70, "B10":350, "B11":350,
                           "B12":350, "B13":700, "B14":700,}
    bandid = {"B01": 0, "B02": 1, "B3B": 2, "B3N": 3,
              "B04": 4, "B05": 5, "B06": 6, "B07": 7, "B08": 8,
              "B09": 9, "B10":10, "B11":11, "B12":12,
              "B13": 13, "B14": 14, }
    gsd = {"B01": 15, "B02": 15, "B3B": 15, "B3N": 15,
           "B04": 30, "B05": 30, "B06": 30, "B07": 30,
           "B08": 30, "B09": 30, "B10": 90, "B11": 90,
           "B12": 90, "B13": 90, "B14": 90, }
    along_track_view_angle = {"B01": 0., "B02": 0., "B3B": -27.6, "B3N": 0.,
                 "B04": 0., "B05": 0., "B06": 0., "B07": 0.,
                 "B08": 0., "B09": 0., "B10": 0., "B11": 0.,
                 "B12": 0., "B13": 0., "B14": 0., }
    solar_illumination = {"B01": 1848., "B02": 1549., "B3B": 1114.,
                          "B3N": 1114., "B04": 225.4, "B05": 86.63,
                          "B06": 81.85, "B07": 74.85, "B08": 66.49,
                          "B09": 59.85, "B10": 0, "B11": 0, "B12": 0,
                          "B13": 0, "B14": 0, }
    common_name = {"B01" : 'blue',  "B02" : 'green',   "B3B" : 'red',
                   "B3N" : 'stereo',"B04" : 'swir16',  "B05" : 'swir22',
                   "B06" : 'swir22',"B07" : 'swir22',  "B08" : 'swir22',
                   "B09" : 'swir22',"B10" : 'lwir',    "B11" : 'lwir',
                   "B12" : 'lwir',  "B13" : 'lwir',    "B13" : 'lwir',
                  }
    d = {
         "center_wavelength": pd.Series(center_wavelength),
         "full_width_half_max": pd.Series(full_width_half_max),
         "gsd": pd.Series(gsd),
         "along_track_view_angle": pd.Series(along_track_view_angle),
         "common_name": pd.Series(common_name),
         "bandid": pd.Series(bandid),
         "solar_illumination": pd.Series(solar_illumination)
         }
    df = pd.DataFrame(d)
    return df

def read_band_as(path, band='3N'):
    """
    This function takes as input the ASTER L1T hdf-filename and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array, plus associated geographical/projection metadata.

    Parameters
    ----------
    path : string
        path of the folder, or full path with filename as well
    band : string, optional
        Terra ASTER band index, for example '02', '3N'.

    Returns
    -------
    data : numpy.array, size=(_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    References
    ----------
    https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/ASTERHigherLevelUserGuideVer2May01.pdf
    https://www.cmascenter.org/ioapi/documentation/all_versions/html/GCTP.html

    """
    assert len(glob.glob(path))!=0, ('file does not seem to be present')

    if len(band)==3:
        band = band[1:]
    # get imagery data
    img = gdal.Open(glob.glob(path)[0], gdal.GA_ReadOnly)
    ds = img.GetSubDatasets()
    for idx,arr in enumerate(ds):
        arr_num = arr[0].split('ImageData')
        if (len(arr_num)>1) and (arr_num[-1] == band):
            pointer = idx
    data = gdal.Open(img.GetSubDatasets()[pointer][0], gdal.GA_ReadOnly).ReadAsArray()

    # get projection data
    ul_xy = np.fromstring(img.GetMetadata()['UpperLeftM'.upper()], sep=', ')[::-1]
    lr_xy = np.fromstring(img.GetMetadata()['LowerRightM'.upper()], sep=', ')[::-1]
    proj_str = img.GetMetadata()['MapProjectionName'.upper()]
    utm_code = img.GetMetadata()['UTMZoneCode1'.upper()]

    if proj_str == 'Universal Transverse Mercator':
        epsg_code = 32600
        epsg_code += int(utm_code)
        center_lat = np.fromstring(img.GetMetadata()['SceneCenter'.upper()],
                                   sep=', ')[0]
        if center_lat<0: # majprity is in southern hemisphere
            epsg_code += 100

        targetprj = osr.SpatialReference()
        targetprj.ImportFromEPSG(epsg_code)
        spatialRef = targetprj.ExportToWkt()
        geoTransform = (ul_xy[0], 15., 0., ul_xy[1], 0., -15.,
                        data.shape[0], data.shape[1])
    else:
        raise('projection not implemented yet')

    return data, spatialRef, geoTransform, targetprj

def read_stack_as(path, as_df):
    """

    Parameters
    ----------
    path : string
        location where the imagery data is situated
    as_df : pandas.DataFrame
        metadata and general multispectral information about the instrument that
        is onboard Terra

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

    See Also
    --------
    list_central_wavelength_as
    """
    for idx, val in enumerate(as_df.index):
        # resolve index and naming issue for ASTER metadata
        if val[-1] in ['N', 'B']:
            boi = val[1:]
        else:
            boi = str(int(val[1:])) # remove leading zero if present

        if idx==0:
            im_stack, spatialRef, geoTransform, targetprj = read_band_as(path, band=boi)
        else: # stack others bands
            if im_stack.ndim==2:
                im_stack = im_stack[:,:,np.newaxis]
            band,_,_,_ = read_band_as(path, band=boi)
            band = band[:,:,np.newaxis]
            im_stack = np.concatenate((im_stack, band), axis=2)
    return im_stack, spatialRef, geoTransform, targetprj

#utc = img.GetMetadata()['CalendarDate'.upper()] + \
#      img.GetMetadata()['TimeOfDay'.upper()][:-1]
#utc = np.array(utc[0:4]+'-'+utc[4:6]+'-'+utc[6:8]+'T'+\
#      utc[8:10]+':'+utc[10:12]+':'+utc[12:14]+\
#      '.'+utc[14:], dtype='datetime64')
