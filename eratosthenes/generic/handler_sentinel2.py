import glob
import os

from xml.etree import ElementTree
from osgeo import ogr, osr

import numpy as np
import pandas as pd

import geopandas

def get_parallax_msi():
    """ create dataframe with metadata about Sentinel-2

    References
    -------
    .. [1] Languille et al. "Sentinel-2 geometric image quality commissioning:
       first results" Proceedings of the SPIE, 2015.
    """
    crossband_parallax = {"B01": 443, "B02": 492, "B03": 560, "B04": 665,
                          "B05": 704, "B06": 741, "B07": 783, "B08": 833, "B8A": 865,
                          "B09": 945, "B10": 1374, "B11": 1614, "B12": 2202,
                          }
    order= {"B01": 21, "B02": 1, "B03": 36, "B04": 31,
                           "B05": 15, "B06": 15, "B07": 20, "B08": 2, "B8A": 21,
                           "B09": 20, "B10": 31, "B11": 91, "B12": 175,
                           }
    gsd = {"B01": 60, "B02": 10, "B03": 10, "B04": 10,
           "B05": 20, "B06": 20, "B07": 20, "B08": 10, "B8A": 20,
           "B09": 60, "B10": 60, "B11": 20, "B12": 20,
           }
    common_name = {"B01": 'coastal', "B02": 'blue',
                   "B03": 'green', "B04": 'red',
                   "B05": 'rededge', "B06": 'rededge',
                   "B07": 'rededge', "B08": 'nir',
                   "B8A": 'nir08', "B09": 'nir09',
                   "B10": 'cirrus', "B11": 'swir16',
                   "B12": 'swir22'}
    d = {
        "center_wavelength": pd.Series(center_wavelength),
        "full_width_half_max": pd.Series(full_width_half_max),
        "gsd": pd.Series(gsd),
        "common_name": pd.Series(common_name),
        "bandid": pd.Series(bandid),
        "field_of_view": pd.Series(field_of_view),
        "solar_illumination": pd.Series(solar_illumination)
    }
    df = pd.DataFrame(d)
    return df

def get_bbox_from_tile_code(tile_code, shp_dir=None, shp_name=None):
    if shp_dir is None:
        shp_dir = '/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles'
    if shp_name is None: shp_name='sentinel2_tiles_world.shp'

    tile_code = tile_code.upper()
    shp_path = os.path.join(shp_dir,shp_name)

    mgrs = geopandas.read_file(shp_path)

    toi = mgrs[mgrs['Name']==tile_code]
    return toi

#todo for OGR geometry
def get_geom_for_tile_code(tile_code, shp_dir=None, shp_name=None):
    if shp_dir is None:
        shp_dir = '/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles'
    if shp_name is None: shp_name='sentinel2_tiles_world.shp'

    tile_code = tile_code.upper()
    shp_path = os.path.join(shp_dir,shp_name)

    # open the shapefile of the coast
    shp = ogr.Open(shp_path)
    lyr = shp.GetLayer()

    for feature in lyr:
        if feature.GetField('Name')== tile_code:
            geom = feature.GetGeometryRef()
    lyr = None
    return geom

def get_array_from_xml(treeStruc):
    """
    Arrays within a xml structure are given line per line
    Output is an array
    """
    for i in range(0, len(treeStruc)):
        Trow = [float(s) for s in treeStruc[i].text.split(' ')]
        if i == 0:
            Tn = Trow
        elif i == 1:
            Trow = np.stack((Trow, Trow), 0)
            Tn = np.stack((Tn, Trow[1, :]), 0)
        else:
            # Trow = np.stack((Trow, Trow),0)
            # Tn = np.concatenate((Tn, [Trow[1,:]]),0)
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn

def get_S2_image_locations(fname,s2_df):
    """
    The Sentinel-2 imagery are placed within a folder structure, where one
    folder has an ever changing name. Fortunately this function finds the path from
    the meta data

    Parameters
    ----------
    fname : string
        path string to the Sentinel-2 folder
    s2_df : dataframe
        index of the bands of interest
    Returns
    -------
    im_paths : series, size=(k,)
        dataframe series with relative folder and file locations of the bands
    datastrip_id : str
        folder name of the metadata

    Example
    -------
    >>> import os
    >>> fpath = '/Users/Data/'
    >>> sname = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> fname = 'MTD_MSIL1C.xml'
    >>> full_path = os.path.join(fpath, sname, fname)
    >>> im_paths,datastrip_id = get_S2_image_locations(full_path)
    >>> im_paths
    ['GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B01',
     'GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B02']
    >>> datastrip_id
    'S2A_OPER_MSI_L1C_DS_VGS1_20200923T200821_S20200923T163313_N02.09'
    """
    assert len(glob.glob(fname)) != 0, ('metafile does not seem to be present')
    #todo see if it is a folder or a file
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()
    granule_list = root[0][0][11][0][0]
    datastrip_id = granule_list.get('datastripIdentifier')

    im_paths, band_id = [], []
    for im_loc in root.iter('IMAGE_FILE'):
        full_path = os.path.join(os.path.split(fname)[0],im_loc.text)
        im_paths.append(full_path)
        band_id.append(im_loc.text[-3:])

    band_path = pd.Series(data=im_paths, index=band_id, name="filepath")
    s2_df_new = pd.concat([s2_df, band_path], axis=1, join="inner")
    return s2_df_new, datastrip_id

def meta_S2string(S2str):
    """ get meta information of the Sentinel-2 file name

    Parameters
    ----------
    S2str : string
        filename of the L1C data

    Returns
    -------
    S2time : string
        date "+YYYY-MM-DD"
    S2orbit : string
        relative orbit "RXXX"
    S2tile : string
        tile code "TXXXXX"

    Example
    -------
    >>> S2str = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> S2time, S2orbit, S2tile = meta_S2string(S2str)
    >>> S2time
    '+2020-09-23'
    >>> S2orbit
    'R140'
    >>> S2tile
    'T15MXV'
    """
    assert type(S2str)==str, ("please provide a string")

    if S2str[0:2]=='S2': # some have info about orbit and sensor
        S2split = S2str.split('_')
        S2time = S2split[2][0:8]
        S2orbit = S2split[4]
        S2tile = S2split[5]
    elif S2str[0:1]=='T': # others have no info about orbit, sensor, etc.
        S2split = S2str.split('_')
        S2time = S2split[1][0:8]
        S2tile = S2split[0]
        S2orbit = None
    else:
        assert True, "please provide a Sentinel-2 file string"
    # convert to +YYYY-MM-DD string
    # then it can be in the meta-data of following products
    S2time = '+' + S2time[0:4] + '-' + S2time[4:6] + '-' + S2time[6:8]
    return S2time, S2orbit, S2tile

def get_S2_folders(im_path):
    S2_list = [x for x in os.listdir(im_path)
               if (os.path.isdir(os.path.join(im_path,x))) & (x[0:2]=='S2')]
    return S2_list

def get_tiles_from_S2_list(S2_list):
    tile_list = [x.split('_')[5] for x in S2_list]
    tile_list = list(set(tile_list))
    return tile_list

def get_utm_from_S2_tiles(tile_list):
    utm_list = [int(x[1:3]) for x in tile_list]
    utm_list = list(set(utm_list))
    return utm_list

def get_utmzone_from_mgrs_tile(tile_code):
    return int(tile_code[:2])

def get_crs_from_mgrs_tile(tile_code):
    """

    Parameters
    ----------
    tile_code : string
        US Military Grid Reference System (MGRS) tile code

    Returns
    -------
    crs : osgeo.osr.SpatialReference
        target projection system

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees
    """
    utm_num = get_utmzone_from_mgrs_tile(tile_code)
    epsg_code = 32600 + utm_num

    # N to X are in the Northern hemisphere
    if tile_code[2] < 'N': epsg_code += 100

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(epsg_code)
    return crs
