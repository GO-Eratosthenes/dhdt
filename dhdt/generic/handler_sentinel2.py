import glob
import os

from xml.etree import ElementTree
from osgeo import ogr, osr

import numpy as np
import pandas as pd

import geopandas

def get_s2_dict(s2_df):
    """

    Parameters
    ----------
    s2_df : pd.dataframe
        metadata and general multi spectral information about the MSI
        instrument that is onboard Sentinel-2

    Returns
    -------
    s2_dict : dictionary
        dictionary with scene specific meta data, giving the following
        information:
        * 'COSPAR' : string
            id of the satellite
        * 'NORAD' : string
            id of the satellite
        * 'instruments' : {'MSI'}
            specific instrument
        * 'launch_date': string, e.g.:'+2015-06-23'
            date when the satellite was launched into orbit
        * 'orbit': string
            specifies the type of orbit
        * 'full_path': string
            the location where the root folder is situated
        * 'MTD_DS_path': string
            the location where metadata about the datastrip is present
        * 'MTD_TL_path': string
            the location where metadata about the tile is situated
        * 'QI_DATA_path': string
            the location where metadata about detector readings are present
        * 'IMG_DATA_path': string
            the location where the imagery is present
        * 'date': string, e.g.: '+2019-10-25'
            date of acquisition
        * 'tile_code': string, e.g.: 'T05VMG'
            MGRS tile coding
        * 'relative_orbit': integer
            orbit from which the imagery were taken

    Notes
    -----
    The metadata is scattered over a folder structure for Sentinel-2, it has
    typically the following set-up:

        .. code-block:: text

        * S2X_MSIL1C_20XX...   <- path is given in s2_dict["full_path"]
        ├ AUX_DATA
        ├ DATASTRIP
        │  └ DS_XXX_XXXX...    <- path is given in s2_dict["MTD_DS_path"]
        │     └ QI_DATA
        │        └ MTD_DS.xml  <- metadata about the data-strip
        ├ GRANULE
        │  └ L1C_TXXXX_XXXX... <- path is given in s2_dict["MTD_TL_path"]
        │     ├ AUX_DATA
        │     ├ IMG_DATA
        │     │  ├ TXXX_X..XXX.jp2
        │     │  └ TXXX_X..XXX.jp2
        │     ├ QI_DATA
        │     └ MTD_TL.xml     <- metadata about the tile
        ├ HTML
        ├ rep_info
        ├ manifest.safe
        ├ INSPIRE.xml
        └ MTD_MSIL1C.xml       <- metadata about the product

    Nomenclature
    ------------
    AUX : auxiliary
    COSPAR : committee on space research international designator
    DS : datastrip
    IMG : imagery
    L1C : product specification,i.e.: level 1, processing step C
    MGRS : military grid reference system
    MTD : metadata
    MSI : multi spectral instrument
    NORAD : north american aerospace defense satellite catalog number
    TL : tile
    QI : quality information
    s2 : Sentinel-2

    """
    assert isinstance(s2_df, pd.DataFrame), ('please provide a dataframe')
    assert 'filepath' in s2_df, ('please first run "get_S2_image_locations"'+
                                ' to find the proper file locations')

    s2_folder = list(filter(lambda x: '.SAFE' in x,
                            s2_df.filepath[0].split('/')))[0]
    if s2_folder[:3] in 'S2A':
        s2_dict = list_platform_metadata_s2a()
    else:
        s2_dict = list_platform_metadata_s2b()

    im_path = s2_df.filepath[0]
    root_path = '/'.join(im_path.split('/')[:-4])

    ds_folder = list(filter(lambda x: 'DS' in x[:2],
                            os.listdir(os.path.join(root_path,
                                                    'DATASTRIP'))))[0]
    tl_path = '/'.join(im_path.split('/')[:-2])
    s2_dict.update({'full_path': root_path,
                    'MTD_DS_path': os.path.join(root_path,
                                                'DATASTRIP', ds_folder),
                    'MTD_TL_path': tl_path,
                    'IMG_DATA_path': os.path.join(tl_path, 'IMG_DATA'),
                    'QI_DATA_path': os.path.join(tl_path, 'QI_DATA'),
                    'date': meta_S2string(s2_folder)[0],
                    'tile_code': meta_S2string(s2_folder)[2],
                    'relative_orbit': int(meta_S2string(s2_folder)[1][1:])})
    return s2_dict

def list_platform_metadata_s2a():
    s2a_dict = {
        'COSPAR': '2015-028A',
        'NORAD': 40697,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch_date': '+2015-06-23',
        'orbit': 'sso'}
    return s2a_dict

def list_platform_metadata_s2b():
    s2b_dict = {
        'COSPAR': '2017-013A',
        'NORAD': 42063,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch_date': '+2017-03-07',
        'orbit': 'sso'}
    return s2b_dict

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
