import os
import re
import glob
import time

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
        * 'tile_code': string, e.g.: '05VMG'
            MGRS tile coding
        * 'relative_orbit': integer, {x ∈ ℕ}
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

    The following acronyms are used:

    - AUX : auxiliary
    - COSPAR : committee on space research international designator
    - DS : datastrip
    - IMG : imagery
    - L1C : product specification,i.e.: level 1, processing step C
    - MGRS : military grid reference system
    - MTD : metadata
    - MSI : multi spectral instrument
    - NORAD : north american aerospace defense satellite catalog number
    - TL : tile
    - QI : quality information
    - s2 : Sentinel-2

    """
    assert isinstance(s2_df, pd.DataFrame), ('please provide a dataframe')
    assert 'filepath' in s2_df, ('please first run "get_S2_image_locations"'+
                                ' to find the proper file locations')

    s2_folder = list(filter(lambda x: '.SAFE' in x,
                            s2_df.filepath[0].split(os.sep)))[0]
    if s2_folder[:3] in 'S2A':
        s2_dict = list_platform_metadata_s2a()
    else:
        s2_dict = list_platform_metadata_s2b()

    im_path = s2_df.filepath[0]
    root_path = os.sep.join(im_path.split(os.sep)[:-4])

    ds_folder = list(filter(lambda x: 'DS' in x[:2],
                            os.listdir(os.path.join(root_path,
                                                    'DATASTRIP'))))[0]
    tl_path = os.sep.join(im_path.split(os.sep)[:-2])
    s2_dict.update({'full_path': root_path,
                    'MTD_DS_path': os.path.join(root_path,
                                                'DATASTRIP', ds_folder),
                    'MTD_TL_path': tl_path,
                    'IMG_DATA_path': os.path.join(tl_path, 'IMG_DATA'),
                    'QI_DATA_path': os.path.join(tl_path, 'QI_DATA'),
                    'date': meta_S2string(s2_folder)[0],
                    'tile_code': meta_S2string(s2_folder)[2][1:],
                    'relative_orbit': int(meta_S2string(s2_folder)[1][1:])})
    return s2_dict

def list_platform_metadata_s2a():
    s2a_dict = {
        'COSPAR': '2015-028A',
        'NORAD': 40697,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch_date': '+2015-06-23',
        'orbit': 'sso',
        'mass': 1129.541, # [kg]
        'inclination': 98.5621, # https://www.n2yo.com/satellite/?s=40697
        'revolutions_per_day': 14.30824258387262,
        'J': np.array([[558, 30, -30],[30, 819, 30],[-30, 30, 1055]])}
    return s2a_dict

def _check_mgrs_code(tile_code):
    assert isinstance(tile_code, str), 'please provide a string'
    tile_code = tile_code.upper()
    assert bool(re.match("[0-9][0-9][A-Z][A-Z][A-Z]", tile_code)), \
        ('please provide a correct MGRS tile code')
    return tile_code

def list_platform_metadata_s2b():
    s2b_dict = {
        'COSPAR': '2017-013A',
        'NORAD': 42063,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch_date': '+2017-03-07',
        'orbit': 'sso',
        'inclination': 98.5664,
        'revolutions_per_day': 14.30818491298178,
        'J': np.array([[558, 30, -30],[30, 819, 30],[-30, 30, 1055]])}
    return s2b_dict

def get_bbox_from_tile_code(tile_code, geom_dir=None, geom_name=None):
    """  get the bounds of a certain MGRS tile

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding
    geom_dir : string
        directory where geometric metadata about Sentinel-2 is situated
    geom_name : string
        filename of metadata, e.g.: 'sentinel2_tiles_world.geojson'

    Returns
    -------
    bbox : numpy.array, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y
    """
    tile_code = _check_mgrs_code(tile_code)
    if geom_dir is None:
        rot_dir = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
        geom_dir = os.path.join(rot_dir, 'data')
    else:
        assert os.path.exists(geom_dir), 'please provide correct folder'
    if geom_name is None: geom_name='sentinel2_tiles_world.geojson'

    tile_code = tile_code.upper()
    geom_path = os.path.join(geom_dir,geom_name)

    mgrs = geopandas.read_file(geom_path)

    toi = mgrs[mgrs['Name']==tile_code].total_bounds
    bbox = np.array([toi[0], toi[2], toi[1], toi[3]])
    return bbox

def get_geom_for_tile_code(tile_code, geom_dir=None, geom_name=None):
    """ get the geometry of a certain MGRS tile

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding
    geom_dir : string
        directory where geometric metadata about Sentinel-2 is situated
    geom_name : string
        filename of metadata, e.g.: 'sentinel2_tiles_world.geojson'

    Returns
    -------
    wkt : string
        well known text of the geometry, i.e.: 'POLYGON ((x y, x y, x y))'

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees

    The following acronyms are used:

    - CRS : coordinate reference system
    - MGRS : US military grid reference system
    - s2 : Sentinel-2
    - WKT : well known text
    """
    tile_code = _check_mgrs_code(tile_code)
    if geom_dir is None:
        rot_dir = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
        geom_dir = os.path.join(rot_dir, 'data')
    else:
        assert os.path.exists(geom_dir), 'please provide correct folder'
    if geom_name is None: geom_name='sentinel2_tiles_world.geojson'

    geom_path = os.path.join(geom_dir,geom_name)

    shp = ogr.Open(geom_path)
    lyr = shp.GetLayer()

    geom = None
    feat = lyr.GetNextFeature()
    while feat:
        if feat.GetField('Name') == tile_code:
            geom = feat.GetGeometryRef()
            time.sleep(.3)
            break
        feat = lyr.GetNextFeature()
    time.sleep(1) #todo: why is a delay needed?
    shp = lyr = None

    if geom is None:
        print('MGRS tile code does not seem to exist')
    return geom.ExportToWkt()

def get_tile_codes_from_geom(geom, geom_dir=None, geom_name=None):
    """
    Get the codes of the MGRS tiles intersecting a given geometry

    Parameters
    ----------
    geom : {shapely.geometry, string}
        geometry object or well known text, i.e.: 'POLYGON ((x y, x y, x y))'
    geom_dir : string
        directory where geometric metadata about Sentinel-2 is situated
    geom_name : string
        filename of metadata, e.g.: 'sentinel2_tiles_world.geojson'

    Returns
    -------
    tile_codes : tuple
        MGRS tile codes

    See Also
    --------
    .get_geom_for_tile_code, .get_bbox_from_tile_code
    """
    raise NotImplemented('Function to be added when working on s2 tiling grid')


def get_generic_s2_raster(tile_code, spac=10):
    """ create spatial metadata of a Sentinel-2, so no downloading is needed.

    Parameters
    ----------
    tile_code : string
        mgrs tile coding, which is also given in the filename ("TXXXXX")
    spac : integer, {10,20,60}, unit=m
        pixel spacing of the raster

    Returns
    -------
    geoTransform : tuple, size=(8,)
        georeference transform of an image.
    crs : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees

    The following acronyms are used:

    - CRS : coordinate reference system
    - MGRS : US military grid reference system
    - s2 : Sentinel-2
    """
    assert spac in (10,20,60,), 'please provide correct pixel resolution'

    tile_code = _check_mgrs_code(tile_code)
    geom = get_geom_for_tile_code(tile_code)
    geom = ogr.CreateGeometryFromWkt(geom)

    points = geom.Boundary().GetPointCount()
    bnd = geom.Boundary()
    lon, lat = np.zeros(points), np.zeros(points)
    for p in np.arange(points):
        lon[p],lat[p],_ = bnd.GetPoint(int(p))
    del geom, bnd
    # specify coordinate system
    spatRef = osr.SpatialReference()
    spatRef.ImportFromEPSG(4326) # WGS84

    utm_zone = get_utmzone_from_tile_code(tile_code)
    northernHemisphere = True if np.sign(np.mean(lat)) == 1 else False
    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS('WGS84')
    proj.SetUTM(utm_zone, northernHemisphere)

    coordTrans = osr.CoordinateTransformation(spatRef, proj)
    x,y = np.zeros(points), np.zeros(points)
    for p in np.arange(points):
        x[p],y[p],_ = coordTrans.TransformPoint(lat[p], lon[p])
    # bounding box in projected coordinate system
    x,y = np.round(x/spac)*spac, np.round(y/spac)*spac

    spac = float(spac)
    geoTransform = (np.min(x), +spac, 0.,
                    np.max(y), 0., -spac,
                    int(np.round(np.ptp(y)/spac)),
                    int(np.round(np.ptp(x)/spac)))
    crs = get_crs_from_mgrs_tile(tile_code)
    return geoTransform, crs

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
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn

def get_root_of_table(path, fname=None):
    if fname is None:
        full_name = path
    else:
        full_name = os.path.join(path, fname)
    if not '*' in full_name:
        assert os.path.exists(full_name), \
            ('please provide correct path and file name')
    dom = ElementTree.parse(glob.glob(full_name)[0])
    root = dom.getroot()
    return root

def get_s2_image_locations(fname,s2_df):
    """
    The Sentinel-2 imagery are placed within a folder structure, where one
    folder has an ever changing name. Fortunately this function finds the path
    from the metadata

    Parameters
    ----------
    fname : string
        path string to the Sentinel-2 folder
    s2_df : pandas.dataframe
        index of the bands of interest

    Returns
    -------
    im_paths : series, size=(k,)
        dataframe series with relative folder and file locations of the bands
    datastrip_id : string
        folder name of the metadata

    Example
    -------
    >>> import os
    >>> fpath = '/Users/Data/'
    >>> sname = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> fname = 'MTD_MSIL1C.xml'
    >>> full_path = os.path.join(fpath, sname, fname)
    >>> im_paths,datastrip_id = get_s2_image_locations(full_path)
    >>> im_paths
    ['GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B01',
     'GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B02']
    >>> datastrip_id
    'S2A_OPER_MSI_L1C_DS_VGS1_20200923T200821_S20200923T163313_N02.09'
    """
    assert os.path.exists(fname), ('metafile does not seem to be present')
    root = get_root_of_table(fname)

    for att in root.iter('Granule'):
        datastrip_full = att.get('datastripIdentifier')
        datastrip_id = '_'.join(datastrip_full.split('_')[4:-1])
    assert datastrip_id != None, ('metafile does not have required info')

    im_paths, band_id = [], []
    for im_loc in root.iter('IMAGE_FILE'):
        full_path = os.path.join(os.path.split(fname)[0],im_loc.text)
        im_paths.append(full_path)
        band_id.append(im_loc.text[-3:])

    band_path = pd.Series(data=im_paths, index=band_id, name="filepath")
    s2_df_new = pd.concat([s2_df, band_path], axis=1, join="inner")
    return s2_df_new, datastrip_id

def get_s2_granule_id(fname, s2_df):
    assert os.path.exists(fname), ('metafile does not seem to be present')
    root = get_root_of_table(fname)

    for im_loc in root.iter('IMAGE_FILE'):
        granule_id = im_loc.text.split(os.sep)[1]
    return granule_id

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
    assert os.path.exists(im_path), 'please specify a folder'
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

def get_utmzone_from_tile_code(tile_code):
    """

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding

    Returns
    -------
    utmzone : integer
        code used to denote the number of the projection column of UTM

    See Also
    --------
    .get_epsg_from_mgrs_tile, .get_crs_from_mgrs_tile

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees

    The following acronyms are used:

    - CRS : coordinate reference system
    - MGRS : US military grid reference system
    - UTM : universal transverse mercator
    - WGS : world geodetic system
    """
    tile_code = _check_mgrs_code(tile_code)
    return int(tile_code[:2])

def get_epsg_from_mgrs_tile(tile_code):
    """

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding

    Returns
    -------
    epsg : integer
        code used to denote a certain database entry

    See Also
    --------
    .get_utmzone_from_mgrs_tile, .get_crs_from_mgrs_tile

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees

    The following acronyms are used:

    - CRS : coordinate reference system
    - EPSG : european petroleum survey group (a coordinate refenence database)
    - MGRS : US military grid reference system
    - UTM : universal transverse mercator
    - WGS : world geodetic system
    """
    tile_code = _check_mgrs_code(tile_code)
    utm_num = get_utmzone_from_tile_code(tile_code)
    epsg_code = 32600 + utm_num

    # N to X are in the Northern hemisphere
    if tile_code[2] < 'N': epsg_code += 100
    return epsg_code

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

    See Also
    --------
    .get_utmzone_from_mgrs_tile, .get_utmzone_from_mgrs_tile

    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees
    """
    tile_code = _check_mgrs_code(tile_code)
    epsg_code = get_epsg_from_mgrs_tile(tile_code)

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(epsg_code)
    return crs
