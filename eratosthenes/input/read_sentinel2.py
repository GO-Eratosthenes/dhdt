# generic libraries
import glob
import os
import warnings

from xml.etree import ElementTree

import numpy as np
import pandas as pd

# geospatial libaries
from osgeo import gdal, osr

# raster/image libraries
from PIL import Image, ImageDraw
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata, interp2d

from ..generic.handler_sentinel2 import get_array_from_xml
from ..generic.mapping_tools import map2pix, ecef2map, get_bbox
from ..generic.mapping_io import read_geo_image, read_geo_info

def list_platform_metadata_s2a():
    s2a_dict = {
        'COSPAR': '2015-028A',
        'NORAD': 40697,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch': '2015-06-23',
        'orbit': 'sso'}
    return s2a_dict

def list_platform_metadata_s2b():
    s2b_dict = {
        'COSPAR': '2017-013A',
        'NORAD': 42063,
        'instruments': {'MSI'},
        'constellation': 'sentinel',
        'launch': '2017-03-07',
        'orbit': 'sso'}
    return s2b_dict

def list_central_wavelength_msi():
    """ create dataframe with metadata about Sentinel-2

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

    Example
    -------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue', 'near infrared']
    >>> s2_df = list_central_wavelength_msi()
    >>> s2_df = s2_df[s2_df['common_name'].isin(boi)]
    >>> s2_df
             wavelength  bandwidth  resolution       name  bandid
    B02         492         66          10           blue       1
    B03         560         36          10          green       2
    B04         665         31          10            red       3
    B08         833        106          10  near infrared       7

    similarly you can also select by pixel resolution:

    >>> s2_df = list_central_wavelength_s2()
    >>> tw_df = s2_df[s2_df['gsd']==20]
    >>> tw_df.index
    Index(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], dtype='object')

    """
    center_wavelength = {"B01": 443, "B02": 492, "B03": 560, "B04": 665,
                  "B05": 704, "B06": 741, "B07": 783, "B08": 833, "B8A": 865,
                  "B09": 945, "B10":1374, "B11":1614, "B12":2202,
                  }
    full_width_half_max = {"B01": 21, "B02": 66, "B03": 36, "B04": 31,
                 "B05": 15, "B06": 15, "B07": 20, "B08":106, "B8A": 21,
                 "B09": 20, "B10": 31, "B11": 91, "B12":175,
                 }
    bandid = {"B01": 0, "B02": 1, "B03": 2, "B04": 3,
              "B05": 4, "B06": 5, "B07": 6, "B08": 7, "B8A": 8,
              "B09": 9, "B10":10, "B11":11, "B12":12,
                  }
    gsd = {"B01": 60, "B02": 10, "B03": 10, "B04": 10,
           "B05": 20, "B06": 20, "B07": 20, "B08": 10, "B8A": 20,
           "B09": 60, "B10": 60, "B11": 20, "B12": 20,
           }
    # along_track_view_angle =
    field_of_view = {"B01": 20.6, "B02": 20.6, "B03": 20.6, "B04": 20.6,
                     "B05": 20.6, "B06": 20.6, "B07": 20.6, "B08": 20.6,
                     "B8A": 20.6, "B09": 20.6, "B10": 20.6, "B11": 20.6,
                     "B12": 20}
    common_name = {"B01": 'coastal',       "B02" : 'blue',
                   "B03" : 'green',        "B04" : 'red',
                   "B05" : 'rededge',      "B06" : 'rededge',
                   "B07" : 'rededge',      "B08" : 'nir',
                   "B8A" : 'nir08',        "B09" : 'nir09',
                   "B10": 'cirrus',        "B11" : 'swir16',
                   "B12": 'swir22'}
    solar_illumination = {"B01":1913, "B02":1942, "B03":1823, "B04":1513,
                  "B05":1426, "B06":1288, "B07":1163, "B08":1036, "B8A":955,
                  "B09": 813, "B10": 367, "B11": 246, "B12": 85,
                  } # these numbers are also given in the meta-data
    d = {
         "center_wavelength": pd.Series(center_wavelength),
         "full_width_half_max": pd.Series(full_width_half_max),
         "gsd": pd.Series(gsd),
         "common_name": pd.Series(common_name),
         "bandid": pd.Series(bandid),
         "field_of_view" : pd.Series(field_of_view),
         "solar_illumination": pd.Series(solar_illumination)
         }
    df = pd.DataFrame(d)
    return df

def s2_dn2toa(I):
    """convert the digital numbers of Sentinel-2 to top of atmosphere (TOA)

    Notes
    -----
    sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/
    level-1c/algorithm
    """
    I_toa = I.astype('float')/10000
    return I_toa

def read_band_s2(path, band=None):
    """
    This function takes as input the Sentinel-2 band name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    path : string
        path of the folder, or full path with filename as well
    band : string, optional
        Sentinel-2 band name, for example '04', '8A'.

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
    list_central_wavelength_s2

    Example
    -------
    >>> path = '/GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/'
    >>> band = '02'
    >>> _,spatialRef,geoTransform,targetprj = read_band_s2(path, band)

    >>> spatialRef
    'PROJCS["WGS 84 / UTM zone 15S",GEOGCS["WGS 84",DATUM["WGS_1984",
    SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
    AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-93],
    PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],
    PARAMETER["false_northing",10000000],UNIT["metre",1,
    AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32715"]]'
    >>> geoTransform
    (600000.0, 10.0, 0.0, 10000000.0, 0.0, -10.0)
    >>> targetprj
    <osgeo.osr.SpatialReference; proxy of <Swig Object of type
    'OSRSpatialReferenceShadow *' at 0x7f9a63ffe450> >
    """
    if band!=None:
        if len(band)==3: #when band : 'B0X'
            fname = os.path.join(path, '*' + band + '.jp2')
        else: # when band: '0X'
            fname = os.path.join(path, '*' + band + '.jp2')
    else:
        fname = path
    assert len(glob.glob(fname))!=0, ('file does not seem to be present')

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0])
    return data, spatialRef, geoTransform, targetprj

def read_stack_s2(path, s2_df):

    assert 'filepath' in s2_df, ('please first run "get_S2_image_locations"'+
                                ' to find the proper file locations')

    roi = np.min(s2_df['gsd'].array) # resolution of interest
    im_scaling = s2_df['gsd'] / roi
    # start with the highest resolution
    for val, idx in enumerate(s2_df.sort_values('gsd').index):
        full_path = s2_df['filepath'][idx] + '.jp2'
        if val==0:
            im_stack, spatialRef, geoTransform, targetprj = read_band_s2(full_path)
        else: # stack others bands
            if im_stack.ndim==2:
                im_stack = im_stack[:,:,np.newaxis]
            band,_,_,_ = read_band_s2(full_path)
            if im_scaling[idx]!=1: # resize image to higher res.
                band = resize(band, (im_stack.shape[0], im_stack.shape[1]), order=3)
            band = band[:,:,np.newaxis]
            im_stack = np.concatenate((im_stack, band), axis=2)

    # in the meta data files there is no mention of its size, hence include
    # it to the geoTransform
    geoTransform = geoTransform + (im_stack.shape[0], im_stack.shape[1])
    return im_stack, spatialRef, geoTransform, targetprj

def get_root_of_table(path, fname):
    full_name = os.path.join(path, fname)
    if not '*' in full_name:
        assert os.path.exists(full_name), \
            ('please provide correct path and file name')
    dom = ElementTree.parse(glob.glob(full_name)[0])
    root = dom.getroot()
    return root

def read_geotransform_s2(path, fname='MTD_TL.xml', resolution=10):
    """

    Parameters
    ----------
    path : string
        location where the meta data is situated
    fname : string
        file name of the meta-data file
    resolution : {float,integer}, unit=meters, default=10
        resolution of the grid

    Returns
    -------
    geoTransform : tuple, size=(1,6)
        affine transformation coefficients
    """
    root = get_root_of_table(path, fname)

    # image dimensions
    for meta in root.iter('Geoposition'):
        res = float(meta.get('resolution'))
        if res == resolution:
            ul_X,ul_Y= float(meta[0].text), float(meta[1].text)
            d_X, d_Y = float(meta[2].text), float(meta[3].text)
    geoTransform = (ul_X, d_X, 0., ul_Y, 0., d_Y)
    return geoTransform

def get_local_bbox_in_s2_tile(fname_1, s2dir):
    _,geoTransform, _, rows, cols, _ = read_geo_info(fname_1)
    bbox_xy = get_bbox(geoTransform, rows, cols)

    s2Transform = read_geotransform_s2(s2dir)
    bbox_i, bbox_j = map2pix(s2Transform, bbox_xy[0:2], bbox_xy[2::])
    bbox_ij = np.concatenate((np.flip(bbox_i), bbox_j)).astype(int)
    return bbox_ij

def read_sun_angles_s2(path, fname='MTD_TL.xml'):
    """ This function reads the xml-file of the Sentinel-2 scene and extracts
    an array with sun angles, as these vary along the scene.

    Parameters
    ----------
    path : string
        path where xml-file of Sentinel-2 is situated

    Returns
    -------
    Zn : np.array, size=(m,n), dtype=float
        array of the solar zenith angles, in degrees.
    Az : np.array, size=(m,n), dtype=float
        array of the solar azimuth angles, in degrees.

    Notes
    -----
    The angle(s) are declared in the following coordinate frame:

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
          +----                 +------

    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |
    """
    root = get_root_of_table(path, fname)

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)), 0)
    znSpac = np.divide(znSpac, 10)  # transform from meters to pixels

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj, znSpac

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)), 0)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj, azSpac

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az

def read_view_angles_s2(path, fname='MTD_TL.xml', det_stack=np.array([]),
                        boi=list_central_wavelength_msi()):
    """ This function reads the xml-file of the Sentinel-2 scene and extracts
    an array with viewing angles of the MSI instrument.

    Parameters
    ----------
    path : string
        path where xml-file of Sentinel-2 is situated
    fname : string
        the name of the metadata file, sometimes this is changed
    band_id : integer, default=4
        each band has a somewhat minute but different view angle

    Returns
    -------
    Zn : np.array, size=(m,n), dtype=float
        array of the solar zenith angles, in degrees.
    Az : np.array, size=(m,n), dtype=float
        array of the solar azimuth angles, in degrees.

    See Also
    --------
    list_central_wavelength_s2

    Notes
    -----
    The azimuth angle is declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

    The angles related to the satellite are as follows:

        .. code-block:: text

               #*#                   #*# satellite
          ^    /                ^    /|
          |   /                 |   / | nadir
          |-- zenith angle      |  /  v
          | /                   | /|
          |/                    |/ | elevation angle
          +----- surface        +------

    """
    assert boi['gsd'].var()==0, \
        ('make sure all bands are the same resolution')
    root = get_root_of_table(path, fname)

    det_list = np.unique(det_stack)
    bnd_list = np.asarray(boi['bandid'])
    Zn_grd, Az_grd = None, None
    # get coarse grids
    for grd in root.iter('Viewing_Incidence_Angles_Grids'):
        bid = float(grd.get('bandId'))
        if not boi['bandid'].isin([bid]).any():
            continue
        grd_idx_3 = np.where(bnd_list==bid)[0][0]

        # link to band in detector stack, by finding the integer index
        det_idx = np.flatnonzero(boi['bandid'].isin([bid]))[0]

        # link to detector id, within this band
        did = float(grd.get('detectorId'))
        grd_idx_4 = np.where(det_list==did)[0][0]

        col_sep = float(grd[0][0].text) # in meters
        row_sep = float(grd[0][0].text) # in meters
        col_res = float(boi.loc[boi.index[det_idx],'gsd'])
        row_res = float(boi.loc[boi.index[det_idx],'gsd'])

        Zarray = get_array_from_xml(grd[0][2])
        if Zn_grd is None:
            Zn_grd = np.zeros((Zarray.shape[0], Zarray.shape[1],
                              len(bnd_list), len(det_list)), dtype=np.float64)
        Zn_grd[:,:,grd_idx_3,grd_idx_4] = Zarray
        Aarray = get_array_from_xml(grd[1][2])
        if Az_grd is None:
            Az_grd = np.zeros((Aarray.shape[0], Aarray.shape[1],
                              len(bnd_list), len(det_list)), dtype=np.float64)
        Az_grd[:,:,grd_idx_3,grd_idx_4] = Aarray

    if det_stack.size == 0:
        roi = boi['gsd'].mode()[0]  # resolution of interest
        # image dimensions
        for meta in root.iter('Size'):
            res = float(meta.get('resolution'))
            if res == roi:
                mI, nI = float(meta[0].text), float(meta[1].text)
    else:
        mI, nI = det_stack.shape[0], det_stack.shape[1]

    # create grid coordinate frame
    I_grd, J_grd = np.mgrid[0:Zn_grd.shape[0], 0:Zn_grd.shape[1]]
    I_grd, J_grd = I_grd.astype('float64'), J_grd.astype('float64')
    I_grd *= (col_sep / col_res)
    J_grd *= (row_sep / row_res)

    Zn, Az = None, None
    det_grp = np.array([[1,3,5],[2,4,6],[7,9,11],[8,10,12]])
    for idx, bnd in enumerate(bnd_list):
        Zn_bnd, Az_bnd = np.zeros((mI, nI)), np.zeros((mI, nI))
        for i in range(det_grp.shape[0]):
            Zn_samp = np.squeeze(Zn_grd[:,:,idx,np.isin(det_list, det_grp[i,:])])
            if Zn_samp.ndim==3:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            r'All-NaN (slice|axis) encountered')
                    Zn_samp = np.nanmax(Zn_samp, axis=2)

            Az_samp = np.squeeze(Az_grd[:,:,idx,np.isin(det_list, det_grp[i,:])])
            if Az_samp.ndim==3:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            r'All-NaN (slice|axis) encountered')
                    Az_samp = np.nanmax(Az_samp, axis=2)

            IN = np.invert(np.isnan(Zn_samp))
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(
                np.stack((I_grd[IN], J_grd[IN])).T)

            det_arr = np.isin(det_stack[:,:,idx], det_grp[i, :])
            det_i, det_j = np.where(det_arr)
            det_i,det_j = det_i.astype('float64'), det_j.astype('float64')

            _,indices = nbrs.kneighbors(np.stack((det_i,det_j)).T)
            Zn_bnd[det_arr] = np.squeeze(Zn_samp[IN][indices])
            Az_bnd[det_arr] = np.squeeze(Az_samp[IN][indices])
        if Zn is None:
            Zn, Az = Zn_bnd.copy(), Az_bnd.copy()
        else:
            Zn, Az = np.dstack((Zn, Zn_bnd)), np.dstack((Az, Az_bnd))
    return Zn, Az

def read_mean_sun_angles_s2(path, fname='MTD_TL.xml'):
    """ Read the xml-file of the Sentinel-2 scene and extract the mean sun angles.

    Parameters
    ----------
    path : string
        path where xml-file of Sentinel-2 is situated

    Returns
    -------
    Zn : float
        Mean solar zentih angle of the scene, in degrees.
    Az : float
        Mean solar azimuth angle of the scene, in degrees

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
          +----                 +------
    """
    root = get_root_of_table(path, fname)

    Zn = float(root[1][1][1][0].text)
    Az = float(root[1][1][1][1].text)
    return Zn, Az

def read_sensing_time_s2(path, fname='MTD_TL.xml'):
    """

    Parameters
    ----------
    path : string
        path where the meta-data is situated
    fname : string
        file name of the metadata.

    Returns
    -------
    rec_time :
        asd

    Example
    -------
    demonstrate the code when in a Scihub data structure

    >>> import os
    >>> fpath = '/Users/Data/'
    >>> sname = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> fname = 'MTD_MSIL1C.xml'
    >>> s2_path = os.path.join(fpath, sname, fname)
    >>> s2_df,_ = get_S2_image_locations(fname, s2_df)
    >>> s2_df['filepath']
    B02 'GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T115MXV...'
    B03 'GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV...'
    >>> full_path = '/'.join(s2_df['filepath'][0].split('/')[:-2])
    'GRANULE/L1C_T15MXV_A027450_20200923T163313'
    >>> rec_time = read_sensing_time_s2(path, fname='MTD_TL.xml')
    >>> rec_time

    """
    root = get_root_of_table(path, fname)
    for att in root.iter('Sensing_Time'.upper()):
        rec_time = np.datetime64(att.text, 'ns')
    return rec_time

def read_detector_mask(path_meta, boi, geoTransform):
    """ create array of with detector identification

    Sentinel-2 records in a pushbroom fasion, collecting reflectance with a
    ground resolution of more than 270 kilometers. This data is stacked in the
    flight direction, and then cut into granules. However, the sensorstrip
    inside the Multi Spectral Imager (MSI) is composed of several CCD arrays.

    This function collects the geometry of these sensor arrays from the meta-
    data. Since this is stored in a gml-file.

    Parameters
    ----------
    path_meta : string
        path where the meta-data is situated.
    boi : DataFrame
        list with bands of interest
    geoTransform : tuple
        affine transformation coefficients

    Returns
    -------
    det_stack : np.array, size=(msk_dim[0],msk_dim[1],len(boi)), dtype=int8
        array where each pixel has the ID of the detector, of a specific band

    See Also
    --------
    read_view_angles_s2

    Example
    -------
    >>> path_meta = '/GRANULE/L1C_T15MXV_A027450_20200923T163313/QI_DATA'
    >>> boi = ['red', 'green', 'blue', 'near infrared']
    >>> s2_df = list_central_wavelength_s2()
    >>> boi_df = s2_df[s2_df['name'].isin(boi)]
    >>> geoTransform = (600000.0, 10.0, 0.0, 10000000.0, 0.0, -10.0)
    >>>
    >>> det_stack = read_detector_mask(path_meta, boi_df, geoTransform)
    """

    if len(geoTransform)>6: # also image size is given
        msk_dim = (geoTransform[-2], geoTransform[-1], len(boi))
        det_stack = np.zeros(msk_dim, dtype='int8')  # create stack

    for i in range(len(boi)):
        im_id = boi.index[i] # 'B01' | 'B8A'
        if type(im_id) is int:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_B'+ \
                                  f'{im_id:02.0f}' + '.gml')
        else:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_'+ im_id +'.gml')
        dom = ElementTree.parse(glob.glob(f_meta)[0])
        root = dom.getroot()

        if 'msk_dim' not in locals():
            # find dimensions of array through its map extent in metadata
            lower_corner = np.fromstring(root[1][0][0].text, sep=' ')
            upper_corner = np.fromstring(root[1][0][1].text, sep=' ')
            lower_x, lower_y = map2pix(geoTransform, lower_corner[0], lower_corner[1])
            upper_x, upper_y = map2pix(geoTransform, upper_corner[0], upper_corner[1])

            msk_dim = (np.maximum(lower_x,upper_x).astype(int),
                       np.maximum(lower_y,upper_y).astype(int),
                       len(boi))
            det_stack = np.zeros(msk_dim, dtype='int8') # create stack

        mask_members = root[2]
        for k in range(len(mask_members)):
            # get detector number from meta-data
            det_id = mask_members[k].attrib
            det_id = list(det_id.items())[0][1].split('-')[2]
            det_num = int(det_id)

            # get footprint
            pos_dim = mask_members[k][1][0][0][0][0].attrib
            pos_dim = int(list(pos_dim.items())[0][1])
            pos_list = mask_members[k][1][0][0][0][0].text
            pos_row = [float(s) for s in pos_list.split(' ')]
            pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))

            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
            ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
            # make mask
            msk = Image.new("L", [np.size(det_stack,0), np.size(det_stack,1)], 0)
            ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                        outline=det_num, fill=det_num)
            msk = np.array(msk)
            det_stack[:,:,i] = np.maximum(det_stack[:,:,i], msk)
    return det_stack

def read_cloud_mask(path_meta, geoTransform):

    f_meta = os.path.join(path_meta, 'MSK_CLOUDS_B00.gml')
    dom = ElementTree.parse(glob.glob(f_meta)[0])
    root = dom.getroot()

#    # find dimensions of array through its map extent in metadata
#    lower_corner = np.fromstring(root[1][0][0].text, sep=' ')
#    upper_corner = np.fromstring(root[1][0][1].text, sep=' ')
#    lower_x, lower_y = map2pix(geoTransform, lower_corner[0], lower_corner[1])
#    upper_x, upper_y = map2pix(geoTransform, upper_corner[0], upper_corner[1])
#    msk_dim = (np.maximum(lower_y, upper_y).astype(int),
#               np.maximum(lower_x, upper_x).astype(int))

    if len(geoTransform)>6: # also image size is given
        msk_dim = (geoTransform[-2], geoTransform[-1])
        msk_clouds = np.zeros(msk_dim, dtype='int8')
    else:
        msk_dim = (10980,10980)
        msk_clouds = np.zeros(msk_dim, dtype='int8')  # create stack

    if len(root)>2: # look into meta-data for cloud polygons
        mask_members = root[2]
        for k in range(len(mask_members)):
            # get footprint
            pos_dim = mask_members[k][1][0][0][0][0].attrib
            pos_dim = int(list(pos_dim.items())[0][1])
            pos_list = mask_members[k][1][0][0][0][0].text
            pos_row = [float(s) for s in pos_list.split(' ')]
            pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))

            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
            ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
            # make mask
            msk = Image.new("L", [msk_dim[1], msk_dim[0]], 0) # in [width, height] format
            ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])),
                                        outline=1, fill=1)
            msk = np.array(msk)
            msk_clouds = np.maximum(msk_clouds, msk)
    return msk_clouds

def read_detector_time_s2(path, fname='MTD_DS.xml'): #todo: make description of function with example

# example : line_period / np.timedelta64(1, 's')
    det_time = np.zeros((13, 12), dtype='datetime64[ns]')
    det_name = [None] * 13
    det_meta = np.zeros((13, 4), dtype='float')

    root = get_root_of_table(path, fname)

    # image dimensions
    for meta in root.iter('Band_Time_Stamp'):
        bnd = int(meta.get('bandId')) # 0...12
        # <Spectral_Information bandId="8" physicalBand="B8A">
        for stamps in meta:
            det = int(stamps.get('detectorId')) # 1...12
            det_time[bnd,det-1] = np.datetime64(stamps[1].text, 'ns')

    # get line_period : not the ground sampling distance, but the temporal
    # sampling distance
    for elem in root.iter('LINE_PERIOD'):
       if elem.get('unit')=='ms': # datetime can only do integers...
           line_period = np.timedelta64(int(float(elem.text)*1e6), 'ns')

    for meta in root.iter('Spectral_Information'):
        bnd = int(meta.get('bandId'))
        bnd_name = meta.get('physicalBand')
        # at a zero if needed, this seems the correct naming convention
        if len(bnd_name)==2:
            bnd_name = bnd_name[0]+'0'+bnd_name[-1]
        det_name[bnd] = bnd_name
        det_meta[bnd,0] = float(meta[0].text) # resolution [m]
        det_meta[bnd,1] = float(meta[1][0].text) # min
        det_meta[bnd,2] = float(meta[1][1].text) # max
        det_meta[bnd,3] = float(meta[1][2].text) # mean
    return det_time, det_name, det_meta, line_period

def get_flight_orientation_s2(path, fname='MTD_DS.xml'):
    """ get the flight path and orientations of the Sentinel-2 satellite during
    acquisition.

    Parameters
    ----------
    path : string
        directory where the meta-data is located
    fname : string
        filename of the meta-data file

    Returns
    -------
    sat_time : np.array, size=(m,1), unit=nanosec
        satellite timestamps
    sat_angles : np.array, size=(m,3)
        satellite orientation angles, given in ECEF

    See Also
    --------
    get_flight_path_s2
    """
    root = get_root_of_table(path, fname)

    for att in root.iter('Corrected_Attitudes'):
        sat_time = np.empty((len(att)), dtype='datetime64[ns]')
        sat_angles = np.zeros((len(att),4))
        counter = 0
        for idx, val in enumerate(att):
            # 'QUATERNION_VALUES'
            qang =  np.fromstring(val[0].text, dtype=float, sep=' ')
            # 'GPS_TIME'
            gps_tim = np.datetime64(val[2].text, 'ns')
            sat_time[idx] = gps_tim
            sat_angles[idx,:] = qang
    return sat_time, sat_angles

def get_flight_path_s2(path, fname='MTD_DS.xml'):
    """

    Parameters
    ----------
    path : string
        location of the metadata file
    fname : string
        name of the xml-file that has the metadata

    Returns
    -------
    sat_time : np.array, size=(m,1), dtype=np.datetime64, unit=
        time stamp of the satellite positions
    sat_xyz : np.array, size=(m,3), dtype=float, unit=meter
        3D coordinates of the satellite within an Earth centered Earth fixed
        (ECEF) frame.
    sat_err : np.array, size=(m,3), dtype=float, unit=meter
        error estimate of the 3D coordinates given by "sat_xyz"
    sat_uvw : np.array, size=(m,3), dtype=float, unit=meter
        3D velocity vectors of the satellite within an Earth centered Earth
        fixed (ECEF) frame.

    Examples
    --------
    Following the file and metadata structure of scihub:

    >>> import os
    >>> import numpy as np

    >>> S2_dir = '/data-dump/examples/
    >>> S2_name = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> fname = os.path.join(S2_dir, S2_name, 'MTD_MSIL1C.xml')
    >>> s2_df = list_central_wavelength_s2()

    >>> s2_df, datastrip_id = get_S2_image_locations(fname, s2_df)
    >>> path_det = os.path.join(S2_dir, S2_name, 'DATASTRIP', datastrip_id[17:-7])

    >>> sat_tim, sat_xyz, sat_err, sat_uvw = get_flight_path_s2(path_det)

    See Also
    --------
    get_flight_orientation_s2
    """
    root = get_root_of_table(path, fname)

    for att in root.iter('GPS_Points_List'):
        sat_time = np.empty((len(att)), dtype='datetime64[ns]')
        sat_xyz = np.zeros((len(att), 3), dtype=float)
        sat_err = np.zeros((len(att), 3), dtype=float)
        sat_uvw = np.zeros((len(att), 3), dtype=float)
        counter = 0
        for idx, point in enumerate(att):
            # 'POSITION_VALUES' : tag
            xyz = np.fromstring(point[0].text, dtype=float, sep=' ')
            if point[0].attrib['unit'] == 'mm': xyz *= 1E-3 # convert to meters
            # 'POSITION_ERRORS'
            err = np.fromstring(point[1].text, dtype=float, sep=' ')
            if point[1].attrib['unit'] == 'mm': err *= 1E-3 # convert to meters
            # 'VELOCITY_VALUES'
            uvw = np.fromstring(point[2].text, dtype=float, sep=' ')
            if point[2].attrib['unit'] == 'mm': uvw *= 1E-3 # convert to meters

            # 'GPS_TIME'
            gps_tim = np.datetime64(point[4].text, 'ns')

            # fill in the arrays
            sat_time[idx] = gps_tim
            sat_xyz[idx, :], sat_err[idx, :], sat_uvw[idx, :] = xyz, err, uvw
    return sat_time, sat_xyz, sat_err, sat_uvw

# use the detector start and finish to make a selection for the flight line
def get_flight_heading_s2(path, spatialRef, rec_tim, fname='MTD_DS.xml'): #todo: make independent of rec_time, add documentation
    """ get the direction/argument/heading of the Sentinel-2 acquisition

    Parameters
    ----------
    path : string
        directory where the meta-data is located
    spatialRef : osgeo.osr.SpatialReference
        projection system used
    rec_tim : {integer,float}
        time stamp of interest
    fname : string
        filename of the meta-data file

    Returns
    -------
    az : np.array, size=(m,1), float
        array with argument values, based upon the map projection given
    """

    sat_tim,sat_xyz,_,_ = get_flight_path_s2(path, fname=fname)
    sat_xy = ecef2map(sat_xyz, spatialRef)

    dif_tim = sat_tim - rec_tim
    idx = np.argmin(np.abs(dif_tim))
    dif_xy = sat_xy[idx + 1] - sat_xy[idx]

    az = np.arctan2(dif_xy[0], dif_xy[1]) * 180 / np.pi
    return az

def get_integration_and_sampling_time_s2(path, fname='MTD_DS.xml'):

    root = get_root_of_table(path, fname)
    integration_tim = np.zeros(13, dtype='timedelta64[ns]')
    # get line sampling interval, in nanoseconds
    for att in root.iter('Line_Period'.upper()):
        line_tim = np.timedelta64(int(float(att.text)*1e6), 'ns')

    # get line sampling interval, in nanoseconds
    for att in root.iter('Spectral_Band_Information'):
        band_idx = int(att.attrib['bandId'])
        for var in att:
            if var.tag == 'Integration_Time'.upper():
                int_tim = np.timedelta64(int(float(var.text)*1e6), 'ns')
        integration_tim[band_idx] = int_tim

    return line_tim, integration_tim