# generic libraries
import glob
import os

from xml.etree import ElementTree

import numpy as np
import pandas as pd

# geospatial libaries
from osgeo import gdal, osr

# raster/image libraries
from PIL import Image, ImageDraw
from scipy.interpolate import griddata

from ..generic.handler_sentinel2 import get_array_from_xml
from ..generic.mapping_tools import map2pix

def list_central_wavelength_s2():
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
    >>> s2_df = list_central_wavelength_s2()
    >>> s2_df = s2_df[s2_df['name'].isin(boi)]
    >>> s2_df
             wavelength  bandwidth  resolution       name  bandid
    B02         492         66          10           blue       1
    B03         560         36          10          green       2
    B04         665         31          10            red       3
    B08         833        106          10  near infrared       7

    similarly you can also select by pixel resolution:

    >>> s2_df = list_central_wavelength_s2()
    >>> tw_df = s2_df[s2_df['resolution']==20]
    >>> tw_df.index
    Index(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], dtype='object')

    """
    wavelength = {"B01": 443, "B02": 492, "B03": 560, "B04": 665, \
                  "B05": 704, "B06": 741, "B07": 783, "B08": 833, "B8A": 865,\
                  "B09": 945, "B10":1374, "B11":1614, "B12":2202, \
                  }
    bandwidth = {"B01": 21, "B02": 66, "B03": 36, "B04": 31, \
                 "B05": 15, "B06": 15, "B07": 20, "B08":106, "B8A": 21,\
                 "B09": 20, "B10": 31, "B11": 91, "B12":175, \
                 }
    bandid = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, \
              "B05": 4, "B06": 5, "B07": 6, "B08": 7, "B8A": 8,\
              "B09": 9, "B10":10, "B11":11, "B12":12, \
                  }
    resolution = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, \
                  "B05": 20, "B06": 20, "B07": 20, "B08": 10, "B8A": 20,\
                  "B09": 60, "B10": 60, "B11": 20, "B12": 20, \
                  }
    name = {"B01": 'aerosol',       "B02" : 'blue', \
            "B03" : 'green',        "B04" : 'red', \
            "B05": 'red edge',      "B06" : 'red edge', \
            "B07" : 'red edge',     "B08" : 'near infrared', \
            "B8A" : 'narrow near infrared',\
            "B09": 'water vapour',  "B10": 'cirrus', \
            "B11": 'shortwave infrared', "B12": 'shortwave infrared', \
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

def s2_dn2toa(I):
    """convert the digital numbers of Sentinel-2 to top of atmosphere (TOA)

    Notes
    -----
    sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/
    level-1c/algorithm
    """
    I_toa = I*10000
    return I_toa

def read_band_s2(path, band='00'):
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
    if band!='00':
        fname = os.path.join(path, '*B'+band+'.jp2')
    else:
        fname = path
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def get_root_of_table(path, fname):
    full_name = os.path.join(path, fname)
    assert os.path.exists(path), ('please provide correct path and file name')
    dom = ElementTree.parse(glob.glob(full_name)[0])
    root = dom.getroot()
    return root

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
                        boi=list_central_wavelength_s2()):
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
    assert boi['resolution'].var()==0, \
        ('make sure all bands are the same resolution')
    root = get_root_of_table(path, fname)

    if det_stack.size==0:
        roi = boi['resolution'].mode()[0] # resolution of interest
        # image dimensions
        for meta in root.iter('Size'):
            res = float(meta.get('resolution'))
            if res == roi:
                mI = float(meta[0].text)
                nI = float(meta[1].text)
        Zn = np.zeros((mI,nI,len(boi)), dtype=np.float64),
        Az = np.zeros((mI,nI,len(boi)), dtype=np.float64)
        det_stack = np.ones((mI,nI,len(boi)))
    else:
        Zn = np.zeros_like(det_stack, dtype=np.float64)
        Az = np.zeros_like(det_stack, dtype=np.float64)
    # get coarse grids
    for grd in root.iter('Viewing_Incidence_Angles_Grids'):
        bid = float(grd.get('bandId'))
        if boi['bandid'].isin([bid]).any():
            # link to band in detector stack, by finding the integer index
            det_idx = np.flatnonzero(boi['bandid'].isin([bid]))[0]

            # link to detector id, within this band
            did = float(grd.get('detectorId'))

            det_arr = det_stack[:,:,det_idx]==did

            col_sep = float(grd[0][0].text) # in meters
            row_sep = float(grd[0][0].text) # in meters
            col_res = float(boi.loc[boi.index[det_idx],'resolution'])
            row_res = float(boi.loc[boi.index[det_idx],'resolution'])

            Zarray = get_array_from_xml(grd[0][2])
            Aarray = get_array_from_xml(grd[1][2])

            I_grd,J_grd = np.mgrid[0:Zarray.shape[0],0:Zarray.shape[1]]
            IN = ~np.isnan(Zarray)
            I_grd,J_grd = I_grd[IN].astype('float64'), J_grd[IN].astype('float64')
            Z_grd,A_grd = Zarray[IN], Aarray[IN]
            I_grd *= (col_sep/col_res)
            J_grd *= (row_sep/row_res)

            det_i,det_j = np.where(det_arr)

            # do linear regression, so extrapolation is possible
            A = np.vstack((I_grd,J_grd, np.ones(I_grd.size) )).T
            x_z = np.linalg.lstsq(A, Z_grd, rcond=None)[0]
            det_Z = x_z[0]*det_i.astype('float64') + \
                x_z[1]*det_j.astype('float64') + x_z[2]
            x_a = np.linalg.lstsq(A, A_grd, rcond=None)[0]
            det_A = x_a[0]*det_i.astype('float64') + \
                x_a[1]*det_j.astype('float64') + x_a[2]

            det_k = np.repeat(det_idx,len(det_j))
            Zn[det_i,det_j,det_k] = det_Z
            Az[det_i,det_j,det_k] = det_A
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

def read_detector_mask(path_meta, msk_dim, boi, geoTransform):
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
    msk_dim : tuple
        dimensions of the stack.
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
    >>> msk_dim = (10980, 10980, len(boi))
    >>> s2_df = list_central_wavelength_s2()
    >>> boi_df = s2_df[s2_df['name'].isin(boi)]
    >>> geoTransform = (600000.0, 10.0, 0.0, 10000000.0, 0.0, -10.0)
    >>>
    >>> det_stack = read_detector_mask(path_meta, msk_dim, boi_df, geoTransform)
    """

    det_stack = np.zeros(msk_dim, dtype='int8')
    for i in range(len(boi)):
        im_id = boi.index[i] # 'B01' | 'B8A'
        if type(im_id) is int:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_B'+ \
                                  f'{im_id:02.0f}' + '.gml')
        else:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_'+ im_id +'.gml')
        dom = ElementTree.parse(glob.glob(f_meta)[0])
        root = dom.getroot()

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

def read_cloud_mask(path_meta, msk_dim, geoTransform): # todo gml:boundedBy gives image envelope
    msk_clouds = np.zeros(msk_dim, dtype='int8')

    f_meta = os.path.join(path_meta, 'MSK_CLOUDS_B00.gml')
    dom = ElementTree.parse(glob.glob(f_meta)[0])
    root = dom.getroot()

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
        msk = Image.new("L", [msk_dim[0], msk_dim[1]], 0)
        ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                    outline=1, fill=1)
        msk = np.array(msk)
        msk_clouds = np.maximum(msk_clouds, msk)
    return msk_clouds

def read_detector_time(path, fname='MTD_DS.xml'):
    det_time = np.zeros((13, 12), dtype='datetime64[ns]')
    det_name = [None] * 13
    det_meta = np.zeros((13, 4), dtype='float')

    root = get_root_of_table(path, fname)
    # image dimensions
    for meta in root.iter('Band_Time_Stamp'):
        bnd = int(meta.get('bandId')) # 0..12
        # <Spectral_Information bandId="8" physicalBand="B8A">
        for stamps in meta:
            det = int(stamps.get('detectorId'))
            det_time[bnd,det-1] = np.datetime64(stamps[1].text, 'ns')

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
    return det_time, det_name, det_meta
