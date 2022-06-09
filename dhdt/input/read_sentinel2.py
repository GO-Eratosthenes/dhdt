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
from scipy.ndimage import label
from scipy.signal import convolve2d

from ..generic.handler_sentinel2 import get_array_from_xml
from ..generic.mapping_tools import map2pix, ecef2map, ecef2llh, get_bbox
from ..generic.mapping_io import read_geo_image, read_geo_info

def list_central_wavelength_msi():
    """ create dataframe with metadata about Sentinel-2

    Returns
    -------
    df : pandas.dataframe
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2, having the following collumns:

            * wavelength, unit=µm : central wavelength of the band
            * bandwidth, unit=µm : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution, unit=m : spatial resolution of a pixel
            * along_pixel_size, unit=µm : physical size of the sensor
            * across_pixel_size, unit=µm : size of the photosensative sensor
            * focal_length, unit=m : lens characteristic of the telescope
            * field_of_view, unit=degrees : angle of swath of instrument
            * crossdetector_parallax, unit=degress : in along-track direction
            * name : general name of the band, if applicable
            * solar_illumination, unit=W m-2 µm-1 :

    Notes
    -----
    The Multi Spectral instrument (MSI) has a Tri Mirror Anastigmat (TMA)
    telescope, which is opened at F/4 and has a focal length of 60 centimeter.

    The crossdetector parallex is given here as well, see [1]. While the
    crossband parallax is 0.018 for VNIR and 0.010 for the SWIR detector,
    respectively. The dimensions of the Silicon CMOS detector are given in [2].
    While the SWIR detector is based on a MCT sensor.

    The following acronyms are used:

        - MSI : multi-spectral instrument
        - MCT : mercury cadmium telluride, HgCdTe
        - TMA : trim mirror anastigmat
        - VNIR : very near infra-red
        - SWIR : short wave infra-red
        - CMOS : silicon complementary metal oxide semiconductor, Si

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

    References
    -------
    .. [1] Languille et al. "Sentinel-2 geometric image quality commissioning:
       first results" Proceedings of the SPIE, 2015.
    .. [2] Martin-Gonthier et al. "CMOS detectors for space applications: From
       R&D to operational program with large volume foundry", Proceedings of the
       SPIE conference on sensors, systems, and next generation satellites XIV,
       2010.
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

    along_pixel_size = {"B01": 45., "B02": 7.5, "B03": 7.5, "B04": 7.5,
                        "B05": 15., "B06": 15., "B07": 15., "B08": 7.5,
                        "B8A": 15., "B09": 45., "B10": 15., "B11": 15.,
                        "B12": 15.,}

    across_pixel_size = {"B01": 15., "B02": 7.5, "B03": 7.5, "B04": 7.5,
                        "B05": 15., "B06": 15., "B07": 15., "B08": 7.5,
                        "B8A": 15., "B09": 15., "B10": 15., "B11": 15.,
                        "B12": 15.,}

    focal_length = {"B01": .598, "B02": .598, "B03": .598, "B04": .598,
                    "B05": .598, "B06": .598, "B07": .598, "B08": .598,
                    "B8A": .598, "B09": .598, "B10": .598, "B11": .598,
                    "B12": .598}
    # along_track_view_angle =
    field_of_view = {"B01": 20.6, "B02": 20.6, "B03": 20.6, "B04": 20.6,
                     "B05": 20.6, "B06": 20.6, "B07": 20.6, "B08": 20.6,
                     "B8A": 20.6, "B09": 20.6, "B10": 20.6, "B11": 20.6,
                     "B12": 20.6}
    # given as base-to-height
    crossdetector_parallax = {"B01": 0.055, "B02": 0.022, "B03": 0.030,
                              "B04": 0.034, "B05": 0.038, "B06": 0.042,
                              "B07": 0.046, "B08": 0.026, "B8A": 0.051,
                              "B09": 0.059, "B10": 0.030, "B11": 0.040,
                              "B12": 0.050,
                          }
    # convert to angles in degrees
    crossdetector_parallax = \
        {k: 2*np.tan(v/2) for k,v in crossdetector_parallax.items()}

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
         "across_pixel_size": pd.Series(across_pixel_size),
         "along_pixel_size": pd.Series(along_pixel_size),
         "focal_length": pd.Series(focal_length),
         "common_name": pd.Series(common_name),
         "bandid": pd.Series(bandid),
         "field_of_view" : pd.Series(field_of_view),
         "solar_illumination": pd.Series(solar_illumination),
         "crossdetector_parallax": pd.Series(crossdetector_parallax)
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
    I_toa = np.divide(I.astype('float'), 1E4)
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
    data : numpy.array, size=(_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    See Also
    --------
    list_central_wavelength_msi : creates a dataframe for the MSI instrument
    read_stack_s2 : reading several Sentinel-2 bands at once into a stack
    dhdt.generic.read_geo_image : basic function to import geographic imagery data

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

def read_stack_s2(s2_df):
    """
    read imagery data of interest into an three dimensional np.array

    Parameters
    ----------
    s2_df : pandas.Dataframe
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2

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
    list_central_wavelength_msi : creates a dataframe for the MSI instrument
    get_S2_image_locations : provides dataframe with specific file locations
    read_band_s2 : reading a single Sentinel-2 band
    """
    assert isinstance(s2_df, pd.DataFrame), ('please provide a dataframe')
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

    Notes
    -----
    The metadata is scattered over the file structure of Sentinel-2, L1C

    .. code-block:: text

        * S2X_MSIL1C_20XX...
        ├ AUX_DATA
        ├ DATASTRIP
        │  └ DS_XXX_XXXX...
        │     └ QI_DATA
        │        └ MTD_DS.xml <- metadata about the data-strip
        ├ GRANULE
        │  └ L1C_TXXXX_XXXX...
        │     ├ AUX_DATA
        │     ├ IMG_DATA
        │     ├ QI_DATA
        │     └ MTD_TL.xml <- metadata about the tile
        ├ HTML
        ├ rep_info
        ├ manifest.safe
        ├ INSPIRE.xml
        └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C
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
    Zn : numpy.array, size=(m,n), dtype=float
        array of the solar zenith angles, in degrees.
    Az : numpy.array, size=(m,n), dtype=float
        array of the solar azimuth angles, in degrees.

    Notes
    -----
    The computation of the solar angles is done in two steps: 1) Computation of
    the solar angles in J2000; 2) Transformation of the vector to the mapping
    frame.

    - 1) The outputs of the first step is the solar direction normalized vector
         with the Earth-Sun distance, considering that the direction of the sun
         is the same at the centre of the Earth and at the centre of the
         Sentinel-2 satellite.

    - 2) Attitude of the satellite platform are used to rotate the solar vector
         to the mapping frame. Also Ground Image Calibration Parameters (GICP
         Diffuser Model) are used to transform from the satellite to the
         diffuser, as Sentinel-2 has a forward and backward looking sensor
         configuration. The diffuser model also has stray-light correction and
         a Bi-Directional Reflection Function model.

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

    The metadata is scattered over the file structure of Sentinel-2, L1C

        .. code-block:: text

            * S2X_MSIL1C_20XX...
            ├ AUX_DATA
            ├ DATASTRIP
            │  └ DS_XXX_XXXX...
            │     └ QI_DATA
            │        └ MTD_DS.xml <- metadata about the data-strip
            ├ GRANULE
            │  └ L1C_TXXXX_XXXX...
            │     ├ AUX_DATA
            │     ├ IMG_DATA
            │     ├ QI_DATA
            │     └ MTD_TL.xml <- metadata about the tile
            ├ HTML
            ├ rep_info
            ├ manifest.safe
            ├ INSPIRE.xml
            └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - s2 : Sentinel-2
    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C
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
                        boi=None):
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
    Zn : numpy.array, size=(m,n), dtype=float
        array of the solar zenith angles, in degrees.
    Az : numpy.array, size=(m,n), dtype=float
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

    The metadata is scattered over the file structure of Sentinel-2, L1C

        .. code-block:: text

            * S2X_MSIL1C_20XX...
            ├ AUX_DATA
            ├ DATASTRIP
            │  └ DS_XXX_XXXX...
            │     └ QI_DATA
            │        └ MTD_DS.xml <- metadata about the data-strip
            ├ GRANULE
            │  └ L1C_TXXXX_XXXX...
            │     ├ AUX_DATA
            │     ├ IMG_DATA
            │     ├ QI_DATA
            │     └ MTD_TL.xml <- metadata about the tile
            ├ HTML
            ├ rep_info
            ├ manifest.safe
            ├ INSPIRE.xml
            └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - s2 : Sentinel-2
    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C

    """
    if boi is None:
        boi = list_central_wavelength_msi()
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
        if not np.any(det_list==did):
            continue
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
            if Zn_samp.size==0:
                continue

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

    The metadata is scattered over the file structure of Sentinel-2, L1C

        .. code-block:: text

            * S2X_MSIL1C_20XX...
            ├ AUX_DATA
            ├ DATASTRIP
            │  └ DS_XXX_XXXX...
            │     └ QI_DATA
            │        └ MTD_DS.xml <- metadata about the data-strip
            ├ GRANULE
            │  └ L1C_TXXXX_XXXX...
            │     ├ AUX_DATA
            │     ├ IMG_DATA
            │     ├ QI_DATA
            │     └ MTD_TL.xml <- metadata about the tile
            ├ HTML
            ├ rep_info
            ├ manifest.safe
            ├ INSPIRE.xml
            └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - s2 : Sentinel-2
    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C
    """
    root = get_root_of_table(path, fname)
    Zn, Az = [], []
    for att in root.iter('Mean_Sun_Angle'):
        Zn = float(att[0].text)
        Az = float(att[1].text)
    return Zn, Az

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
    boi : pandas.DataFrame
        list with bands of interest
    geoTransform : tuple
        affine transformation coefficients

    Returns
    -------
    det_stack : numpy.array, size=(msk_dim[0],msk_dim[1],len(boi)), dtype=int8
        array where each pixel has the ID of the detector, of a specific band

    See Also
    --------
    list_central_wavelength_msi : creates a dataframe for the MSI instrument
    read_view_angles_s2 : read grid of Sentinel-2 observation angles

    Notes
    -----
    The metadata is scattered over the file structure of Sentinel-2, L1C

    .. code-block:: text

        * S2X_MSIL1C_20XX...
        ├ AUX_DATA
        ├ DATASTRIP
        │  └ DS_XXX_XXXX...
        │     └ QI_DATA
        │        └ MTD_DS.xml <- metadata about the data-strip
        ├ GRANULE
        │  └ L1C_TXXXX_XXXX...
        │     ├ AUX_DATA
        │     ├ IMG_DATA
        │     ├ QI_DATA
        │     └ MTD_TL.xml <- metadata about the tile
        ├ HTML
        ├ rep_info
        ├ manifest.safe
        ├ INSPIRE.xml
        └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C

    Example
    -------
    >>> path_meta = '/GRANULE/L1C_T15MXV_A027450_20200923T163313/QI_DATA'
    >>> boi = ['red', 'green', 'blue', 'near infrared']
    >>> s2_df = list_central_wavelength_s2()
    >>> boi_df = s2_df[s2_df['common_name'].isin(boi)]
    >>> geoTransform = (600000.0, 10.0, 0.0, 10000000.0, 0.0, -10.0)
    >>>
    >>> det_stack = read_detector_mask(path_meta, boi_df, geoTransform)
    """
    assert isinstance(boi, pd.DataFrame), ('please provide a dataframe')

    if len(geoTransform)>6: # also image size is given
        msk_dim = (geoTransform[-2], geoTransform[-1], len(boi))
        det_stack = np.zeros(msk_dim, dtype='int8')  # create stack
    else:
        msk_dim = None

    for i in range(len(boi)):
        im_id = boi.index[i] # 'B01' | 'B8A'
        if type(im_id) is int:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_B'+ \
                                  f'{im_id:02.0f}' + '.gml')
        else:
            f_meta = os.path.join(path_meta, 'MSK_DETFOO_'+ im_id +'.gml')
        dom = ElementTree.parse(glob.glob(f_meta)[0])
        root = dom.getroot()

        if msk_dim is None:
            msk_dim = get_msk_dim_from_gml(gml_struct)
            msk_dim = msk_dim + (len(boi),)
            det_stack = np.zeros(msk_dim, dtype='int8') # create stack

        mask_members = root[2]
        for k in range(len(mask_members)):
            pos_arr, det_num = get_xy_poly_from_gml(mask_members, k)

            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
            ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
            # make mask
            msk = Image.new("L", [np.size(det_stack,0), np.size(det_stack,1)], 0)
            ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])),
                                        outline=det_num, fill=det_num)
            msk = np.array(msk)
            det_stack[:,:,i] = np.maximum(det_stack[:,:,i], msk)
    return det_stack

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
    rec_time : numpy.datetime64
        time of image acquisition

    Notes
    -----
    The recording time is the average sensing within the tile, which is a
    combination of forward and backward looking datastrips. Schematically, this
    might look like:

        .. code-block:: text

             x recording time of datastrip, i.e.: the start of the strip
             × recording time of tile, i.e.: the weighted average
                                        _________  datastrip
                              ____x____/        /_________
                             /        /        /        /
                            /        /        /        /
                           /        /        /        /
                          /        /        /        /
                         /        /        /        /
                        /        /        /        /
                       /    ┌---/------┐ /        /
                      /     |  /       |/        /
                     /      | /  ×     /        /
                    /       |/        /|       /
                   /        /--------/-┘      /
                  /        /  tile  /        /
                 /        /        /        /
                /        /        /        /
               /        /        /        /
              /        /        /        /
             /        _________/        /
             _________        /________/

    Examples
    --------
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

def get_timing_stack_s2(s2_df, det_stack,
                        spac=10, sat_height=750E3, sat_velo=7.5):
    """
    Using the parallax angles, geometric and positioning info to get the time
    configuration of the acquisition by the instrument onboard Sentinel-2

    Parameters
    ----------
    s2_df : pandas.DataFrame
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2
    det_stack : numpy.array, size=(m,n,b), ndim={2,3}, dtype=int
        numbers of the different detectors for different bands, given by s2_df
    path_det: str
        asd
    spac : float, unit=meter, default=10
        pixel spacing

    Returns
    -------
    tim_stack : np.array, size=(m,n,b), ndim={2,3}, unit=seconds
        relative timing of acquisition for different bands
    cross_grd : np.array, size=(m,n), ndim=2, unit=meter
        cross orbit reference frame

    See Also
    --------
    list_central_wavelength_msi : creates a dataframe for the MSI instrument
    read_detector_mask : get ids of the different sensors that make up an image
    """
    assert isinstance(s2_df, pd.DataFrame), ('please provide a dataframe')

    # get_bearing_from_detector_mask
    psi = get_flight_bearing_from_detector_mask_s2(det_stack[:, :, 0])

    # construct grid
    x_grd, y_grd = np.meshgrid(np.linspace(1, det_stack.shape[1],
                                           det_stack.shape[1]),
                               np.linspace(1, det_stack.shape[0],
                                           det_stack.shape[0]),
                               indexing='xy')
    t_grd = -np.sin(np.deg2rad(psi)) * x_grd + np.cos(np.deg2rad(psi)) * y_grd
    t_grd *= line_period.astype('float')

    cross_grd = -np.cos(np.deg2rad(psi))*x_grd - np.sin(np.deg2rad(psi))*y_grd
    cross_grd *= spac # convert from pixels to meters

    cross_parallax_timing = (1E9 * s2_df.crossdetector_parallax * height /
                             sat_velo).to_numpy().astype('timedelta64[ns]')

    # odd detector numbers are forward, thus earlier
    band_timing = cross_parallax_timing / 2

    tim_stack = np.zeros_like(det_stack, dtype=float)
    # run through stack, band per band
    for band_id in range(tim_stack.shape[2]):
        tim_band = 1 - 2 * ((det_stack[:, :, band_id] % 2) == 0).astype('float')
        tim_band *= band_timing[band_id].astype('float')
        tim_band += t_grd
        tim_band /= np.timedelta64(int(1E9), 'ns').astype('float')
        tim_stack[:, :, band_id] = tim_band
    return tim_stack, cross_grd

def get_xy_poly_from_gml(gml_struct,idx):
    # get detector number from meta-data
    det_id = gml_struct[idx].attrib
    det_id = list(det_id.items())[0][1].split('-')[2]
    det_num = int(det_id)

    # get footprint
    pos_dim = gml_struct[idx][1][0][0][0][0].attrib
    pos_dim = int(list(pos_dim.items())[0][1])
    pos_list = gml_struct[idx][1][0][0][0][0].text
    pos_row = [float(s) for s in pos_list.split(' ')]
    pos_arr = np.array(pos_row).reshape((int(len(pos_row) / pos_dim), pos_dim))
    return pos_arr, det_num

def get_msk_dim_from_gml(gml_struct):
    # find dimensions of array through its map extent in metadata
    lower_corner = np.fromstring(gml_struct[1][0][0].text, sep=' ')
    upper_corner = np.fromstring(gml_struct[1][0][1].text, sep=' ')
    lower_x, lower_y = map2pix(geoTransform, lower_corner[0], lower_corner[1])
    upper_x, upper_y = map2pix(geoTransform, upper_corner[0], upper_corner[1])

    msk_dim = (np.maximum(lower_x, upper_x).astype(int),
               np.maximum(lower_y, upper_y).astype(int))
    return msk_dim



def read_cloud_mask(path_meta, geoTransform):
    """

    Parameters
    ----------
    path_meta : str
        directory where meta data is situated, 'MSK_CLOUDS_B00.gml' is typically
        the file of interest
    geoTransform : tuple
        affine transformation coefficients

    Returns
    -------
    msk_clouds : numpy.array, size=(m,n), dtype=int
        array which highlights where clouds could be
    """

    f_meta = os.path.join(path_meta, 'MSK_CLOUDS_B00.gml')
    dom = ElementTree.parse(glob.glob(f_meta)[0])
    root = dom.getroot()

    if len(geoTransform)>6: # also image size is given
        msk_dim = (geoTransform[-2], geoTransform[-1])
        msk_clouds = np.zeros(msk_dim, dtype='int8')
    else:
        msk_dim = get_msk_dim_from_gml(root)
        msk_clouds = np.zeros(msk_dim, dtype='int8')  # create stack

    if len(root)>2: # look into meta-data for cloud polygons
        mask_members = root[2]
        for k in range(len(mask_members)):
            pos_arr = get_xy_poly_from_gml(mask_members, k)[0]

            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:, 0], pos_arr[:, 1])
            ij_arr = np.hstack((j_arr[:, np.newaxis], i_arr[:, np.newaxis]))

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

def get_flight_bearing_from_detector_mask_s2(Det):
    """

    Parameters
    ----------
    Det : numpy.array, size=(m,n), ndim={2,3}, dtype=integer
        array with numbers of the different detectors in Sentinel-2

    Returns
    -------
    psi : float, unit=degrees
        general bearing of the satellite
    """
    if Det.ndim==3:
        D = Det[:, :, 0]
    else:
        D = Det

    D_x = convolve2d(D, np.outer([+1., +2., +1.], [-1., 0., +1.]),
                     boundary='fill', mode='same')
    D_x[0, :], D_x[-1, :], D_x[:, 0], D_x[:, -1] = 0, 0, 0, 0
    D_tr = np.abs(D_x) >= 2

    L, Lab_num = label(D_tr, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    psi = np.zeros(Lab_num)
    for i in range(Lab_num):
        [i_x, j_x] = np.where(L == i + 1)
        psi[i] = np.rad2deg(np.arctan2(np.min(j_x) - np.max(j_x),
                                       np.min(i_x) - np.max(i_x)))
    psi = np.median(psi)
    return psi

# use the detector start and finish to make a selection for the flight line
def get_flight_bearing_from_gnss_s2(path, spatialRef, rec_tim, fname='MTD_DS.xml'):
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
    az : numpy.array, size=(m,1), float
        array with argument values, based upon the map projection given
    """

    sat_tim,sat_xyz,_,_ = get_flight_path_s2(path, fname=fname)
    sat_xy = ecef2map(sat_xyz, spatialRef)

    dif_tim = sat_tim - rec_tim
    idx = np.argmin(np.abs(dif_tim))
    dif_xy = sat_xy[idx + 1] - sat_xy[idx]

    az = np.arctan2(dif_xy[0], dif_xy[1]) * 180 / np.pi
    return az

def get_flight_path_s2(ds_path, fname='MTD_DS.xml', s2_dict=None):
    """

    It is also possible to only give the dictionary, as this has such metadata
    within.

    Parameters
    ----------
    ds_path : string
        location of the metadata file
    fname : string, default='MTD_DS.xml'
        name of the xml-file that has the metadata
    s2_dict : dictonary
        metadata of the Sentinel-2 platform

    Returns
    -------
    sat_time : numpy.array, size=(m,1), dtype=np.datetime64, unit=ns
        time stamp of the satellite positions
    sat_xyz : numpy.array, size=(m,3), dtype=float, unit=meter
        3D coordinates of the satellite within an Earth centered Earth fixed
        (ECEF) frame.
    sat_err : numpy.array, size=(m,3), dtype=float, unit=meter
        error estimate of the 3D coordinates given by "sat_xyz"
    sat_uvw : numpy.array, size=(m,3), dtype=float, unit=meter sec-1
        3D velocity vectors of the satellite within an Earth centered Earth
        fixed (ECEF) frame.
    s2_dict : dictonary
        updated with keys: "xyz", "uvw", "time", "error", "altitude", "speed"

    See Also
    --------
    get_flight_orientation_s2 : get the quaternions of the flight path

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
    """
    if isinstance(ds_path,dict):
        # only dictionary is given, which already has all metadata within
        s2_dict = ds_path
        assert ('MTD_DS_path' in s2_dict.keys()), 'please run get_s2_dict'

        root = get_root_of_table(s2_dict['MTD_DS_path'], 'MTD_MSIL1C.xml')
    else:
        root = get_root_of_table(ds_path, fname)

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
            if point[2].attrib['unit'] == 'mm/s': uvw *= 1E-3
            # convert to meters per second

            # 'GPS_TIME'
            gps_tim = np.datetime64(point[4].text, 'ns')

            # fill in the arrays
            sat_time[idx] = gps_tim
            sat_xyz[idx, :], sat_err[idx, :], sat_uvw[idx, :] = xyz, err, uvw
    if s2_dict is None:
        return sat_time, sat_xyz, sat_err, sat_uvw
    else: # include into dictonary
        s2_dict.update({'xyz': sat_xyz, 'uvw': sat_uvw, 'error': sat_error})
        if 'time' not in s2_dict:
            s2_dict.update({'time': sat_time})
        # estimate the altitude above the ellipsoid, and platform speed
        llh = ecef2llh(sat_xyz)
        velo = np.sqrt(np.sum(sat_uvw**2, axis=1))
        s2_dict.update({'altitude': np.squeeze(llh[:,-1]),
                        'velocity': np.squeeze(velo)})
        return s2_dict

def get_flight_orientation_s2(ds_path, fname='MTD_DS.xml', s2_dict=None):
    """ get the flight path and orientations of the Sentinel-2 satellite during
    acquisition.

    It is also possible to only give the dictionary, as this has such metadata
    within.

    Parameters
    ----------
    ds_path : string
        directory where the meta-data is located
    fname : string, default='MTD_DS.xml'
        filename of the meta-data file
    s2_dict : dictonary, default=None
        metadata of the Sentinel-2 platform
    Returns
    -------
    sat_time : numpy.array, size=(m,1), unit=nanosec
        satellite timestamps
    sat_angles : numpy.array, size=(m,3)
        satellite orientation angles, given in ECEF
    s2_dict : dictonary
        updated with keys: "time", "quat"

    See Also
    --------
    get_flight_path_s2 : get positions of the flight path
    """
    if isinstance(ds_path,dict):
        # only dictionary is given, which already has all metadata within
        s2_dict = ds_path
        assert ('MTD_DS_path' in s2_dict.keys()), 'please run get_s2_dict'

        root = get_root_of_table(s2_dict['MTD_DS_path'], 'MTD_MSIL1C.xml')
    else:
        root = get_root_of_table(ds_path, fname)

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
    if s2_dict is None:
        return sat_time, sat_angles
    else: # include into dictonary
        s2_dict.update({'quat': sat_angles})
        if 'time' not in s2_dict:
            s2_dict.update({'time': sat_time})
        return s2_dict

def get_integration_and_sampling_time_s2(ds_path, fname='MTD_DS.xml',
                                         s2_dict=None): #todo: create s2_dict methodology
    """

    Parameters
    ----------
    ds_path : string
        location where metadata of datastrip is situated
    fname : string, default='MTD_DS.xml'
        metadata filename
    s2_dict : dictionary, default=None
        metadata dictionary, can be used instead of the string based method

    Returns
    -------

    Notes
    -----
    The metadata is scattered over the file structure of Sentinel-2, L1C

    .. code-block:: text

        * S2X_MSIL1C_20XX...
        ├ AUX_DATA
        │  └ DS_XXX_XXXX...
        │     └ QI_DATA
        │        └ MTD_DS.xml <- metadata about the data-strip
        ├ GRANULE
        │  └ L1C_TXXXX_XXXX...
        │     ├ AUX_DATA
        │     ├ IMG_DATA
        │     ├ QI_DATA
        │     └ MTD_TL.xml <- metadata about the tile
        ├ HTML
        ├ rep_info
        ├ manifest.safe
        ├ INSPIRE.xml
        └ MTD_MSIL1C.xml <- metadata about the product

    The following acronyms are used:

    - DS : datastrip
    - TL : tile
    - QI : quality information
    - AUX : auxiliary
    - MTD : metadata
    - MSI : multi spectral instrument
    - L1C : product specification,i.e.: level 1, processing step C

    """
    if isinstance(ds_path,dict):
        # only dictionary is given, which already has all metadata within
        s2_dict = ds_path
        assert ('MTD_DS_path' in s2_dict.keys()), 'please run get_s2_dict'

        root = get_root_of_table(s2_dict['MTD_DS_path'], 'MTD_MSIL1C.xml')
    else:
        root = get_root_of_table(ds_path, fname)

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

def get_view_angles_s2(Det, boi, s2_dict, Z=None):

    s2_dict = get_flight_path_s2(s2_dict)

    line_tim, integration_tim = get_integration_and_sampling_time_s2(s2_dict)

    # get map coordinates of imagery

    # transform to angles, via intrinsic camera parameters

    # include DEM

    # transform to ECEF
    XXX = []
    return XXX

def get_intrinsic_camera_mat_s2(s2_df, det, boi):
    """

    Parameters
    ----------
    s2_df : pandas.DataFrame
        metadata and general multispectral information about the MSI
        instrument that is onboard Sentinel-2
    det: list
        list with detector numbers that are within the tile
    boi : list
        list with bands of interest

    Returns
    -------
    F : numpy.array, size=(3,3)
        fundamental matrix

    Notes
    -----
    The pushbroom detector onboard Sentinel-2 has the folowing configuration:

        .. code-block:: text

                               nadir
         det1                    |
         ====#   #===#   #===#   #===#   #===#   #===#
            #===#   #===#   #===#   #===#   #===#   #====
                                 |                   det12
         ===== : 2592 pixels for 10 meter, 1296 pixels for 20-60 meter
            # : 98 pixels overlap for 20 meter data

    References
    ----------
    .. [1] Hartley & Gupta, "Linear pushbroom cameras" Proceedings of the
       European conference on computer vision, 1994.
    .. [2] Moons et al. "3D reconstruction from multiple images part 1:
       Principles." Foundations and trends® in computer graphics and vision,
       vol.4(4) pp.287-404, 2010.
    """
    #convert focal length to pixel scale

    s2_oi = s2_df[s2_df['common_name'].isin(boi)]

    # get focal length
    f = np.divide(s2_oi['focal_length'].to_numpy()[0]/1E-6,
                  s2_oi['across_pixel_size'].to_numpy()[0])
    F = np.eye(3)
    F[1,1], F[0,0] = f, f

    # get slant angle
    F[0,2] = np.sin(np.deg2rad(s2_oi['crossdetector_parallax'].to_numpy())) * f
    # is positive or negative...

    det_step = 2592-(98*2*2)
    det_step += 98

    det_out = np.ceil(np.abs(det-6.5))*det_step
    det_in = (det_out - det_step) - (98*2)
    det_out +=  98*2
    return F


