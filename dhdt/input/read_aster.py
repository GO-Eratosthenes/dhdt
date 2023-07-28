# generic libraries
import os
import glob

import numpy as np
import pandas as pd

# geospatial libaries
from osgeo import gdal, osr
from datetime import datetime, timedelta

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import \
    ecef2llh, estimate_geoTransform, ll2map, ref_rotate
from dhdt.generic.gis_tools import get_utm_zone, create_crs_from_utm_zone
from dhdt.generic.handler_aster import get_as_image_locations
from dhdt.postprocessing.solar_tools import vector_to_sun_angles


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

    See Also
    --------
    dhdt.input.read_sentinel2.list_central_wavelength_msi :
        for the instrument data of MSI on Sentinel-2
    dhdt.input.read_landsat.list_central_wavelength_oli :
        for the instrument data of Landsat 8 or 9
    dhdt.input.read_venus.list_central_wavelength_vssc :
        for the instrument data of VSSC on the VENµS satellite

    Examples
    --------
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
    >>> sw_df = as_df[as_df['gsd']==30]
    >>> sw_df.index
    Index(['B04', 'B05', 'B06', 'B07', 'B08', 'B09'], dtype='object')

    """
    center_wavelength = {
        "B01": 560,
        "B02": 660,
        "B3B": 810,
        "B3N": 810,
        "B04": 1650,
        "B05": 2165,
        "B06": 2205,
        "B07": 2360,
        "B08": 2330,
        "B09": 2395,
        "B10": 8300,
        "B11": 8650,
        "B12": 9100,
        "B13": 10600,
        "B14": 11300,
    }
    full_width_half_max = {
        "B01": 80,
        "B02": 60,
        "B3B": 100,
        "B3N": 100,
        "B04": 100,
        "B05": 40,
        "B06": 40,
        "B07": 50,
        "B08": 70,
        "B09": 70,
        "B10": 350,
        "B11": 350,
        "B12": 350,
        "B13": 700,
        "B14": 700,
    }
    bandid = {
        "B01": 0,
        "B02": 1,
        "B3B": 2,
        "B3N": 3,
        "B04": 4,
        "B05": 5,
        "B06": 6,
        "B07": 7,
        "B08": 8,
        "B09": 9,
        "B10": 10,
        "B11": 11,
        "B12": 12,
        "B13": 13,
        "B14": 14,
    }
    gsd = {
        "B01": 15,
        "B02": 15,
        "B3B": 15,
        "B3N": 15,
        "B04": 30,
        "B05": 30,
        "B06": 30,
        "B07": 30,
        "B08": 30,
        "B09": 30,
        "B10": 90,
        "B11": 90,
        "B12": 90,
        "B13": 90,
        "B14": 90,
    }
    along_track_view_angle = {
        "B01": 0.,
        "B02": 0.,
        "B3B": -27.6,
        "B3N": 0.,
        "B04": 0.,
        "B05": 0.,
        "B06": 0.,
        "B07": 0.,
        "B08": 0.,
        "B09": 0.,
        "B10": 0.,
        "B11": 0.,
        "B12": 0.,
        "B13": 0.,
        "B14": 0.,
    }
    solar_illumination = {
        "B01": 1848.,
        "B02": 1549.,
        "B3B": 1114.,
        "B3N": 1114.,
        "B04": 225.4,
        "B05": 86.63,
        "B06": 81.85,
        "B07": 74.85,
        "B08": 66.49,
        "B09": 59.85,
        "B10": 0,
        "B11": 0,
        "B12": 0,
        "B13": 0,
        "B14": 0,
    }
    common_name = {
        "B01": 'blue',
        "B02": 'green',
        "B3B": 'red',
        "B3N": 'stereo',
        "B04": 'swir16',
        "B05": 'swir22',
        "B06": 'swir22',
        "B07": 'swir22',
        "B08": 'swir22',
        "B09": 'swir22',
        "B10": 'lwir',
        "B11": 'lwir',
        "B12": 'lwir',
        "B13": 'lwir',
        "B13": 'lwir',
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


def read_band_as(im_path, band='3N'):
    if im_path.endswith('.hdf'):
        data, spatialRef, geoTransform, targetprj = \
            read_band_as_hdf(im_path, band=band)
    else:
        data, spatialRef, geoTransform, targetprj = \
            read_geo_image(im_path)
    return data, spatialRef, geoTransform, targetprj


def read_band_as_hdf(path, band='3N'):
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
    data : numpy.ndarray, size=(_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    References
    ----------
    .. [wwwASTR] https://asterweb.jpl.nasa.gov/content/03_data/04_Documents/ASTERHigherLevelUserGuideVer2May01.pdf
    .. [wwwGCTP] https://www.cmascenter.org/ioapi/documentation/all_versions/html/GCTP.html

    """
    assert os.path.exists(path), ('file does not seem to be present')

    if len(band) == 3:
        band = band[1:]
    # get imagery data
    img = gdal.Open(glob.glob(path)[0], gdal.GA_ReadOnly)
    ds = img.GetSubDatasets()
    for idx, arr in enumerate(ds):
        arr_num = arr[0].split('ImageData')
        if (len(arr_num) > 1) and (arr_num[-1] == band):
            pointer = idx
    data = gdal.Open(img.GetSubDatasets()[pointer][0],
                     gdal.GA_ReadOnly).ReadAsArray()

    # get projection data
    ul_xy = np.fromstring(img.GetMetadata()['UpperLeftM'.upper()],
                          sep=', ')[::-1]
    proj_str = img.GetMetadata()['MapProjectionName'.upper()]
    utm_code = img.GetMetadata()['UTMZoneCode1'.upper()]

    if proj_str == 'Universal Transverse Mercator':
        epsg_code = 32600
        epsg_code += int(utm_code)
        center_lat = np.fromstring(img.GetMetadata()['SceneCenter'.upper()],
                                   sep=', ')[0]
        if center_lat < 0:  # majprity is in southern hemisphere
            epsg_code += 100

        targetprj = osr.SpatialReference()
        targetprj.ImportFromEPSG(epsg_code)
        spatialRef = targetprj.ExportToWkt()
        geoTransform = (ul_xy[0], 15., 0., ul_xy[1], 0., -15., data.shape[0],
                        data.shape[1])
    else:
        raise ('projection not implemented yet')

    return data, spatialRef, geoTransform, targetprj


def read_stack_as(path, as_df, level=1):
    if level == 1:
        data, spatialRef, geoTransform, targetprj = read_stack_as_l1(
            path, as_df)
    elif level == 3:
        data, spatialRef, geoTransform, targetprj = read_stack_as_l3(as_df)
    return data, spatialRef, geoTransform, targetprj


def read_stack_as_l1(path, as_df):
    """ read the band stack of Level-1 data

    Parameters
    ----------
    path : string
        location where the imagery data is situated
    as_df : pandas.DataFrame
        metadata and general multispectral information about the instrument that
        is onboard Terra

    Returns
    -------
    im_stack : numpy.ndarray, size=(_,_,_)
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
    if any((path == '', path is None, 'filepath' not in as_df)):
        assert False, ('please first run "get_as_image_locations" to find' +
                       ' the proper file locations, or provide path directly')

    for idx, val in enumerate(as_df.index):
        # resolve index and naming issue for ASTER metadata
        boi = val[1:].lstrip('0')  # remove leading zero if present
        im_path = as_df['filepath'][idx]
        if idx == 0:
            im_stack, spatialRef, geoTransform, targetprj = \
                read_band_as(im_path, band=boi)
        else:  # stack others bands
            if im_stack.ndim == 2:
                im_stack = np.atleast_3d(im_stack)
            band = np.atleast_3d(read_band_as(im_path, band=boi)[0])
            im_stack = np.concatenate((im_stack, band), axis=2)
    return im_stack, spatialRef, geoTransform, targetprj


def read_stack_as_l3(as_df):
    assert isinstance(as_df, pd.DataFrame), ('please provide a dataframe')
    assert 'filepath' in as_df, ('please first run "get_as_image_locations"' +
                                 ' to find the proper file locations')
    assert len(set(as_df['gsd'].array))==1, \
        ('only single resolution is implemented here')

    for val, idx in enumerate(as_df.index):
        f_path = as_df['filepath'][idx]
        if val == 0:
            im_stack, spatialRef, geoTransform, targetprj = read_geo_image(
                f_path)
        else:
            if im_stack.ndim == 2:
                im_stack = np.atleast_3d(im_stack)
            band = np.atleast_3d(read_geo_image(f_path)[0])
            im_stack = np.concatenate((im_stack, band), axis=2)

    # create masked array, if this is not already done so
    if type(im_stack) not in (np.ma.core.MaskedArray, ):
        Mask = np.all(im_stack == 0, axis=2)[..., np.newaxis]
        im_stack = np.ma.array(im_stack,
                               mask=np.tile(Mask, (1, 1, im_stack.shape[2])))
    return im_stack, spatialRef, geoTransform, targetprj


def get_flight_path_as(as_path,
                       fname='AST_L1A_*.Ancillary_Data.txt',
                       as_dict=None):
    """

    Parameters
    ----------
    as_path : string
    fname : string
        file name of the accillary data
    as_dict: dictionary

    Returns
    -------

    See Also
    --------
    dhdt.input.read_aster.list_central_wavelength_as

    """

    if fname.find('*') != -1:  # search for ancillary data file
        starter, ending = fname.split('*')
        file_list = [
            x for x in os.listdir(as_path)
            if (x.endswith(ending) and x.startswith(starter))
        ]
        if len(file_list) != 1:
            assert False, 'file does not seem to be present'
        else:
            fname = file_list[0]

    with open(os.path.join(as_path, fname)) as file:
        counter = 0
        for line in file:
            counter += 1
            if counter == 1: continue
            line = " ".join(line.rstrip().split())
            line = np.fromstring(line, dtype=int, sep=' ')

            time_sat, time_cor = line[:4], line[19:22]
            time_sat[1:] += time_cor

            time_utc = d_space = datetime.strptime("01/01/1958", "%m/%d/%Y")
            time_utc += timedelta(days=int(time_sat[0]),
                                  minutes=int(time_sat[1]),
                                  seconds=int(time_sat[2]),
                                  milliseconds=int(time_sat[3]))
            time_sat = np.datetime64(time_utc, 'ns')
            sat_ecef, sat_velo = line[22:25].astype(float), \
                                 line[25:28].astype(float)

            if counter == 2:
                sat_time, sat_xyz, sat_uvw = time_sat.copy(), \
                                             np.atleast_2d(sat_ecef), \
                                             np.atleast_2d(sat_velo)
            else:
                sat_time, sat_xyz, sat_uvw = np.vstack((sat_time, time_sat)), \
                                             np.vstack((sat_xyz, sat_ecef)), \
                                             np.vstack((sat_uvw, sat_velo))
    sat_uvw *= 244E-9
    sat_xyz *= .125
    if as_dict is None:
        return sat_time, sat_xyz, sat_uvw
    else:
        as_dict.update({
            'gps_xyz': sat_xyz,
            'gps_uvw': sat_uvw,
            'gps_tim': sat_time
        })
        # estimate the altitude above the ellipsoid, and platform speed
        llh = ecef2llh(sat_xyz)
        velo = np.sqrt(np.sum(sat_uvw**2, axis=1))
        as_dict.update({
            'altitude': np.squeeze(llh[:, -1]),
            'velocity': np.squeeze(velo)
        })
        return as_dict


def _read_meta_as_3d(ffull, angles=True):
    a, b, A, B = np.array([]), np.array([]), np.array([]), np.array([])
    with open(ffull) as f:
        for line in f:
            line = line.rstrip()
            if (line == "") and (a.size != 0):  # ending of a row
                A = np.vstack([A, a]) if A.size else a
                B = np.vstack([B, b]) if B.size else b
                a, b = np.array([]), np.array([])
            elif line != "":
                elem = np.fromstring(line, sep=' ')
                if angles:
                    elem = vector_to_sun_angles(elem[0], elem[1], elem[2])
                a0, b0 = np.atleast_2d(elem[0]), np.atleast_2d(elem[1])
                a = np.hstack([a, a0]) if a.size else a0
                b = np.hstack([b, b0]) if b.size else b0
    return A, B


def _read_meta_as_2d(ffull):
    assert os.path.isfile(ffull), 'make sure file exists'
    Dat = np.array([])
    with open(ffull) as f:
        for line in f:
            line = line.rstrip()
            if line == "":  # ending of a row
                continue
            elem = np.atleast_2d(np.fromstring(line, sep=' '))
            Dat = np.vstack([Dat, elem]) if Dat.size else elem
    return Dat


def _read_meta_as_1d(ffull):
    assert os.path.isfile(ffull), 'make sure file exists'
    arr, Dat = np.array([]), np.array([])
    with open(ffull) as f:
        for line in f:
            line = line.rstrip()
            if (line == "") and (arr.size != 0):  # ending of a row
                Dat = np.vstack([Dat, arr]) if Dat.size else arr
                arr = np.array([])
            elif line != "":
                elem = np.atleast_2d(np.fromstring(line, sep=' '))
                arr = np.hstack([arr, elem]) if arr.size else elem
    return Dat


def _pad_to_same(A, B, size_diff, constant_values=0):
    for idx, val in enumerate(size_diff):
        if np.sign(val) == -1:
            A = _pad_along_axis(A,
                                np.abs(val),
                                axis=idx,
                                constant_values=constant_values)
        else:
            B = _pad_along_axis(B,
                                val,
                                axis=idx,
                                constant_values=constant_values)
    return A, B


def _pad_along_axis(A, pad_size, axis=0, constant_values=0):
    if pad_size <= 0: return A

    npad = [(0, 0)] * A.ndim
    npad[axis] = (0, pad_size)

    A_new = np.pad(A,
                   pad_width=npad,
                   mode='constant',
                   constant_values=constant_values)
    return A_new


def read_view_angles_as(as_dir, boi_df=None):

    if not ('filepath' in boi_df.keys()):
        boi_df = get_as_image_locations(as_dir, boi_df)

    Zn, Az = np.array([]), np.array([])
    for idx, f_path in boi_df['filepath'].items():
        fnew = '.'.join(f_path.split('.')[:-2] + ['SightVector', 'txt'])
        zn, az = _read_meta_as_3d(fnew, angles=True)

        if Zn.size:
            # make compatable, as the latice grids can vary, especially for
            # the stereo bands
            dim_diff = np.array(Zn.shape[:2]) - np.array(zn.shape[:2])
            if np.any(dim_diff != 0):
                Zn, zn = _pad_to_same(Zn, zn, dim_diff, constant_values=np.nan)
                Az, az = _pad_to_same(Az, az, dim_diff, constant_values=np.nan)

            Zn = np.dstack([Zn, zn])
            Az = np.dstack([Az, az])
        else:
            Zn, Az = zn, az
    return Zn, Az


def read_lattice_as(as_dir, boi_df=None):
    if not ('filepath' in boi_df.keys()):
        boi_df = get_as_image_locations(as_dir, boi_df)

    Lat, Lon = np.array([]), np.array([])  # spherical coordiantes
    I, J = np.array([]), np.array([])  # local image frame
    for idx, f_path in boi_df['filepath'].items():
        fbase = f_path.split('.')[:-2]
        ϕ = _read_meta_as_2d('.'.join(fbase + ['Latitude', 'txt']))
        λ = _read_meta_as_2d('.'.join(fbase + ['Longitude', 'txt']))
        fnew = '.'.join(fbase + ['LatticePoint', 'txt'])
        i, j = _read_meta_as_3d(fnew, angles=False)

        if Lat.size:
            # make compatable, as the latice grids can vary, especially for
            # the stereo bands
            dim_diff = np.array(Lat.shape[:2]) - np.array(ϕ.shape[:2])
            if np.any(dim_diff != 0):
                Lat, ϕ = _pad_to_same(Lat, ϕ, dim_diff, constant_values=np.nan)
                Lon, λ = _pad_to_same(Lon, λ, dim_diff, constant_values=np.nan)
                I, i = _pad_to_same(I, i, dim_diff, constant_values=np.nan)
                J, j = _pad_to_same(J, j, dim_diff, constant_values=np.nan)

            Lat, Lon = np.dstack([Lat, ϕ]), np.dstack([Lon, λ])
            I, J = np.dstack([I, i]), np.dstack([J, j])
        else:
            Lat, Lon, I, J = ϕ, λ, i, j
    return Lat, Lon, I, J
