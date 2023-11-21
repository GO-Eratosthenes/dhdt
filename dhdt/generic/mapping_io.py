import datetime
import glob
import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
from osgeo import gdal, ogr, osr

from dhdt.__version__ import __version__
from dhdt.generic.mapping_tools import pix_centers
from dhdt.generic.unit_check import correct_geoTransform, is_crs_an_srs
from dhdt.generic.unit_conversion import deg2compass
from dhdt.testing.mapping_tools import create_local_crs


def read_geo_info(fname):
    """ This function takes as input the geotiff name and the path of the
    folder that the images are stored, reads the geographic information of
    the image

    Parameters
    ----------
    fname : string
        path and file name of a geotiff image

    Returns
    -------
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(8,1)
        affine transformation coefficients, but also giving the image
        dimensions
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    rows : integer, {x ∈ ℕ | x ≥ 0}
        number of rows in the image, that is its height
    cols : integer, {x ∈ ℕ | x ≥ 0}
        number of collumns in the image, that is its width
    bands : integer, {x ∈ ℕ | x ≥ 1}
        number of bands in the image, that is its depth

    See Also
    --------
    read_geo_image : basic function to import geographic imagery data
    """
    assert len(glob.glob(fname)) != 0, ('file does not seem to be present')

    img = gdal.Open(fname)
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    geoTransform = tuple(float(x) for x in geoTransform)
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    rows = int(img.RasterYSize)
    cols = int(img.RasterXSize)
    bands = img.RasterCount

    geoTransform += (
        rows,
        cols,
    )
    return spatialRef, geoTransform, targetprj, rows, cols, bands


def read_geo_image(fname, boi=np.array([]), no_dat=None):
    """ This function takes as input the geotiff name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    fname : string
        geotiff file name and path.
    boi : numpy.array, size=(k,1)
        bands of interest, if a multispectral image is read, a selection can
        be specified
    no_dat : {integer,float}
         no data value

    Returns
    -------
    data : {numpy.array, numpy.masked.array}, size=(m,n), ndim=2
        data array of the band
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(8,)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)

    See Also
    --------
    make_geo_im : basic function to write out geographic data
    read_geo_info : basic function to get meta data of geographic imagery
    read_nc_im : basic function to read geographic raster out of nc-file

    Examples
    --------
    >>> import os
    >>> from dhdt.generic.mapping_io import read_geo_image
    >>> fpath = os.path.join(os.getcwd(), "data.jp2" )

    >>> I, spatialRef, geoTransform, targetPrj = read_geo_image(fpath)

    >>> I_ones = np.zeros(I.shape, dtype=bool)
    >>> make_geo_im(I_ones, geoTransformM, spatialRefM, "ones.tif")
    """
    assert os.path.isfile(fname), ('file does not seem to be present')

    img = gdal.Open(fname)
    assert img is not None, ('could not open dataset ' + fname)

    # imagery can consist of multiple bands
    if len(boi) == 0:
        for counter in range(img.RasterCount):
            band = np.array(img.GetRasterBand(counter + 1).ReadAsArray())
            if no_dat is None:
                no_dat = img.GetRasterBand(counter + 1).GetNoDataValue()
            # create masked array
            if no_dat is not None:
                band = np.ma.array(band, mask=band == no_dat)
            data = band if counter == 0 else np.dstack(
                (data, np.atleast_3d(band)))
    else:
        num_bands = img.RasterCount
        assert (np.max(boi) +
                1) <= num_bands, 'bands of interest is out of range'
        for band_id, counter in enumerate(boi):
            band = np.array(img.GetRasterBand(band_id + 1).ReadAsArray())
            no_dat = img.GetRasterBand(counter + 1).GetNoDataValue()
            np.putmask(band, band == no_dat, np.nan)
            data = band if counter == 0 else np.dstack(
                (data, np.atleast_3d(band)))
    spatialRef = img.GetProjection()
    geoTransform = tuple(float(x)
                         for x in img.GetGeoTransform()) + data.shape[:2]
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj


def read_nc_image(fname, layer_name):
    assert os.path.isfile(fname), ('file does not seem to be present')

    ds = gdal.Open("NETCDF:{0}:{1}".format(fname, layer_name))
    assert ds is not None, ('could not open dataset ' + fname)

    data = np.array(ds.ReadAsArray())
    data = np.ma.array(data, mask=data == ds.GetRasterBand(1).GetNoDataValue())

    spatialRef = ds.GetProjection()
    geoTransform = ds.GetGeoTransform() + data.shape
    targetprj = osr.SpatialReference(wkt=ds.GetProjection())
    return data, spatialRef, geoTransform, targetprj


def _correct_sun_angle_string(sun_angles):
    if isinstance(sun_angles, list):
        sun_angles[0] = deg2compass(sun_angles[0])
        sun_angles = f'az:{int(sun_angles[0]):03d}-zn:{int(sun_angles[1]):02d}'
    return sun_angles


# output functions
def make_geo_im(Z,
                R,
                crs,
                fName,
                meta_descr='project Eratosthenes',
                no_dat=np.nan,
                sun_angles='az:360-zn:90',
                date_created='-0276-00-00'):
    """ Create georeferenced tiff file (a GeoTIFF)

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        band image
    R : list, size=(1,6)
        GDAL georeference transform of an image
    crs : string
        coordinate reference string
    fname : string
        filename for the image with extension
    no_dat : datatype, integer
        no data value
    sun_angles : string
        string giving meta data about the illumination angles
    date_created : string
        string given the acquistion date in +YYYY-MM-DD

    Examples
    --------

    >>> import os
    >>> fpath = os.path.join(os.getcwd(), "data.jp2")

    >>> (I, spatialRef, geoTransform, targetPrj) = read_geo_image(fpath)

    >>> Z_ones = np.zeros(Z.shape, dtype=bool)
    >>> make_geo_im(Z_ones, geoTransformM, spatialRefM, 'ones.tif')
    """
    if crs is None:
        crs = create_local_crs()
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()
    R = correct_geoTransform(R)
    sun_angles = _correct_sun_angle_string(sun_angles)

    bands = Z.shape[2] if Z.ndim == 3 else 1

    drv = gdal.GetDriverByName("GTiff")  # export image
    if Z.dtype == 'float64':  # make it type dependent
        gdal_dtype = gdal.GDT_Float64
        predictor = "3"
    elif Z.dtype == 'float32':
        gdal_dtype = gdal.GDT_Float32
        predictor = "3"
    elif Z.dtype in ('bool', 'uint8'):
        gdal_dtype = gdal.GDT_Byte
        predictor = "2"
    else:
        gdal_dtype = gdal.GDT_Int32
        predictor = "2"

    ds = drv.Create(
        fName,
        xsize=Z.shape[1],
        ysize=Z.shape[0],
        bands=bands,
        eType=gdal_dtype,
        options=['TFW=YES', 'COMPRESS=LZW', "PREDICTOR=" + predictor])

    # set metadata in datasource
    ds.SetMetadata({
        'TIFFTAG_SOFTWARE': 'dhdt v' + __version__,
        'TIFFTAG_ARTIST': 'bas altena and team Atlas',
        'TIFFTAG_COPYRIGHT': 'contains modified Copernicus data',
        'TIFFTAG_IMAGEDESCRIPTION': meta_descr,
        'TIFFTAG_RESOLUTIONUNIT': sun_angles,
        'TIFFTAG_DATETIME': date_created
    })

    # set georeferencing metadata
    if len(R) != 6:
        R = R[:6]

    ds.SetGeoTransform(R)
    ds.SetProjection(crs)
    if Z.ndim == 3:
        for count in np.arange(1, Z.shape[2] + 1, 1):
            band = ds.GetRasterBand(int(count))
            band.WriteArray(Z[:, :, count - 1], 0, 0)
            if count == 1:
                band.SetNoDataValue(no_dat)
            band = None
    else:
        ds.GetRasterBand(1).WriteArray(Z)
        ds.GetRasterBand(1).SetNoDataValue(no_dat)
    if ds is None:
        print(gdal.GetLastErrorMsg())
    ds = None
    del ds


def make_multispectral_vrt(df, fpath=None, fname='multispec.vrt'):
    """ virtual raster tile (VRT) is a description of datasets written in an
    XML format, it eases the display of multi-spectral data or other means.

    Parameters
    ----------
    df : pandas.DataFrame
        organization of the different spectral bands
    fpath : string
        path of the directory of interest
    fname : string
        file name of the virtual raster tile

    Examples
    --------
    >>> s2_df = list_central_wavelength_msi()
    >>> s2_df = s2_df[s2_df['gsd']==10]
    >>> s2_df, datastrip_id = get_s2_image_locations(IM_PATH, s2_df)

    create file in directory where imagery is situated
    >>> make_multispectral_vrt(s2_df, fname='multi_spec.vrt')
    """
    assert isinstance(df, pd.DataFrame), ('please provide a dataframe')
    assert 'filepath' in df, ('please first run "get_s2_image_locations"' +
                              ' to find the proper file locations')

    if fpath is None:
        fpath = os.path.commonpath(df.filepath.tolist())

    ffull = os.path.join(fpath, fname)
    vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour,
                                       addAlpha=False,
                                       separate=True,
                                       srcNodata=0)
    my_vrt = gdal.BuildVRT(ffull, [f + '.jp2' for f in df['filepath']],
                           options=vrt_options)
    if my_vrt is None:
        print(gdal.GetLastErrorMsg())
    my_vrt = None

    # modify the vrt-file to include band names
    tree = ElementTree.parse(ffull)
    root = tree.getroot()
    for idx, band in enumerate(root.iter("VRTRasterBand")):
        description = ElementTree.SubElement(band, "Description")
        description.text = df.common_name[idx]
    tree.write(ffull)  # update the file on disk
    return


def make_nc_image(Z,
                  R,
                  crs,
                  fName,
                  date_created=datetime.datetime(2022, 11, 28)):
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()

    X, Y = pix_centers(R, make_grid=False)

    dsout = Dataset(fName, "w", format="NETCDF4")

    if is_crs_an_srs(crs):
        nc_struct = ('times', 'nor', 'eas')
        nor = dsout.createDimension('nor', Y.size)
        nor = dsout.createVariable('nor', 'f4', ('nor', ))
        nor.standard_name = 'northing'
        nor.units = 'metres_north'
        nor.axis = "Y"
        nor[:] = Y

        eas = dsout.createDimension('eas', X.size)
        eas = dsout.createVariable('eas', 'f4', ('eas', ))
        eas.standard_name = 'easting'
        eas.units = 'metres_east'
        eas.axis = "X"
        eas[:] = X
    else:
        nc_struct = nc_struct = ('times', 'lat', 'lon')
        lat = dsout.createDimension('lat', Y.size)
        lat = dsout.createVariable('lat', 'f4', ('lat', ))
        lat.standard_name = 'latitude'
        lat.units = 'degrees_north'
        lat.axis = "Y"
        lat[:] = Y

        lon = dsout.createDimension('lon', X.size)
        lon = dsout.createVariable('lon', 'f4', ('lon', ))
        lon.standard_name = 'longitude'
        lon.units = 'degrees_east'
        lon.axis = "X"
        lon[:] = X

    time = dsout.createDimension('times', 0)
    times = dsout.createVariable('times', 'f4', ('times', ))
    times.standard_name = 'time'
    times.long_name = 'time'
    times.units = 'hours since 1970-01-01 00:00:00'
    times.calendar = 'standard'
    times[:] = date2num(date_created,
                        units=times.units,
                        calendar=times.calendar)

    if np.iscomplexobj(Z):
        U = dsout.createVariable('U',
                                 'f4',
                                 nc_struct,
                                 zlib=True,
                                 complevel=4,
                                 least_significant_digit=1,
                                 fill_value=-9999)
        U[:] = np.real(Z[np.newaxis, ...])
        U.standard_name = 'horizontal_velocity'
        U.units = 'm/s'
        U.setncattr('grid_mapping', 'spatial_ref')

        V = dsout.createVariable('V',
                                 'f4',
                                 nc_struct,
                                 zlib=True,
                                 complevel=4,
                                 least_significant_digit=1,
                                 fill_value=-9999)
        V[:] = np.imag(Z[np.newaxis, ...])
        V.standard_name = 'vertical_velocity'
        V.units = 'm/s'
        V.setncattr('grid_mapping', 'spatial_ref')
    else:
        Z = dsout.createVariable('I',
                                 'f4',
                                 nc_struct,
                                 zlib=True,
                                 complevel=4,
                                 least_significant_digit=1,
                                 fill_value=-9999)
        Z[:] = Z[np.newaxis(), ...]
        Z.setncattr('grid_mapping', 'spatial_ref')

    proj = dsout.createVariable('spatial_ref', 'i4')
    proj.spatial_ref = crs

    dsout.close()
    return


def make_im_from_geojson(geoTransform,
                         crs,
                         out_name,
                         geom_name,
                         out_dir=None,
                         geom_dir=None,
                         aoi=None):
    geoTransform = correct_geoTransform(geoTransform)
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()
    if out_dir is None:
        out_dir = os.getcwd()
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if geom_dir is None:
        rot_dir = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
        geom_dir = os.path.join(rot_dir, 'data')

    out_path = os.path.join(out_dir, out_name)
    geom_path = os.path.join(geom_dir, geom_name)
    assert os.path.isfile(geom_path), 'file does not seem to exist'

    # read shapefile
    shp = ogr.Open(geom_path)
    lyr = shp.GetLayer()

    # create raster environment
    driver = gdal.GetDriverByName('GTiff')
    if aoi is not None:
        target = driver.Create(out_path,
                               geoTransform[6],
                               geoTransform[7],
                               bands=1,
                               eType=gdal.GDT_UInt16)
    else:
        target = driver.Create(out_path,
                               geoTransform[6],
                               geoTransform[7],
                               bands=1,
                               eType=gdal.GDT_Byte)

    target.SetGeoTransform(geoTransform[:6])
    target.SetProjection(crs)

    # get attributes
    if aoi is not None:
        assert isinstance(aoi, str), 'please provide a string'
        gdal.RasterizeLayer(target, [1],
                            lyr,
                            None,
                            options=["ATTRIBUTE=" + aoi])
    # convert polygon to raster
    gdal.RasterizeLayer(target, [1], lyr, None, burn_values=[1])
    lyr = shp = None
    return
