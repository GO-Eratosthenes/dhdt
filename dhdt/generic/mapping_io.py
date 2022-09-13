import os
import glob

import numpy as np
import pandas as pd

# geospatial libaries
from osgeo import gdal, osr
from xml.etree import ElementTree

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
        affine transformation coefficients, but also giving the image dimensions
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    rows : integer
        number of rows in the image, that is its height
    cols : integer
        number of collumns in the image, that is its width
    bands : integer
        number of bands in the image, that is its depth

    See Also
    --------
    read_geo_image : basic function to import geographic imagery data
    """
    assert len(glob.glob(fname)) != 0, ('file does not seem to be present')

    img = gdal.Open(fname)
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    rows = img.RasterYSize    
    cols = img.RasterXSize
    bands = img.RasterCount

    geoTransform += (rows, cols,)
    return spatialRef, geoTransform, targetprj, rows, cols, bands

def read_geo_image(fname, boi=np.array([])):
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

    Returns
    -------
    data : {numpy.array, numpy.masked.array}, size=(m,n), ndim=2
        data array of the band
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)

    See Also
    --------
    make_geo_im : basic function to write out geographic data
    read_geo_info : basic function to get meta data of geographic imagery
    read_nc_im : basic function to read geographic raster out of nc-file

    Example
    -------
    >>> import os
    >>> from dhdt.generic.mapping_io import read_geo_image
    >>> fpath = os.path.join(os.getcwd(), "data.jp2" )
    
    >>> I, spatialRef, geoTransform, targetPrj = read_geo_image(fpath)
    
    >>> I_ones = np.zeros(I.shape, dtype=bool)
    >>> make_geo_im(I_ones, geoTransformM, spatialRefM, "ones.tif")
    assert os.path.exists(fname), ('file must exist')

    """
    assert os.path.exists(fname), ('file does not seem to be present')

    img = gdal.Open(fname)
    assert img is not None, ('could not open dataset ' + fname)

    # imagery can consist of multiple bands
    if len(boi) == 0:
        for counter in range(img.RasterCount):
            band = np.array(img.GetRasterBand(counter+1).ReadAsArray())
            no_dat = img.GetRasterBand(counter+1).GetNoDataValue()
            # create masked array
            if no_dat is not None:
                band = np.ma.array(band, mask=band==no_dat)
            data = band if counter == 0 else np.dstack((data,
                                                        band[:,:,np.newaxis]))
    else:
        num_bands = img.RasterCount
        assert (np.max(boi)+1)<=num_bands, 'bands of interest is out of range'
        for band_id, counter in enumerate(boi):
            band = np.array(img.GetRasterBand(band_id+1).ReadAsArray())
            no_dat = img.GetRasterBand(counter+1).GetNoDataValue()
            np.putmask(band, band==no_dat, np.nan)
            data = band if counter==0 else np.dstack((data,
                                                      band[:, :, np.newaxis]))
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform() + data.shape[:2]
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_nc_image(fname, layer_name):
    assert os.path.exists(fname), ('file does not seem to be present')

    ds = gdal.Open("NETCDF:{0}:{1}".format(fname, layer_name))
    assert ds is not None, ('could not open dataset ' + fname)

    data = np.array(ds.ReadAsArray())
    np.putmask(data, data==ds.GetRasterBand(1).GetNoDataValue(), np.nan)

    spatialRef = ds.GetProjection()
    geoTransform = ds.GetGeoTransform() + data.shape
    targetprj = osr.SpatialReference(wkt=ds.GetProjection())
    return data, spatialRef, geoTransform, targetprj

# output functions
def make_geo_im(I, R, crs, fName, meta_descr='project Eratosthenes',
              no_dat=np.nan, sun_angles='az:360-zn:90', date_created='-0276-00-00'):
    """ Create georeferenced tiff file (a GeoTIFF)

    Parameters
    ----------
    I : numpy.array, size=(m,n)
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
        string given the acquistion date in YYYY-MM-DD

    Example
    -------
    
    >>> import os
    >>> fpath = os.path.join(os.getcwd(), "data.jp2")
    
    >>> (I, spatialRef, geoTransform, targetPrj) = read_geo_image(fpath)
    
    >>> I_ones = np.zeros(I.shape, dtype=bool)
    >>> make_geo_im(I_ones, geoTransformM, spatialRefM, ‘ones.tif’)
    """
    drv = gdal.GetDriverByName("GTiff")  # export image
    
    if I.ndim == 3:
        bands=I.shape[2]
    else:
        bands = 1

    # make it type dependent
    if I.dtype == 'float64':
        ds = drv.Create(fName,xsize=I.shape[1], ysize=I.shape[0],bands=bands,
                        eType=gdal.GDT_Float64)
    elif I.dtype == 'float32':
        ds = drv.Create(fName,xsize=I.shape[1], ysize=I.shape[0],bands=bands,
                        eType=gdal.GDT_Float32)
    elif I.dtype == 'bool':
        ds = drv.Create(fName, xsize=I.shape[1], ysize=I.shape[0], bands=bands,
                        eType=gdal.GDT_Byte)
    elif I.dtype == 'uint8':
        ds = drv.Create(fName, xsize=I.shape[1], ysize=I.shape[0], bands=bands,
                        eType=gdal.GDT_Byte)
    else:
        ds = drv.Create(fName, xsize=I.shape[1], ysize=I.shape[0], bands=bands,
                        eType=gdal.GDT_Int32)

    # set metadata in datasource
    ds.SetMetadata({'TIFFTAG_SOFTWARE':'dhdt v0.1',
                    'TIFFTAG_ARTIST':'bas altena and team Atlas',
                    'TIFFTAG_COPYRIGHT': 'contains modified Copernicus data',
                    'TIFFTAG_IMAGEDESCRIPTION': meta_descr,
                    'TIFFTAG_RESOLUTIONUNIT' : sun_angles,
                    'TIFFTAG_DATETIME': date_created})
    
    # set georeferencing metadata
    if len(R)!=6: R = R[:6]

    ds.SetGeoTransform(R)
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()
    ds.SetProjection(crs)
    if I.ndim == 3:
        for count in np.arange(1,I.shape[2]+1,1):
            band = ds.GetRasterBand(int(count))
            band.WriteArray(I[:,:,count-1],0,0)
            if count==1:
                band.SetNoDataValue(no_dat)
            band = None
    else:
        ds.GetRasterBand(1).WriteArray(I)
        ds.GetRasterBand(1).SetNoDataValue(no_dat)
    ds = None
    del ds

def make_multispectral_vrt(df, fpath=None, fname='multispec.vrt'):
    """ virtual raster tile (VRT) is a description of datasets written in an XML
    format, it eases the display of multi-spectral data or other means.

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


    """
    assert isinstance(df, pd.DataFrame), ('please provide a dataframe')
    assert 'filepath' in df, ('please first run "get_S2_image_locations"'+
                                ' to find the proper file locations')

    if fpath is None:
        fpath = os.path.commonpath(df.filepath.tolist())

    ffull = os.path.join(fpath, fname)
    vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour,
                                       addAlpha=False,
                                       separate=True,
                                       srcNodata=0)
    my_vrt = gdal.BuildVRT(ffull, [f+'.jp2' for f in df['filepath']],
                           options=vrt_options)
    my_vrt = None

    # modify the vrt-file to include band names
    tree = ElementTree.parse(ffull)
    root = tree.getroot()
    for idx, band in enumerate(root.iter("VRTRasterBand")):
        description = ElementTree.SubElement(band, "Description")
        description.text = df.common_name[idx]
    tree.write(ffull) # update the file on disk
    return
