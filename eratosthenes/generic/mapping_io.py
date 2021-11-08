import os
import glob

import numpy as np

# geospatial libaries
from osgeo import gdal, osr

def read_geo_info(fname):  # generic
    """
    This function takes as input the geotiff name and the path of the
    folder that the images are stored, reads the geographic information of
    the image
    input:   band           string            geotiff name
             path           string            path of the folder
    output:  spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
             rows           integer           number of rows in the image             
             cols           integer           number of collumns in the image
             bands          integer           number of bands in the image             
    """
    img = gdal.Open(fname)
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    rows = img.RasterYSize    
    cols = img.RasterXSize
    bands = img.RasterCount
    return spatialRef, geoTransform, targetprj, rows, cols, bands

def read_geo_image(fname):
    """
        This function takes as input the geotiff name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   band           string            geotiff name
             path           string            path of the folder
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference



    Parameters
    ----------
    fname : string
        geotiff file name and path.

    Returns
    -------
    data : np.array, size=(m,n), ndim=2
        data array of the band
    spatialRef : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)

    See Also
    --------
    make_geo_im, read_geo_info

    Example
    -------
    >>> import os
    >>> fpath = os.path.join(os.getcwd(), ‘data.jp2')
    
    >>> (I, spatialRef, geoTransform, targetPrj) = read_geo_image(fpath)
    
    >>> I_ones = np.zeros(I.shape, dtype=bool)
    >>> make_geo_im(I_ones, geoTransformM, spatialRefM, ‘ones.tif’)
    assert os.path.exists(fname), ('file must exist')

    """
    assert len(glob.glob(fname)) != 0, ('file does not seem to be present')

    img = gdal.Open(fname)
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

# I/O functions
def make_geo_im(I, R, crs, fName, meta_descr='project Eratosthenes', \
              no_dat=np.nan, sun_angles='az:360-zn:90', date_created='-0276-00-00'):
    """
    Create georeference GeoTIFF
    input:   I              array (n x m)     band image
             R              array (1 x 6)     GDAL georeference transform of
                                              an image
             crs            string            coordinate reference string
             fname          string            filename for the image
    output:  writes file into current folder

    Example
    -------
    
    >>> import os
    >>> fpath = os.path.join(os.getcwd(), ‘data.jp2')
    
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
    elif I.dtype == 'bool':
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
    ds.SetGeoTransform(R)
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
