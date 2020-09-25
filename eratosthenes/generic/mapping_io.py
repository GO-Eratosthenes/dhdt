import numpy as np
import pathlib

from osgeo import gdal, osr


def read_geo_info(fname):  # generic
    """
    This function takes as input the name of the raster file where the image is
    stored and reads the geographic information of the image
    input:   path           string or Path    path of the folder
    output:  spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
             rows           integer           number of rows in the image
             cols           integer           number of collumns in the image
             bands          integer           number of bands in the image
    """
    fname = pathlib.Path(fname).as_posix()
    img = gdal.Open(fname)
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    rows = img.RasterYSize
    cols = img.RasterXSize
    bands = img.RasterCount
    return spatialRef, geoTransform, targetprj, rows, cols, bands


def read_geo_image(fname):  # generic
    """
    This function takes as input the name of the raster file where the image is
    stored, reads the image and returns the data as an array
    input:   path           string or Path    path of the file
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
    """
    fname = pathlib.Path(fname).as_posix()
    img = gdal.Open(fname)
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj


# I/O functions
def makeGeoIm(data, geoTransform, crs, fName):
    """
    Create georeference GeoTIFF
    input:   data           array (n x m)     band image
             geoTransform   array (1 x 6)     georeference transform of
                                              an image
             crs            string            coordinate reference string
             fname          string            filename for the image
    output:  writes file into current folder
    """
    fName = pathlib.Path(fName).as_posix()
    ncols, nrows = data.shape
    drv = gdal.GetDriverByName("GTiff")  # export image
    ftype = gdal.GDT_Float64 if data.dtype == 'float64' else gdal.GDT_Int32
    ds = drv.Create(fName, ncols, nrows, 6, ftype, ["COMPRESS=LZW"])
    ds.SetGeoTransform(geoTransform)
    ds.SetProjection(crs)
    ds.GetRasterBand(1).WriteArray(data)
