import os
import numpy as np

from scipy import ndimage

from osgeo import ogr, osr, gdal


def create_crs_from_utm_zone(utm_code):
    """ generate GDAL SpatialReference from UTM code


    Parameters
    ----------
    utm_code : string
        Universal Transverse Mercator code of projection (e.g: '06N')

    Returns
    -------
    crs : osr.SpatialReference
        GDAL SpatialReference

    """
    utm_zone = int(utm_code[:-1])
    utm_sphere = utm_code[-1].upper()

    crs = osr.SpatialReference()
    crs.SetWellKnownGeogCS("WGS84")
    if utm_sphere != 'N':
        crs.SetProjCS("UTM " + str(utm_zone) +
                      " (WGS84) in northern hemisphere.")
        crs.SetUTM(utm_zone, True)
    else:
        crs.SetProjCS("UTM " + str(utm_zone) +
                      " (WGS84) in southern hemisphere.")
        crs.SetUTM(utm_zone, False)
    return crs


def ll2utm(ll_fname, utm_fname, crs, aoi='RGIId'):
    """ transfrom shapefile in Lat-Long to UTM coordinates

    Parameters
    ----------
    ll_fname : string
        path and filename of input file
    utm_fname : string
        path and filename of output file
    crs : string
        coordinate reference code
    aoi : string
        atribute to include. The default is 'RGIId'
    """
    # making the shapefile as an object.
    inputShp = ogr.Open(ll_fname)
    # getting layer information of shapefile.
    inLayer = inputShp.GetLayer()
    # get info for coordinate transformation
    inSpatialRef = inLayer.GetSpatialRef()
    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromWkt(crs)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    # create the output layer
    if os.path.exists(utm_fname):
        driver.DeleteDataSource(utm_fname)
    outDataSet = driver.CreateDataSource(utm_fname)
    outLayer = outDataSet.CreateLayer("reproject",
                                      outSpatialRef,
                                      geom_type=ogr.wkbMultiPolygon)
    # outLayer = outDataSet.CreateLayer(
    #     "reproject", geom_type=ogr.wkbMultiPolygon
    # )

    # add fields
    fieldDefn = ogr.FieldDefn('RGIId', ogr.OFTInteger)
    fieldDefn.SetWidth(14)
    fieldDefn.SetPrecision(1)
    outLayer.CreateField(fieldDefn)
    # inLayerDefn = inLayer.GetLayerDefn()
    # for i in range(0, inLayerDefn.GetFieldCount()):
    #     fieldDefn = inLayerDefn.GetFieldDefn(i)
    #     outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        rgiStr = inFeature.GetField(aoi)
        outFeature.SetField(aoi, int(rgiStr[9:]))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()
    outDataSet = None  # this creates an error but is needed?????


def shape2raster(shp_fname,
                 im_fname,
                 geoTransform,
                 rows,
                 cols,
                 spatialRef,
                 aoi='RGIId'):
    """ converts shapefile into raster

    Parameters
    ----------
    shp_fname : string
        path and filename of shape file
    im_fname : string
        path and filename of output file
    geoTransform : tuple, size=(1,6)
        affine transformation coefficients
    rows : integer, {x ∈ ℕ | x ≥ 0}
        number of rows in the image
    cols : integer, {x ∈ ℕ | x ≥ 0}
        number of collumns in the image
    aoi : string
        attribute of interest. The default is 'RGIId'
    """
    if not shp_fname[-4:] in (
            '.shp',
            '.SHP',
    ):
        shp_fname += '.shp'
    if not im_fname[-4:] in (
            '.tif',
            '.TIF',
    ):
        im_fname += '.tif'
    assert os.path.isfile(shp_fname), ('make sure the shapefiles exist')

    # making the shapefile as an object.
    rgiShp = ogr.Open(shp_fname)  # getting layer information of shapefile.
    rgiLayer = rgiShp.GetLayer()  # get required raster band.

    driver = gdal.GetDriverByName('GTiff')

    raster = driver.Create(im_fname, cols, rows, 1, gdal.GDT_UInt16)
    raster.SetGeoTransform(geoTransform[:6])
    raster.SetProjection(spatialRef)

    # main conversion method
    gdal.RasterizeLayer(raster, [1],
                        rgiLayer,
                        None,
                        options=['ATTRIBUTE=' + aoi])
    gdal.RasterizeLayer(raster, [1], rgiLayer, None, burn_values=[1])
    raster = None  # missing link.....
    return


def get_mask_boundary(Msk):
    inner = ndimage.binary_erosion(Msk)
    Bnd = np.logical_xor(Msk, inner)
    return Bnd


def reproject_shapefile(path, in_file, targetprj):
    """ transforms shapefile into other projection

    Parameters
    ----------
    path : string
        path of input file
    in_file : string
        filename of input file
    targetprj : osgeo-structure
        spatial reference

    Returns
    -------
    out_file : string
        filename of output file

    """
    out_file = in_file[:-4] + '_utm.shp'

    # getting layer information of shapefile.
    inShp = ogr.Open(os.path.join(path, in_file))
    inLayer = inShp.GetLayer()
    inSpatialRef = inLayer.GetSpatialRef()

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, targetprj)

    # create the output layer
    outputShapefile = os.path.join(path, out_file)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.isfile(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer(out_file, geom_type=ogr.wkbMultiPolygon)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(
                outLayerDefn.GetFieldDefn(i).GetNameRef(),
                inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()

    # Save and close the shapefiles
    inShp = None
    outDataSet = None

    return out_file


def get_utm_zone(ϕ, λ):
    """ get the UTM zone for a specific location

    Parameters
    ----------
    ϕ_lim, λ_lim : float, unit=degrees
        latitude and longitude of a point of interest

    Returns
    -------
    utm_zone : string
        string specifying the UTM zone
    """
    ϕ_zones = [
        chr(i) for i in list(range(67, 73)) + list(range(74, 79)) +
        list(range(80, 89))
    ]  # tile letters
    ϕ_cen = np.append(np.arange(-80, 72 + 1, 8), 84)
    λ_cen = np.arange(-180, 180 + 1, 6)

    ϕ_idx, λ_num = np.argmin(np.abs(ϕ_cen - ϕ)), np.argmin(np.abs(λ_cen - λ))
    λ_num += 1  # OBS: not a python index, but a numbering
    ϕ_idx, λ_num = np.minimum(ϕ_idx, 20), np.minimum(λ_num, 60)

    # in Southern Norway and Svalbard, the utM zones are merged
    if np.all((λ_num > 31, λ_num < 37, np.mod(λ_num, 2) != 1, ϕ_idx == 19)):
        # is location situated on Svalbard?
        if λ_num == 32:
            λ_num = 31 if ϕ < 9 else 33
        elif λ_num == 34:
            λ_num = 33 if ϕ < 21 else 35
        elif λ_num == 36:
            λ_num = 35 if ϕ < 33 else 37
    elif np.all((λ_num == 31, ϕ_idx == 17, λ > 3)):
        # is location situated in Southern Norway?
        λ_num = 32

    utm_zone = str(λ_num).zfill(2) + ϕ_zones[ϕ_idx]
    return utm_zone


def utm_zone_limits(utm_zone):
    """ get the extent of a UTM zone

    Parameters
    ----------
    utm_zone : string
        string specifying the UTM zone

    Returns
    -------
    ϕ_lim, λ_lim : numpy.ndarray, size=(2,)
        minimum and maximum extent
    """
    ϕ_zones = [
        chr(i) for i in list(range(67, 73)) + list(range(74, 79)) +
        list(range(80, 89))
    ]  # tile letters

    ϕ_min = np.append(np.arange(-80, 64 + 1, 8), 72)
    ϕ_max = np.append(np.arange(-80, 64 + 1, 8), 84)
    λ_min = np.arange(-180, 174 + 1, 6)
    λ_max = np.arange(-174, 180 + 1, 6)

    if utm_zone[-1].isalpha():
        ϕ_idx = [i for i, s in enumerate(ϕ_zones) if utm_zone[-1] in s][0]
        λ_idx = int(utm_zone[:-1])
    else:
        ϕ_idx = None
        λ_idx = int(utm_zone)

    λ_lim = np.array([λ_min[λ_idx], λ_max[λ_idx]])

    if ϕ_idx is None:
        ϕ_lim = np.nan * np.ones_like(λ_lim)
    else:
        ϕ_lim = np.array([ϕ_min[ϕ_idx], ϕ_max[ϕ_idx]])
    return ϕ_lim, λ_lim


def get_geom_for_utm_zone(utm_zone, d_buff=0):
    """ get the boundary of a UTM zone as a geometry

    Parameters
    ----------
    utm_zone : signed integer
        universal transverse Mercator projection zone
    d_buff : float, unit=degrees
        buffer around the zone. The default is 0.

    Returns
    -------
    utm_poly : ogr.geometry, dtype=polygon
        polygon of the UTM zone.
    """
    central_lon = np.arange(-177., 177., 6)

    lat_min = central_lon[utm_zone - 1] - 3 - d_buff
    lat_max = central_lon[utm_zone - 1] + 3 + d_buff

    # counter clockwise to describe exterior ring
    ring = ogr.Geometry(ogr.wkbLinearRing)

    ring.AddPoint(lat_min, +89.99)
    for i in np.flip(np.arange(-89.9, +89.9, 10.)):
        ring.AddPoint(lat_min, i)
    ring.AddPoint(lat_min, -89.99)

    ring.AddPoint(lat_max, -89.99)
    for i in np.arange(-89.9, +89.9, 10.):
        ring.AddPoint(lat_max, i)
    ring.AddPoint(lat_max, +89.99)
    ring.AddPoint(lat_min, +89.99)

    utm_poly = ogr.Geometry(ogr.wkbPolygon)
    utm_poly.AddGeometry(ring)

    # it seems it generates a 3D polygon, which is not needed
    utm_poly.FlattenTo2D()
    return utm_poly


def polylines2shapefile(strokes, spatialRef, path, out_file='strokes.shp'):
    """ create a shapefile with polylines, as they are provided within a list

    Parameters
    ----------
    strokes : list
        list with arrays that describe polylines
    spatialRef : string
        osr.SpatialReference in well known text
    path : string
        location where the shapefile can be placed
    out_file : string
        filename of the shapefile

    """

    def create_line(coords):
        line = ogr.Geometry(type=ogr.wkbLineString)
        for xy in coords:
            line.AddPoint_2D(xy[0], xy[1])
        return line

    DriverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(DriverName)

    # look at destination
    out_full = os.path.join(path, out_file)
    if os.path.isfile(out_full):
        driver.DeleteDataSource(out_full)

    # create files
    datasource = driver.CreateDataSource(out_full)
    srs = ogr.osr.SpatialReference(wkt=spatialRef)
    layer = datasource.CreateLayer('streamplot',
                                   srs,
                                   geom_type=ogr.wkbLineString)
    namedef = ogr.FieldDefn("ID", ogr.OFTString)
    namedef.SetWidth(32)
    layer.CreateField(namedef)

    # fill data
    counter = 1
    for stroke in strokes:
        feature = ogr.Feature(layer.GetLayerDefn())
        line = create_line(stroke)
        feature.SetGeometry(line)
        feature.SetField("ID", str(counter).zfill(4))
        layer.CreateFeature(feature)
        counter += 1

    datasource, layer = None, None  # close dataset to write to disk
    return
