import os
import numpy as np

from scipy import ndimage

from osgeo import ogr, osr, gdal

def create_crs_from_utm_number(utm_code):
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
        crs.SetProjCS("UTM " + str(utm_zone) + \
                      " (WGS84) in northern hemisphere.")
        crs.SetUTM(utm_zone,True)
    else:
        crs.SetProjCS("UTM " + str(utm_zone) + \
              " (WGS84) in southern hemisphere.")
        crs.SetUTM(utm_zone,False)
    return crs

def ll2utm(ll_fname,utm_fname,crs,aoi='RGIId'):
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
    #making the shapefile as an object.
    inputShp = ogr.Open(ll_fname)
    #getting layer information of shapefile.
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
    outLayer = outDataSet.CreateLayer("reproject", outSpatialRef, geom_type=ogr.wkbMultiPolygon)
 #               outLayer = outDataSet.CreateLayer("reproject", geom_type=ogr.wkbMultiPolygon)
    
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
    outDataSet = None # this creates an error but is needed?????

def shape2raster(shp_fname,im_fname,geoTransform,rows,cols,spatialRef,
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
    rows : integer
        number of rows in the image 
    cols : integer
        number of collumns in the image
    aoi : string
        attribute of interest. The default is 'RGIId'
    """
    if not shp_fname[-4:] in ('.shp', '.SHP',):
        shp_fname += '.shp'
    if not im_fname[-4:] in ('.tif', '.TIF',):
        im_fname += '.tif'
    assert os.path.exists(shp_fname), ('make sure the shapefiles exist')

    #making the shapefile as an object.
    rgiShp = ogr.Open(shp_fname)    #getting layer information of shapefile.
    rgiLayer = rgiShp.GetLayer()    #get required raster band.
    
    driver = gdal.GetDriverByName('GTiff')

    raster = driver.Create(im_fname, cols, rows, 1, gdal.GDT_UInt16)
    raster.SetGeoTransform(geoTransform[:6])
    raster.SetProjection(spatialRef)
    
    #main conversion method
    gdal.RasterizeLayer(raster, [1], rgiLayer, None, options=['ATTRIBUTE='+aoi])
    gdal.RasterizeLayer(raster, [1], rgiLayer, None, burn_values=[1])
    raster = None # missing link.....
    return

def get_mask_boundary(Msk):
    inner = ndimage.morphology.binary_erosion(Msk)
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
    out_file = in_file[:-4]+'_utm.shp'
    
    #getting layer information of shapefile.
    inShp = ogr.Open(os.path.join(path, in_file))
    inLayer = inShp.GetLayer()
    inSpatialRef = inLayer.GetSpatialRef()
   
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, targetprj)
       
    # create the output layer
    outputShapefile = os.path.join(path, out_file)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputShapefile):
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
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()
    
    # Save and close the shapefiles
    inShp = None
    outDataSet = None
    
    return out_file

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
    # central_lon = np.arange(-177., 177., 6)
    
    lat_min = central_lon[utm_zone-1]-3-d_buff
    lat_max = central_lon[utm_zone-1]+3+d_buff   
    
    # counter clockwise to describe exterior ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    
    ring.AddPoint(lat_min, +89.99)    
    for i in np.flip(np.arange( -89.9, +89.9, 10.)):
        ring.AddPoint(lat_min, i)    
    ring.AddPoint(lat_min, -89.99)
    
    ring.AddPoint(lat_max, -89.99)
    for i in np.arange( -89.9, +89.9, 10.):
        ring.AddPoint(lat_max, i)        
    ring.AddPoint(lat_max, +89.99)
    ring.AddPoint(lat_min, +89.99)

    utm_poly = ogr.Geometry(ogr.wkbPolygon)
    utm_poly.AddGeometry(ring)
    
    # it seems it generates a 3D polygon, which is not needed
    utm_poly.FlattenTo2D()
    return utm_poly

def polylines2shapefile(strokes,spatialRef, path, out_file='strokes.shp'):
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
    if os.path.exists(out_full):
        driver.DeleteDataSource(out_full)

    # create files
    datasource = driver.CreateDataSource(out_full)
    srs = ogr.osr.SpatialReference(wkt=spatialRef)
    layer = datasource.CreateLayer('streamplot', srs,
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

    datasource, layer = None, None # close dataset to write to disk
    return