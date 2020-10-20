import os

from osgeo import ogr, osr, gdal

def ll2utm(ll_fname,utm_fname,crs,aoi='RGIId'):
    """
    Transfrom shapefile in Lat-Long to UTM coordinates
    
    input:   ll_fname       string            path and filename of input file
             utm_fname      string            path and filename of output file           
             crs            string            coordinate reference code
             aoi            string            atribute to include         
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

def shape2raster(shp_fname,im_fname,geoTransform,rows,cols,aoi='RGIId'):
    """
    Converts shapefile into raster
    
    input:   shp_fname      string            path and filename of shape file
             im_fname       string            path and filename of output file           
             geoTransform   array (1 x 6)     reference matrix
             rows           integer           number of rows in the image             
             cols           integer           number of collumns in the image             
             aoi            string            attribute of interest 
    """
    #making the shapefile as an object.
    rgiShp = ogr.Open(shp_fname)
    #getting layer information of shapefile.
    rgiLayer = rgiShp.GetLayer()
    #get required raster band.
    
    driver = gdal.GetDriverByName('GTiff')
    rgiRaster = driver.Create(im_fname+'.tif', rows, cols, 1, gdal.GDT_Int16)
    #            rgiRaster = gdal.GetDriverByName('GTiff').Create(dat_path+sat_tile+'.tif', mI, nI, 1, gdal.GDT_Int16)           
    rgiRaster.SetGeoTransform(geoTransform)
    band = rgiRaster.GetRasterBand(1)
    #assign no data value to empty cells.
    band.SetNoDataValue(0)
    # band.FlushCache()
    
    #main conversion method
    gdal.RasterizeLayer(rgiRaster, [1], rgiLayer, options=['ATTRIBUTE='+aoi])
    rgiRaster = None # missing link.....

def reproject_shapefile(path, in_file, targetprj):
    """
    Transforms shapefile into other projection
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