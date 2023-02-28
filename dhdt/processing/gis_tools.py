import os
import numpy as np

from osgeo import ogr, osr, gdal

from ..generic.mapping_io import read_geo_info

def make_casting_couple_shapefile(fpath):
    '''
    make shapefile, to be used in a GIS for analysis
    
    S_1(x) S_1(y) S_2(x) S_2(y) C(x) C(y)
    
    '''

    fdir, fname = os.path.split(fpath)
    oname = fname[:-3]+'shp'
    
    # get metadata
    f = open(fpath, 'r')
    crs = f.readlines(1)[0][:-1]
    f.close()
    
    xy_list = np.loadtxt(fname = fpath, skiprows=1)
    
    # set up the shapefile driver   
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # create the spatial reference, WGS84
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromWkt(crs) # hardcoded
    # create the output layer
    dataSet = driver.CreateDataSource(os.path.join(fdir, oname))
    # create the layer
    layer = dataSet.CreateLayer("castlines", spatialRef, \
                                geom_type=ogr.wkbLineString)
    # add field
    layer.CreateField(ogr.FieldDefn("dh", ogr.OFTReal))

    # process the text file and add the attributes and features to the shapefile
    for p in range(xy_list.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # set the attributes using the values from the delimited text file
        feature.SetField("dh", xy_list[p,6])
      
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(xy_list[p,0], xy_list[p,1]) # S_1
        line.AddPoint(xy_list[p,4], xy_list[p,5]) # C
        line.AddPoint(xy_list[p,2], xy_list[p,3]) # S_2      
        # set the feature geometry using the line
        feature.SetGeometry(line)
        # create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # dereference the feature
        feature = None
    
    # save and close the data source
    dataSet = None

def make_conn_txt_a_shapefile(fpath):
    '''
    make shapefile, to be used in a GIS for analysis
    
    C(x) C(y) S(x) S(y) azi(x) elev(y)
    
    '''

    fdir, fname = os.path.split(fpath)
    oname = fname[:-3]+'shp' # output name
    
    # get metadata
    crs,_,_,_,_,_ = read_geo_info(os.path.join(fdir, 'shadow.tif'))
  
    xy_list = np.loadtxt(fname = fpath) #, skiprows=1)
    
    # set up the shapefile driver   
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # create the spatial reference, WGS84
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromWkt(crs) # hardcoded
    # create the output layer
    dataSet = driver.CreateDataSource(os.path.join(fdir, oname))
    # create the layer
    layer = dataSet.CreateLayer("castlines", spatialRef, \
                                geom_type=ogr.wkbLineString)
    # add field
    layer.CreateField(ogr.FieldDefn("sunAz", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("sunZn", ogr.OFTReal))
    
    # process the text file and add the attributes and features to the shapefile
    for p in range(xy_list.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # set the attributes using the values from the delimited text file
        feature.SetField("sunAz", xy_list[p,4])        
        feature.SetField("sunZn", xy_list[p,5])
      
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(xy_list[p,0], xy_list[p,1]) # C
        line.AddPoint(xy_list[p,2], xy_list[p,3]) # S      
        # set the feature geometry using the line
        feature.SetGeometry(line)
        # create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # dereference the feature
        feature = None
    
    # save and close the data source
    dataSet = None    

# def find_dh_from_DEM():
    