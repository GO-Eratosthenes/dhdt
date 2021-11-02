import os
import numpy as np

from osgeo import ogr, osr, gdal

def dh_txt2shp(dh_mat, DEM_1, DEM_2, shp_name, srs):
    '''
    dh matrix to shapefile with matches 

    '''    
    driver = ogr.GetDriverByName('ESRI Shapefile') # set up the shapefile driver
    
    if os.path.exists(shp_name):
         driver.DeleteDataSource(shp_name)
    ds = driver.CreateDataSource(shp_name) # create the data source
        
    # create the layer
    layer = ds.CreateLayer('photohypso_matches', srs, geom_type=ogr.wkbLineString)
    
    # Add the fields we're interested in
    layer.CreateField(ogr.FieldDefn('dh', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('h1', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('h2', ogr.OFTReal))
    
    # Process the text file and add the attributes and features to the shapefile
    for j in range(dh_mat.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField('dh', dh_mat[j,6])
        feature.SetField('h1', DEM_1[j])
        feature.SetField('h2', DEM_2[j])
          
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(dh_mat[j,0], dh_mat[j,1])
        line.AddPoint(dh_mat[j,2], dh_mat[j,3])
        feature.SetGeometry(line)
        
        layer.CreateFeature(feature) # Create the feature in the layer (shapefile)
        feature = None # Dereference the feature
    layer = None
    # Save and close the data source
    ds = None