import os

import numpy as np
from osgeo import gdal, ogr, osr


def dh_txt2shp(dh_mat, DEM_1, DEM_2, shp_name, srs):
    """
    dh matrix to shapefile with matches
    """

    if isinstance(srs, str):
        srs = osr.SpatialReference(wkt=srs)

    driver = ogr.GetDriverByName(
        'ESRI Shapefile')  # set up the shapefile driver

    if os.path.isfile(shp_name):
        driver.DeleteDataSource(shp_name)
    ds = driver.CreateDataSource(shp_name)  # create the data source

    # create the layer
    layer = ds.CreateLayer('photohypso_matches',
                           srs,
                           geom_type=ogr.wkbLineString)

    # Add the fields we're interested in
    layer.CreateField(ogr.FieldDefn('dh', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('h1', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('h2', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('score', ogr.OFTReal))

    # Process the text file and add attributes and features to the shapefile
    for j in range(dh_mat.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField('dh', dh_mat[j, -2])
        feature.SetField('score', dh_mat[j, -1])
        feature.SetField('h1', DEM_1[j])
        feature.SetField('h2', DEM_2[j])

        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(dh_mat[j, 0], dh_mat[j, 1])
        line.AddPoint(dh_mat[j, 2], dh_mat[j, 3])
        feature.SetGeometry(line)

        layer.CreateFeature(
            feature)  # Create the feature in the layer (shapefile)
        feature = None  # Dereference the feature
    layer = None
    # Save and close the data source
    ds = None
    return


def casting_pairs_mat2shp(xy_1, caster, xy_2, shp_name, srs):
    if isinstance(srs, str):
        srs = osr.SpatialReference(wkt=srs)

    driver = ogr.GetDriverByName(
        'ESRI Shapefile')  # set up the shapefile driver

    if os.path.isfile(shp_name):
        driver.DeleteDataSource(shp_name)
    ds = driver.CreateDataSource(shp_name)  # create the data source

    # create the layer
    layer = ds.CreateLayer('photohypso_pairs',
                           srs,
                           geom_type=ogr.wkbLineString)

    # Process the text file and add attributes and features to the shapefile
    for j in range(caster.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file

        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(xy_1[j, 0], xy_1[j, 1])
        line.AddPoint(caster[j, 0], caster[j, 1])
        line.AddPoint(xy_2[j, 0], xy_2[j, 1])
        feature.SetGeometry(line)

        layer.CreateFeature(
            feature)  # Create the feature in the layer (shapefile)
        feature = None  # Dereference the feature
    layer = None
    # Save and close the data source
    ds = None
    return


def casting_pairs_mat2shp(xy_1, xy_2, shp_name, srs):
    """ create shapefile of the refined matching

    Parameters
    ----------
    xy_1 : np.array, size=(m,2), unit=meters
        map location of the initial shadow edge
    xy_2 : np.array, size=(m,2), unit=meters
        map location of the refined shadow edge
    shp_name : str
        filename or filepath of the newly created shapefile
    srs : {str, osr.SpatialReference}
        spatial reference system in which the map coordinates are defined
    """
    if isinstance(srs, str):
        srs = osr.SpatialReference(wkt=srs)

    driver = ogr.GetDriverByName(
        'ESRI Shapefile')  # set up the shapefile driver

    if os.path.isfile(shp_name):
        driver.DeleteDataSource(shp_name)
    ds = driver.CreateDataSource(shp_name)  # create the data source

    # create the layer
    layer = ds.CreateLayer('shadow_edge', srs, geom_type=ogr.wkbLineString)

    # Process the text file and add attributes and features to the shapefile
    for j in range(xy_1.shape[0]):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file

        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(xy_1[j, 0], xy_1[j, 1])
        line.AddPoint(xy_2[j, 0], xy_2[j, 1])
        feature.SetGeometry(line)

        layer.CreateFeature(
            feature)  # Create the feature in the layer (shapefile)
        feature = None  # Dereference the feature
    layer = None
    # Save and close the data source
    ds = None
    return
