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
def get_intersection(xy_1, xy_2, xy_3, xy_4):
    """ given two points per line, estimate the intersection

    Parameters
    ----------
    xy_1,xy_2 : numpy.array, size=(m,2), ndim={1,2}
        array with 2D coordinate of start and end from the first line.
    xy_3, xy_4 : numpy.array, size=(m,2), ndim={1,2}
        array with 2D coordinate of start and end from the second line.

    Returns
    -------
    xy : numpy.array, size=(m,2)
        array with 2D coordinate of intesection

    Notes
    -----
    The geometry can be seen like this:

    .. code-block:: text

                 + xy_1
                 |
                 |    + xy_4
                 |   /
                 |  /
                 | /
                 |/
                 x xy
                /|
               / |
              /  +xy_2
             /
            + xy_3
    """
    assert (isinstance(xy_1, (float, np.ndarray)) and
            isinstance(xy_2, (float, np.ndarray)))
    assert (isinstance(xy_3, (float, np.ndarray)) and
            isinstance(xy_4, (float, np.ndarray)))
    if type(xy_1) in (np.ndarray,):
        assert len(set({xy_1.shape[0], xy_2.shape[0],
                        xy_3.shape[0], xy_4.shape[0], })) == 1, \
            ('please provide arrays of the same size')

    x_1,x_2,x_3,x_4 = xy_1[...,0], xy_2[...,0], xy_3[...,0], xy_4[...,0]
    y_1,y_2,y_3,y_4 = xy_1[...,1], xy_2[...,1], xy_3[...,1], xy_4[...,1]

    numer_1, numer_2 = x_1*y_2 - y_1*x_2, x_3*y_4 - y_3*x_4
    dx_12, dy_12 = x_1 - x_2, y_1 - y_2
    dx_34, dy_34 = x_3 - x_4, y_3 - y_4
    denom = (dx_12*dy_34 - dy_12*dx_34)

    x = np.divide( numer_1 * dx_34 - dx_12 * numer_2, denom,
                   where=denom!=0, out=np.zeros_like(denom))
    y = np.divide( numer_1 * dy_34 - dy_12 * numer_2, denom,
                   where=denom!=0, out=np.zeros_like(denom))

    xy = np.column_stack([x,y])
    return xy