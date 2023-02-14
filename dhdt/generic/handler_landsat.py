import os
import numpy as np
from osgeo import ogr, osr, gdal

import geopandas

def get_wrs_url(version=2):
    """ get the location where the geometric data of Landsats path-row
    footprints are situated

    References
    ----------
    .. [1] https://www.usgs.gov/media/files/landsat-wrs-2-descending-path-row-shapefile
    """
    wrs_url = 'https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/' + \
              'production/s3fs-public/atoms/files/' + \
              'WRS2_descending_' + str(version) + '.zip'
    return wrs_url

def get_bbox_from_path_row(path, row,
                           shp_dir='/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles',
                           shp_name='wrs2_descending.shp'):
    path,row = int(path), int(row)
    shp_path = os.path.join(shp_dir,shp_name)
    
    wrs2 = geopandas.read_file(shp_path)
    
    toi = wrs2[(wrs2['PATH']==path) & (wrs2['ROW']==row)]
    return toi

def meta_LSstring(LSstr):  # generic
    """
    get meta data of the Landsat file name
    input:   LSstr         string            filename of the L1TP data
    output:  LStime        string            date "+YYYY-MM-DD"
             LSpath        string            satellite path "PXXX"
             LSrow         string            tile row "RXXX"
    """
    # e.g.: LANstr = 'LE07_L1TP_062018_20190829_20190924_01_T1_B8'
    
    LSsplit = LSstr.split('_')
    LStime = LSsplit[3]
    # convert to +YYYY-MM-DD string 
    # then it can be in the meta-data of following products
    LStime = '+' + LStime[0:4] +'-'+ LStime[4:6] +'-'+ LStime[6:8]
    LSpath = int(LSsplit[2][0:3])
    LSrow = int(LSsplit[2][3:7])
    return LStime, LSpath, LSrow

def get_LS_footprint(wrs_path, poi, roi):
    # Get the WRS footprints in lat-lon
    wrsShp = ogr.Open(wrs_path)
    wrsLayer = wrsShp.GetLayer()
    
    # loop through the regions and see if there is an intersection
    for i in range(0, wrsLayer.GetFeatureCount()):
        # Get the input Feature
        wrsFeature = wrsLayer.GetFeature(i)
        
        if (wrsFeature.GetField('Path')==poi) & \
            (wrsFeature.GetField('Row')==roi):
                geom = wrsFeature.GetGeometryRef()
                break
    for ring in geom:
        points = ring.GetPointCount()
        ll_footp = np.zeros((points,2), dtype=float)
        for p in range(points):
            (λ, ϕ, _) = ring.GetPoint(p)
            ll_footp[p,0], ll_footp[p,1] = λ, ϕ
            
    spatialRef = wrsLayer.GetSpatialRef()        
    return ll_footp, spatialRef




