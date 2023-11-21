import os

import geopandas
import numpy as np
from osgeo import gdal, ogr, osr

from dhdt.generic.handler_www import get_zip_file


def get_wrs_url(version=2):
    """ get the location where the geometric data of Landsats path-row
    footprints are situated, for info see [wwwUSGS]_

    References
    ----------
    .. [wwwUSGS] https://www.usgs.gov/media/files/landsat-wrs-2-descending-path-row-shapefile
    """  # noqa: E501
    assert isinstance(version, int), 'please provide an integer'
    assert 0 < version < 3, 'please provide a correct version, i.e.: {1,2}'
    wrs_url = (
        'https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS'  # noqa: E501
        + str(version) + '_descending_0.zip')
    return wrs_url


def get_wrs_dataset(geom_dir, geom_name=None, version=2):
    if geom_name is None:
        geom_name = 'wrs' + str(version) + '.geojson'
    ffull = os.path.join(geom_dir, geom_name)
    if os.path.isfile(ffull):
        return ffull

    wrs_url = get_wrs_url(version=version)
    file_list = get_zip_file(wrs_url, dump_dir=geom_dir)

    soi = 'WRS' + str(version) + '_descending.shp'
    sfull = os.path.join(geom_dir, soi)  # full path of shapefile of interest

    wrs = geopandas.read_file(sfull)

    header = wrs.keys()
    head_drop = [k for k in header if k not in (
        'PATH',
        'ROW',
        'geometry',
    )]
    wrs.drop(head_drop, axis=1, inplace=True)

    # write out the geometry to a GeoJSON file
    wrs.to_file(ffull, driver='GeoJSON')

    for f in file_list:
        os.remove(os.path.join(geom_dir, f))
    return ffull


def get_bbox_from_path_row(
        path,
        row,
        shp_dir='/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles',
        shp_name='wrs2_descending.shp'):
    path, row = int(path), int(row)
    shp_path = os.path.join(shp_dir, shp_name)

    wrs2 = geopandas.read_file(shp_path)

    toi = wrs2[(wrs2['PATH'] == path) & (wrs2['ROW'] == row)]
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
    LStime = '+' + LStime[0:4] + '-' + LStime[4:6] + '-' + LStime[6:8]
    LSpath = int(LSsplit[2][0:3])
    LSrow = int(LSsplit[2][3:7])
    return LStime, LSpath, LSrow


def get_ls_footprint(wrs_path, poi, roi):
    # Get the WRS footprints in lat-lon
    wrsShp = ogr.Open(wrs_path)
    wrsLayer = wrsShp.GetLayer()

    # loop through the regions and see if there is an intersection
    for i in range(0, wrsLayer.GetFeatureCount()):
        # Get the input Feature
        wrsFeature = wrsLayer.GetFeature(i)

        if (wrsFeature.GetField('Path') == poi) & \
                (wrsFeature.GetField('Row') == roi):
            geom = wrsFeature.GetGeometryRef()
            break
    for ring in geom:
        points = ring.GetPointCount()
        ll_footp = np.zeros((points, 2), dtype=float)
        for p in range(points):
            (λ, ϕ, _) = ring.GetPoint(p)
            ll_footp[p, 0], ll_footp[p, 1] = λ, ϕ

    spatialRef = wrsLayer.GetSpatialRef()
    return ll_footp, spatialRef
