# functions to work with the Randolph Glacier Inventory
import os
import tempfile
import numpy as np

import geopandas

from osgeo import ogr, osr, gdal

from ..generic.handler_www import url_exist, get_zip_file, get_tar_file
from ..generic.handler_sentinel2 import get_utmzone_from_tile_code, \
    get_generic_s2_raster, get_geom_for_tile_code, get_bbox_from_tile_code, \
    _check_mgrs_code

def rgi_code2url(rgi_num, version=7):
    """
    Given an RGI-number, give the location of its shapefiles on GLIMS-website

    Parameters
    ----------
    rgi_num : integer, range=1...19
        Randolph glacier inventory number

    Returns
    -------
    rgi_url : string
        location on a server where data is situated

    Notes
    -----
    The following nomenclature is used:

    RGI : Randolph glacier inventory
    UTM : universal transverse Mercator
    """


    if version==6:
        urls = (
            'rgi60_files/01_rgi60_Alaska.zip',
            'rgi60_files/02_rgi60_WesternCanadaUS.zip',
            'rgi60_files/03_rgi60_ArcticCanadaNorth.zip',
            'rgi60_files/04_rgi60_ArcticCanadaSouth.zip',
            'rgi60_files/05_rgi60_GreenlandPeriphery.zip',
            'rgi60_files/06_rgi60_Iceland.zip',
            'rgi60_files/07_rgi60_Svalbard.zip',
            'rgi60_files/08_rgi60_Scandinavia.zip',
            'rgi60_files/09_rgi60_RussianArctic.zip',
            'rgi60_files/10_rgi60_NorthAsia.zip',
            'rgi60_files/11_rgi60_CentralEurope.zip',
            'rgi60_files/12_rgi60_CaucasusMiddleEast.zip',
            'rgi60_files/13_rgi60_CentralAsia.zip',
            'rgi60_files/14_rgi60_SouthAsiaWest.zip',
            'rgi60_files/15_rgi60_SouthAsiaEast.zip',
            'rgi60_files/16_rgi60_LowLatitudes.zip',
            'rgi60_files/17_rgi60_SouthernAndes.zip',
            'rgi60_files/18_rgi60_NewZealand.zip',
            'rgi60_files/19_rgi60_AntarcticSubantarctic.zip'
            )
    else:
        urls = tuple('RGI'+str(n).zfill(2)+'.tar.gz' for n in range(19))
    rgi_url = ()
    for i in rgi_num:
        if isinstance(i, str): i = int(i)
        if version == 6:
            url_root = 'https://www.glims.org/RGI/'
        else:
            url_root = '/'.join(('https://cluster.klima.uni-bremen.de',
                                 '~fmaussion','misc','rgi7_data',
                                 'l3_rgi7a_tar'))
        rgi_url += ('/'.join((url_root, urls[i])),)
    return rgi_url

def which_rgi_region(poi, version=7, rgi_dir=None, rgi_file=None):
    """
    Discover which Randolph Glacier Inventory (RGI) region is used
    for a given coordinate, and if not present in directory, download this data

    Parameters
    ----------
    poi : {numpy.array, string}, unit=degrees
        location of interest, given in lat lon
        MGRS tile code
    version : {6,7}
        version of the glacier inventory
    rgi_dir : string
        location where the geospatial files of the RGI are situated
    rgi_file : string
        file name of the shapefile that describes the RGI regions

    Returns
    -------
    rgi_file : list
        list with strings of filenames
    rgi_dump : string
        location where the geometric files are situated

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - POI : point of interest
    """
    assert isinstance(version, int), 'please provide an integer'
    assert version in (6,7,), 'only version 6 and 7 are supported'
    if (version==7) and (rgi_dir is None):
        rot_dir = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
        rgi_dir = os.path.join(rot_dir, 'data')
    if (version==7) and (rgi_file is None):
        rgi_file = '00_rgi70_O1Regions.geojson'
    else:
        assert isinstance(rgi_file, str), 'please provide a string'
    if not os.path.isdir(rgi_dir): os.makedirs(rgi_dir)

    if isinstance(poi, str):
        poi = _check_mgrs_code(poi)
        toi = get_geom_for_tile_code(poi)

    rgi_full = os.path.join(rgi_dir, rgi_file)
    rgi_reg = geopandas.read_file(rgi_full)

    if isinstance(poi, np.ndarray):
        pnt = geopandas.GeoDataFrame(index=[0], crs='epsg:4326',
            geometry=geopandas.GeoSeries.from_xy([poi[0]],[poi[1]]))
    else:
        pnt = geopandas.GeoDataFrame(index=[0], crs='epsg:4326',
            geometry=geopandas.GeoSeries.from_wkt([toi]))

    roi = rgi_reg.sjoin(pnt)
    # spatial join
    rgi_list = roi['o1region'].values.tolist()

    if len(rgi_list)==0:
        print('selection not in Randolph glacier region')
        return

    rgi_url = rgi_code2url(rgi_list)
    # does RGI region exist?

    rgi_file = ()
    for i, rgi_reg in enumerate(rgi_url):
        rgi_zip = rgi_reg.split('/')[-1]
        rgi_fil = rgi_zip.split('.')[0]
        if version == 6:
            rgi_fil += '.shp'
        else:
            rgi_fil += '.geojson'

        dir_entries = os.scandir(rgi_dir)
        found_rgi = False
        for entry in dir_entries:
            if entry.name == rgi_fil:
                print(entry.name + ' is found in the folder')
                rgi_file += (rgi_fil, )
                found_rgi = True

        # appearantly the file is not present
        if not(url_exist(rgi_reg)):
            continue

        print('downloading RGI region')
        if version == 6:
            get_zip_file(rgi_reg, rgi_dir)
        else:
            shp_files = get_tar_file(rgi_reg, rgi_dir)
            shp_file = [f for f in shp_files if '.shp' in f]
            # transform shapefiles to geojson
            rgi_dat = geopandas.read_file(os.path.join(rgi_dir,
                                                       shp_file[0]))

            header = rgi_dat.keys()
            head_drop = [k for k in header if not k in ('glac_id','geometry',)]
            rgi_dat.drop(head_drop, axis=1, inplace=True)

            rgi_dat.to_file(os.path.join(rgi_dir, rgi_fil), driver='GeoJSON')

            # remove files/directory of shapefiles
            for f in list(reversed(shp_files)):
                if '.' in f:
                    os.remove(os.path.join(rgi_dir, f))
                else:
                    os.rmdir(os.path.join(rgi_dir, f))
            
    return rgi_file, rgi_dir

def create_projected_rgi_shp(rgi_path, rgi_name, utm_zone, northernHemisphere,
                             rgi_out):
    """ convert general RGI shapfile to projected shapefile

    Parameters
    ----------
    rgi_path : string
        directory where the shapefiles are located
    rgi_name : string
        name of the shapefile of the region of interest
    utm_zone : integer
        specific utm zone, to project towards
    northernHemisphere : bool
        is zone in the northern hemisphere
    rgi_out : string
        location where the shapefile should be positioned

    Notes
    -----
    The following nomenclature is used:

    - RGI : Randolph glacier inventory
    - UTM : universal transverse Mercator
    - GIS : geographic information system
    """

    # open the shapefile of RGI
    shp = ogr.Open(os.path.join(rgi_path, rgi_name))
    lyr = shp.GetLayer()

    # transform from lat-lon to UTM
    spatRef = lyr.GetSpatialRef()
    driver = ogr.GetDriverByName('ESRI Shapefile')

    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS('WGS84')
    proj.SetUTM(utm_zone, northernHemisphere)

    coordTrans = osr.CoordinateTransformation(spatRef, proj)

    # create the output layer
    if os.path.exists(os.path.join(rgi_path, rgi_out)):
        driver.DeleteDataSource(os.path.join(rgi_path, rgi_out))
    outDataSet = driver.CreateDataSource(os.path.join(rgi_path, rgi_out))
    outLyr = outDataSet.CreateLayer("", None, geom_type=ogr.wkbMultiPolygon)

    # add field with RGI-id
    outLyr.CreateField(ogr.FieldDefn("RGIId", ogr.OFTInteger))
    outLyr.CreateField(ogr.FieldDefn("RGIReg", ogr.OFTInteger))
    outLyr.CreateField(ogr.FieldDefn("RGIVer", ogr.OFTInteger))
    defn = outLyr.GetLayerDefn()

    rgi_reg = int(rgi_name.split('_')[0])
    rgi_ver = int(rgi_name.split('_')[1][-2:])
    for feat in lyr:
        geom = feat.GetGeometryRef()  # get the input geometry
        geom.Transform(coordTrans)    # reproject the geometry from ll to UTM
        rgi_num = int(feat.GetField('RGIId').split('.')[1])

        outFeat = ogr.Feature(defn)   # create a new feature
        outFeat.SetGeometry(geom)  # set the geometry and attribute
        outFeat.SetField('RGIId', rgi_num) # assign RGIid, but removing regionID
        outFeat.SetField('RGIReg', rgi_reg)
        outFeat.SetField('RGIVer', rgi_ver)

        outLyr.CreateFeature(outFeat)  # add the feature to the shapefile

        outFeat = None # dereference the features and get the next input feature
    outLyr = None
    outDataSet = None
    shp = None

    proj.MorphToESRI()  # provide correct prj-file for GIS
    file = open(os.path.join(rgi_path, rgi_out[:-4]+'.prj'), 'w')
    file.write(proj.ExportToWkt())
    file.close()
    return

def create_rgi_raster(rgi_path, rgi_name, toi, rgi_out=None):
    """
    Creates a raster file in the location given by "rgi_out", the content of
    this raster are the RGI glaciers

    Parameters
    ----------
    rgi_path : string
        directory where the shapefiles are located
    rgi_name : string
        name of the shapefile of the region of interest
    toi : string
        MGRS tile code of the Sentinel-2 scene
    rgi_out : string
        location where the shapefile should be positioned

    Returns
    -------
    rgi_fful : sting
        location and filename of the rater dataset

    Notes
    -----
    The following nomenclature is used:

    RGI : Randolph glacier inventory
    S2 : Sentinel-2
    TOI : tile of interest
    """
    assert os.path.isdir(rgi_path), 'make sure folder location exists'
    if isinstance(rgi_name, tuple): rgi_name = rgi_name[0]
    if rgi_out is None:
        rgi_out = rgi_path

    geoTransform,crs = get_generic_s2_raster(toi)
    crs.AutoIdentifyEPSG()
    s2_epsg = crs.GetAttrValue('AUTHORITY',1)

    # read shapefile
    shp = ogr.Open(os.path.join(rgi_path, rgi_name))
    lyr = shp.GetLayer()

    spatRef = lyr.GetSpatialRef()
    spatRef.AutoIdentifyEPSG()
    geom_epsg = spatRef.GetAttrValue('AUTHORITY',1)
    if not (s2_epsg is geom_epsg): # transform to same projection, if not the same

        src_new = ogr.osr.SpatialReference()
        src_new.ImportFromEPSG(int(s2_epsg))
        drv = shp.GetDriver()
        tmp_name = tempfile.NamedTemporaryFile(
            suffix='.'+rgi_name.split('.')[-1]).name
        ds = drv.CreateDataSource(tmp_name)
        lyr_new = ds.CreateLayer("glaciers", srs=src_new,
                                 geom_type=ogr.wkbMultiPolygon)
        lyr_new.CreateField(ogr.FieldDefn("RGIId", ogr.OFTInteger))

        coordTrans = osr.CoordinateTransformation(spatRef, crs)
        counter = 1
        for feat in lyr:
            geom = feat.GetGeometryRef()  # get the input geometry
            geom.Transform(coordTrans)    # reproject the geometry from ll to UTM
            geom.AssignSpatialReference(src_new)

            feat_new = ogr.Feature(lyr_new.GetLayerDefn())
                # create a new feature
            feat_new.SetGeometry(geom)  # set the geometry and attribute
            feat_new.SetField('RGIId', counter)
                # assign RGIid, but does not seem to be explicit in version 7

            lyr_new.CreateFeature(feat_new)  # add the feature to the geometry
            counter += 1
            feat_new = None  # dereference the features and get the next input feature

    # create raster environment
    rgi_ffull = os.path.join(rgi_out, toi + '-RGI.tif')
    driver = gdal.GetDriverByName('GTiff')
    target = driver.Create(rgi_ffull,
                           int(geoTransform[6]), int(geoTransform[7]), 1,
                           gdal.GDT_UInt16)

    target.SetGeoTransform(geoTransform[:6])

    # convert polygon to raster
    if not (s2_epsg is geom_epsg):
        target.SetProjection(src_new.ExportToWkt())
        gdal.RasterizeLayer(target, [1], lyr_new, options=["ATTRIBUTE=RGIId"],
                            burn_values=[1])
        lyr_new = ds = None
    else:
        target.SetProjection(crs.ExportToWkt())
        gdal.RasterizeLayer(target, [1], lyr, None, options=["ATTRIBUTE=RGIId"])
        gdal.RasterizeLayer(target, [1], lyr, None, burn_values=[1])
    lyr = shp = None
    return rgi_ffull

def create_rgi_tile_s2(toi, geom_path=None, geom_out=None):
    """
    Creates a raster file with the extent of a generic Sentinel-2 tile, that is
    situated at a specific location

    Parameters
    ----------
    toi : string
        MGRS tile code of the Sentinel-2 scene
    geom_path : string
        directory where the glacier geometry files are located
    geom_out : string
        location where the glacier geometry files should be positioned

    Returns
    -------
    geom_outpath : string
        location and filename of the glacier raster

    See Also
    --------
    which_rgi_region, create_rgi_raster

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - POI : point of interest
    - S2 : Sentinel-2
    - TOI : tile of interest
    """
    if geom_out is None: # place glacier data in generic data folder
        geom_out = geom_path

    bbox = get_bbox_from_tile_code(toi)
    poi = np.mean(bbox.reshape((2, 2)), axis=1)

    # search in which region the glacier is situated, and download its
    # geometric data, if this in not present on the local machine
    rgi_regions,rgi_dump = which_rgi_region(poi, rgi_dir=geom_path)

    if geom_path is None:
        geom_path = rgi_dump

    utm_zone = get_utmzone_from_tile_code(toi)
    northernHemisphere = True if np.sign(poi[1]) == 1 else False

    geom_outpath = create_rgi_raster(geom_path, rgi_regions, toi,
                                     rgi_out=geom_out)
    return geom_outpath

