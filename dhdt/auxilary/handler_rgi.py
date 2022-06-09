# functions to work with the Randolph Glacier Inventory
import os
import numpy as np

from osgeo import ogr, osr, gdal

from dhdt.generic.unit_conversion import deg2arg
from dhdt.generic.handler_www import url_exist, get_zip_file
from dhdt.generic.handler_sentinel2 import get_utmzone_from_mgrs_tile, \
    get_generic_s2_raster

def rgi_code2url(rgi_num):
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
    rgi_url = ()
    for i in range(0,len(rgi_num)):
        rgi_url += ('https://www.glims.org/RGI/' + urls[rgi_num[i]-1],)
    return rgi_url

def which_rgi_region(rgi_path, poi, rgi_file='00_rgi60_O1Regions.shp'):
    """
    Discover which Randolph Glacier Inventory (RGI) region is used
    for a given coordinate, and if not present in directory, download this data

    Parameters
    ----------
    rgi_path : string
        location where the geospatial files of the RGI are situated
    poi : numpy.array, unit=degrees
        location of interest, given in lat lon
    rgi_file : string
        file name of the shapefile that describes the RGI regions

    Returns
    -------
    rgi_file : list
        list with strings of filenames

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - POI : point of interest
    """

    if np.ndim(poi)==1:
        # lon in range -180:180
        poi[1] = deg2arg(poi[1])
        poi = np.flip(poi) # given in lon lat....
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(poi[0], poi[1])

        # Get the RGI gegions
        rgi_full = os.path.join(rgi_path, rgi_file)
        assert os.path.exists(rgi_full), ('please make sure ' + rgi_file +
                                          ' is present')
        rgiShp = ogr.Open(rgi_full)
        rgiLayer = rgiShp.GetLayer()

        rgi_list = ()
        # loop through the regions and see if there is an intersection
        for i in range(0, rgiLayer.GetFeatureCount()):
            # Get the input Feature
            rgiFeature = rgiLayer.GetFeature(i)
            geom = rgiFeature.GetGeometryRef()
            
            if point.Within(geom):
                rgi_list += (rgiFeature.GetField('RGI_CODE'),)
                print('it seems the region you are interest in falls within '+
                      rgiFeature.GetField('FULL_NAME'))
                
    else: # polygon based
        print('not yet implemented!')

    
    if len(rgi_list)==0:
        print('selection not in Randolph glacier region')
    else:
        rgi_url = rgi_code2url(rgi_list)
        # does RGI region exist?
        
        rgi_file = ()
        for i in range(0, len(rgi_url)):
            rgi_reg = rgi_url[i]
            rgi_zip = rgi_reg.split('/')[-1]
            rgi_shp = rgi_zip[:-4] + '.shp'
            
            dir_entries = os.scandir(rgi_path)
            found_rgi = False
            for entry in dir_entries:
                if entry.name == rgi_shp:
                    print(entry.name + ' is found in the folder')
                    rgi_file += (rgi_shp, )
                    found_rgi = True
                
            # appearantly the file is not present
            if (url_exist(rgi_reg) and not(found_rgi)):
                print('downloading RGI region')
                get_zip_file(rgi_reg, rgi_path)
                rgi_file += (rgi_shp, )
            
    return rgi_file

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

    Notes
    -----
    The following nomenclature is used:

    RGI : Randolph glacier inventory
    S2 : Sentinel-2
    TOI : tile of interest
    """
    if rgi_out is None:
        rgi_out = rgi_path

    geoTransform,crs = get_generic_s2_raster(toi)

    # read shapefile
    shp = ogr.Open(os.path.join(rgi_path, rgi_name))
    lyr = shp.GetLayer()

    # create raster environment
    driver = gdal.GetDriverByName('GTiff')
    target = driver.Create(os.path.join(rgi_out, toi + '-RGI.tif'),
                           int(geoTransform[6]), int(geoTransform[7]), 1,
                           gdal.GDT_UInt16)

    target.SetGeoTransform(geoTransform[:6])
    target.SetProjection(crs.ExportToWkt())

    # convert polygon to raster
    gdal.RasterizeLayer(target, [1], lyr, None, options=["ATTRIBUTE=RGIId"])
    gdal.RasterizeLayer(target, [1], lyr, None, burn_values=[1])
    lyr = shp = None
    return

def create_rgi_tile_s2(rgi_path, poi, toi, rgi_out=None):
    """
    Creates a raster file with the extent of a generic Sentinel-2 tile, that is
    situated at a specific location

    Parameters
    ----------
    rgi_path : string
        directory where the RGI shapefiles are located
    poi : numpy.array, unit=degrees
        location of interest, given in lat lon
    toi : string
        MGRS tile code of the Sentinel-2 scene
    rgi_out : string
        location where the shapefile should be positioned

    Returns
    -------
    rgi_outpath : string
        location and filename of the RGI raster

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
    if rgi_out is None:
        rgi_out = rgi_path

    rgi_regions = which_rgi_region(rgi_path, poi)

    utm_zone = get_utmzone_from_mgrs_tile(toi)
    northernHemisphere = True if np.sign(poi[1]) == 1 else False


    for rgi_name in rgi_regions:
        fname_shp_utm = rgi_name[:-4] + '_utm' + str(utm_zone).zfill(2) + '.shp'
        if not os.path.exists(os.path.join(rgi_path, fname_shp_utm)):
        # project shapefile RGI to map projection
            create_projected_rgi_shp(rgi_path, rgi_name,
                                     utm_zone, northernHemisphere,
                                     fname_shp_utm)

    #todo: merge shapefiles of different regions, if neccessary

    rgi_out_full = os.path.join(rgi_path, toi+'-RGI.tif')
    if not os.path.exists(rgi_out_full):
        create_rgi_raster(rgi_path, fname_shp_utm, toi, rgi_out)

    return rgi_out_full

