# functions to work with the coastal datasetimport osfrom osgeo import osr, ogr, gdal#import geopandasfrom ..generic.gis_tools import \    ll2utm, shape2raster, create_crs_from_utm_number, get_geom_for_utm_zonefrom ..generic.handler_sentinel2 import \    get_bbox_from_tile_code, get_geom_for_tile_codefrom ..generic.handler_landsat import \    get_bbox_from_path_rowdef get_coastal_polygon_of_tile(tile_code, utm_zone, tile_system='MGRS',                                aoi='land', shp_dir=None, shp_name=None):    # tile_code = '15NXA'    # utm_zone = '15N'    # wrs - landsat    # attribute of interest    if shp_dir is None:        shp_dir = '/Users/Alten005/surfdrive/Eratosthenes/Coastline'    if shp_name is None: shp_name='GSHHS_f_L6.shp'    if tile_system=='MGRS':        toi = get_geom_for_tile_code(tile_code) # tile of interest    else: # WRS2        #todo        path,row = [],[]        toi = get_bbox_from_path_row(path,row)    # get UTM extent, since not the whole world needs to be included    utm_poly = get_geom_for_utm_zone(utm_zone, d_buff=1)    shp_path = os.path.join(shp_dir,shp_name)    # open the shapefile of the coast    shp = ogr.Open(shp_path)    lyr = shp.GetLayer()    # transform to lat-lon to UTM    spatRef = lyr.GetSpatialRef()    driver = ogr.GetDriverByName('ESRI Shapefile')    proj = osr.SpatialReference()    proj.SetWellKnownGeogCS('WGS84')    proj.SetUTM(utm_zone, True)    coordTrans = osr.CoordinateTransformation(spatRef, proj)    # create the output layer    fname_shp_utm = shp_name[0:-4] + '_utm' + str(utm_zone).zfill(2) + '.shp'    if os.path.exists(shp_path + fname_shp_utm):        print('deleting already existing shapefile')        driver.DeleteDataSource(shp_path + fname_shp_utm)    outDataSet = driver.CreateDataSource(shp_path + fname_shp_utm)    outLyr = outDataSet.CreateLayer(aoi, geom_type=ogr.wkbMultiPolygon)    lyrDefn = ogr.FieldDefn('level', ogr.OFTInteger)    lyrDefn.SetWidth(2)    outLyr.CreateField(lyrDefn)    outLyrDefn = outLyr.GetLayerDefn()    feat = lyr.GetNextFeature()    while feat:        geom = feat.GetGeometryRef()  # get the input geometry        if geom.Intersects(utm_poly):            intersection = geom.Intersection(utm_poly)            # reproject the geometry from ll to UTM            intersection.Transform(coordTrans)            # create a new feature            outFeat = ogr.Feature(outLyrDefn)            # set the geometry and attribute            outFeat.SetGeometry(intersection)            outFeat.SetField('level', 8)            # add the feature to the shapefile            outLyr.CreateFeature(outFeat)            # dereference the features and get the next input feature            outFeat.Destroy()        feat = lyr.GetNextFeature()        # Save and close the shapefiles    outDataSet = outLyr = None    proj.MorphToESRI()  # provide correct prj-file for GIS    file = open(shp_path + fname_shp_utm[0:-4] + '.prj', 'w')    file.write(proj.ExportToWkt())    file.close()    del spatRef, proj, file    print('written ' + fname_shp_utm)    return#    gshhs = geopandas.read_file(shp_path)#    mask = gshhs.intersection(toi)#    gshhs_sub = gshhs[mask]       