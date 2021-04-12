import os
import osr
import ogr
import gdal

from eratosthenes.generic.handler_s2 import get_S2_folders, get_tiles_from_S2_list, get_utm_from_S2_tiles
from eratosthenes.generic.gis_tools import get_geom_for_utm_zone

# source directories
src_path = '/Users/Alten005/surfdrive/Eratosthenes/Coastline/'
cst_name = 'GSHHS_f_L1.shp'
im_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'

# processing directory
prc_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Processing/'

# make folder with output, if not existing
if not os.path.exists(prc_path):
    os.makedirs(prc_path)

# get tiles
S2_list = get_S2_folders(im_path)
tile_list = get_tiles_from_S2_list(S2_list)
# find out how many utmzones are included
utm_list = get_utm_from_S2_tiles(tile_list)


for i in range(len(utm_list)):
    # get UTM extent, since not the whole world needs to be included
    utm_poly = get_geom_for_utm_zone(utm_list[i], d_buff=1)
    #utm_poly = utm_poly.GetGeometryRef(0)
    
    # open the shapefile of RGI
    shp = ogr.Open(src_path+cst_name)
    lyr = shp.GetLayer()
    
    # transform to lat-lon to UTM
    spatRef = lyr.GetSpatialRef()
    driver = ogr.GetDriverByName('ESRI Shapefile')

    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS('WGS84')
    
    proj.SetUTM(utm_list[i], True)
    
    coordTrans = osr.CoordinateTransformation(spatRef, proj)
    
    # create the output layer
    fname_shp_utm = cst_name[0:-4] + '_utm' + str(utm_list[i]).zfill(2) + '.shp'
    if os.path.exists(prc_path+fname_shp_utm):
        driver.DeleteDataSource(prc_path+fname_shp_utm)
    outDataSet = driver.CreateDataSource(prc_path+fname_shp_utm)
    
    outLyr = outDataSet.CreateLayer('land', geom_type=ogr.wkbMultiPolygon)
    
    lyrDefn = ogr.FieldDefn('level', ogr.OFTInteger)
    lyrDefn.SetWidth(2)
    outLyr.CreateField(lyrDefn)
    
    outLyrDefn = outLyr.GetLayerDefn()
    
    feat = lyr.GetNextFeature()
    while feat:     
        geom = feat.GetGeometryRef() # get the input geometry     

        if geom.Intersects(utm_poly):    
            intersection = geom.Intersection(utm_poly)
            # reproject the geometry from ll to UTM
            intersection.Transform(coordTrans) 
            # create a new feature           
            outFeat = ogr.Feature(outLyrDefn)
            # set the geometry and attribute
            outFeat.SetGeometry(intersection) 
            outFeat.SetField('level', 8)
            # add the feature to the shapefile
            outLyr.CreateFeature(outFeat) 
            # dereference the features and get the next input feature
            outFeat.Destroy()
    
        feat = lyr.GetNextFeature()   
    
    # Save and close the shapefiles
    outDataSet = outLyr = None    
    proj.MorphToESRI() # provide correct prj-file for GIS
    file = open(prc_path+fname_shp_utm[0:-4]+'.prj', 'w')
    file.write(proj.ExportToWkt())
    file.close()
    del spatRef, proj, file
    
    print('written ' + fname_shp_utm)
    del fname_shp_utm, outDataSet, outLyr, outLyrDefn, feat, geom, outFeat, lyr, shp, driver

# loop through the tiles
for i in range(len(tile_list)):
#    tileName = 'T05VNK';
    tile_name = tile_list[i]
    if not os.path.exists(prc_path + tile_name + '_land.tif'):
        # load coastal data in correct UTM zone
        fname_shp_utm = cst_name[0:-4] + '_utm' + str(utm_list[i]).zfill(2) + '.shp'
        shp = ogr.Open(prc_path+fname_shp_utm)
        lyr = shp.GetLayer()
        
        # find S2 data in the given tile
        S2_name = [x for x in S2_list 
                  if x.split('_')[5]==tile_name]
        S2_name = S2_name[0]
        gran_path = im_path+S2_name+'/GRANULE/'
        S2_L1C_name = [ item for item in os.listdir(gran_path) if os.path.isdir(os.path.join(gran_path, item)) ]
        S2_full = im_path+S2_name+'/GRANULE/'+S2_L1C_name[0]+'/IMG_DATA/'+ \
            tile_name+'_'+S2_name.split('_')[2]+'_B04.jp2'
        
        
        # create raster environment
        S2src = gdal.Open(S2_full)
        driver = gdal.GetDriverByName('GTiff')
        target = driver.Create(prc_path + tile_name + '_land.tif', 
                               S2src.RasterXSize, S2src.RasterYSize,
                               1, gdal.GDT_UInt16)
        target.SetGeoTransform(S2src.GetGeoTransform())
        target.SetProjection(S2src.GetProjection())
        # convert polygon to raster
        #gdal.RasterizeLayer(target, [1], lyr, None, options = ["ATTRIBUTE=level"])
        gdal.RasterizeLayer(target, [1], lyr, options = ["ATTRIBUTE=level"])
        target.GetRasterBand(1).SetNoDataValue(0.0) 
        target = None
        
        
        #gdal.RasterizeLayer(target, [1], lyr, None, burn_values=[1])
        
        print('written ' + tile_name)
