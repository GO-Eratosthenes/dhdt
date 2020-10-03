import os
import numpy as np
import matplotlib.pyplot as plt

from osgeo import ogr, osr, gdal

from eratosthenes.generic.handler_s2 import meta_S2string
from eratosthenes.generic.handler_www import url_exist, get_tar_file, \
    change_url_resolution
from eratosthenes.generic.mapping_io import makeGeoIm, read_geo_image, \
    read_geo_info
from eratosthenes.generic.mapping_tools import get_bbox_polygon, \
    find_overlapping_DEM_tiles, get_bbox
    
from eratosthenes.generic.gis_tools import ll2utm, shape2raster, \
    reproject_shapefile

from eratosthenes.preprocessing.handler_multispec import create_shadow_image, \
    create_caster_casted_list_from_polygons
from eratosthenes.preprocessing.shadow_geometry import create_shadow_polygons

from eratosthenes.processing.coregistration import coregister, \
    get_coregistration, getNetworkBySunangles
from eratosthenes.processing.coupling_tools import couple_pair


dat_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/'
#im_path = 'Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VPL_20180225T232042/'
#fName = 'T05VPL_20180225T214531_B'

im_path = ('Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VNK_20180225T232042/',
          'Data/S2B_MSIL1C_20190225T214529_N0207_R129_T05VNK_20190226T010218/',
          'Data/S2A_MSIL1C_20200225T214531_N0209_R129_T05VNK_20200225T231600/')
fName = ('T05VNK_20180225T214531_B', 'T05VNK_20190225T214529_B', 'T05VNK_20200225T214531_B')

shadow_transform = 'ruffenacht' # method to deploy for shadow enhancement

# do a subset of the imagery
minI = 4000 # minimal row coordiante
maxI = 6000 # maximum row coordiante
minJ = 4000 # minimal collumn coordiante
maxJ = 6000 # maximum collumn coordiante

bbox = (minI, maxI, minJ, maxJ)

for i in range(len(im_path)):
    sen2Path = dat_path + im_path[i]
    (sat_time,sat_orbit,sat_tile) = meta_S2string(im_path[i])
    print('working on '+ fName[i][0:-2])
    if not os.path.exists(sen2Path + 'shadows.tif'):
        (M, geoTransform, crs) = create_shadow_image(dat_path, im_path[i], \
                                                   shadow_transform, \
                                                   minI, maxI, minJ, maxJ \
                                                   )
        print('produced shadow transform for '+ fName[i][0:-2])
        makeGeoIm(M, geoTransform, crs, sen2Path + 'shadows.tif')
    else:
        (M, crs, geoTransform, targetprj) = read_geo_image(
            sen2Path + 'shadows.tif')

    if not os.path.exists(sen2Path + 'labelCastConn.tif'):
        (labels, cast_conn) = create_shadow_polygons(M,sen2Path, \
                                                     minI, maxI, minJ, maxJ \
                                                     )

        makeGeoIm(labels, geoTransform, crs, sen2Path + 'labelPolygons.tif')
        makeGeoIm(cast_conn, geoTransform, crs, sen2Path + 'labelCastConn.tif')
        print('labelled shadow polygons for '+ fName[i][0:-2])

# make raster with Randolph glacier mask for the tile
if not os.path.exists(dat_path + sat_tile + '.tif'):
    # create RGI raster for the extent of the image
    rgi_path = dat_path+'GIS/'
    rgi_file = '01_rgi60_Alaska.shp'
    out_shp = rgi_path+rgi_file[:-4]+'_utm'+sat_tile[1:3]+'.shp'
    # get geo-meta data for a tile
    fname = dat_path + im_path[0] + fName[0] + '04.jp2'
    crs, geoTransform, targetprj, rows, cols, bands = read_geo_info(fname)
    aoi = 'RGIId'
    if not os.path.exists(out_shp):  # project RGI shapefile
        # transform shapefile from lat-long to UTM
        ll2utm(rgi_path+rgi_file,out_shp,crs,aoi)
    # convert polygon file to raster file
    shape2raster(out_shp, dat_path+sat_tile, geoTransform, rows, cols, aoi)

# make raster with elevation data for the tile
if not os.path.exists(dat_path + sat_tile + '_DEM.tif'):
    print('building digital elevation model (DEM) for '+ sat_tile )
    # get geo info of tile
    crs, geoTransform, targetprj, rows, cols, bands = read_geo_info(dat_path + 
                                                                    sat_tile + 
                                                                    '.tif')
    
    ### OBS
    # randolph tile seems to have a missing crs... etc OGR does not see anything
    ###
    
    fname = dat_path + im_path[0] + fName[0] + '04.jp2'
    crs, geoTransform, targetprj, rows, cols, bands = read_geo_info(fname)
    
    # get DEM tile structure
    dem_path = dat_path+'GIS/'
    dem_file = 'ArcticDEM_Tile_Index_Rel7.shp'
    
    # find overlapping tiles of the granual
    poly_tile = get_bbox_polygon(geoTransform, rows, cols)
    dem_proj_file = reproject_shapefile(dem_path, dem_file, targetprj)
    url_list = find_overlapping_DEM_tiles(dem_path,dem_proj_file, poly_tile)
    print('found '+str(len(url_list))+ ' elevation chips connected to this tile')
    
    # create sampling grid   
    bbox = get_bbox(geoTransform, rows, cols)
    
    new_res = 10 # change to 10meter url
    
    # because the shapefiles are in 2 meter, the tiles are 4 fold, therfore 
    # make a selection, to bypass duplicates
    tiles = ()
    for i in url_list: 
        tiles += (i.split('/')[-2],)
    uni_set = set(tiles)
    ids = []
    for i in range(len(uni_set)):
        idx = tiles.index(uni_set.pop())
        ids.append(idx)
    url_list = [url_list[i] for i in ids]    
    print('reduced to '+str(len(url_list))+ ' elevation chips')
    for i in range(len(url_list)):
        gran_url = url_list[i]
        gran_url_new = change_url_resolution(gran_url,new_res)
        
        # download and integrate DEM data into tile
        print('starting download of DEM tile')
        if url_exist(gran_url_new):
            tar_names = get_tar_file(gran_url_new, dem_path)
        else:
            tar_names = get_tar_file(gran_url, dem_path)
        print('finished download of DEM tile')
            
        # load data, interpolate into grid
        dem_name = [s for s in tar_names if 'dem.tif' in s]
        if i ==0:
            dem_new_name = sat_tile + '_DEM.tif'
        else:
            dem_new_name = dem_name[0][:-4]+'_utm.tif'
        
        ds = gdal.Warp(os.path.join(dem_path, dem_new_name), 
                       os.path.join(dem_path, dem_name[0]), 
                       dstSRS=crs,
                       outputBounds=(bbox[0], bbox[2], bbox[1], bbox[3]),
                       xRes=new_res, yRes=new_res,
                       outputType=gdal.GDT_Float64)
        ds = None
        
        if i>0: # mosaic tiles togehter
            merge_command = ['python', 'gdal_merge.py', 
                             '-o', os.path.join(dem_path, sat_tile + '_DEM.tif'), 
                             os.path.join(dem_path, sat_tile + '_DEM.tif'), 
                             os.path.join(dem_path, dem_new_name)]
            my_env = os.environ['CONDA_DEFAULT_ENV']
            os.system('conda run -n ' + my_env + ' '+
                      ' '.join(merge_command[1:]))
            os.remove(os.path.join(dem_path,dem_new_name))
            
        for fn in tar_names:
            os.remove(os.path.join(dem_path,fn))
    print('end of loop, rename now!')        
        # remove original DEM
 
    #os.system('conda run -n '+os.environ['CONDA_DEFAULT_ENV']+''+' '.join(merge_command[1:]))
    
     # driver = gdal.GetDriverByName('GTiff')
    # rgiRaster = driver.Create(im_fname+'.tif', rows, cols, 1, gdal.GDT_Int16)
    # #            rgiRaster = gdal.GetDriverByName('GTiff').Create(dat_path+sat_tile+'.tif', mI, nI, 1, gdal.GDT_Int16)           
    # rgiRaster.SetGeoTransform(geoTransform)
    # band = rgiRaster.GetRasterBand(1)
    # #assign no data value to empty cells.
    # band.SetNoDataValue(0)   
    
    

(rgi_mask, crs, geoTransform, targetprj) = read_geo_image(dat_path
                                                          +sat_tile+'.tif')
rgi_mask = rgi_mask[minI:maxI,minJ:maxJ]

# TO DO:
# make raster with elevation values for the tile


# make stack of Labels & Connectivity
rgi_glac_id = 22216 # None,  select a single glacier
if rgi_glac_id is not None:
    print(f'looking at glacier with Randolph ID { rgi_glac_id }.')
for i in range(len(im_path)):
    create_caster_casted_list_from_polygons(dat_path, im_path[i], bbox,  
                                            rgi_glac_id)

print('co-registring imagery')
coregister(im_path, dat_path, connectivity=2, step_size=True, temp_size=15,
           bbox=bbox, lstsq_mode='ordinary')
#lkTransform = RefScale(RefTrans(subTransform,tempSize/2,tempSize/2),tempSize)
#makeGeoIm(Dstack[0],lkTransform,crs,"DispAx1.tif")


## processing

# get co-registration information
(coName,coReg) = get_coregistration(dat_path, im_path)

# construct connectivity
connectivity = 2
GridIdxs = getNetworkBySunangles(dat_path, im_path, connectivity)
for i in range(GridIdxs.shape[1]):
    fname1 = im_path[GridIdxs[0][i]]
    fname2 = im_path[GridIdxs[1][i]]
    
    xy_1, xy_2, casters, dh = couple_pair(dat_path, fname1, fname2,
                                          coName, coReg)

    # write elevation distance    
    (time_1, orbit_1, tile_1) = meta_S2string(fname1)
    (time_2, orbit_2, tile_2) = meta_S2string(fname2)
    dh_fname = ('dh-' + time_1 +'-'+ time_2 +'-'+ 
                orbit_1 + '-' + orbit_2 +'-'+ 
                tile_1 +'-'+ tile_2)
    if rgi_glac_id is not None:
        dh_fname = (dh_fname + '-' + '{:08d}'.format(rgi_glac_id) + '.txt')
    else:
        dh_fname = (dh_fname + '.txt')
        
    f = open(dat_path + 'dh_fname', 'w')
    for k in range(dh.shape[0]):
        line = ('{:+8.2f}'.format(   xy_1[k, 0]) + ' ' + 
                '{:+8.2f}'.format(   xy_1[k, 1]) + ' ' +
                '{:+8.2f}'.format(   xy_2[k, 0]) + ' ' +
                '{:+8.2f}'.format(   xy_2[k, 1]) + ' ' +
                '{:+8.2f}'.format(casters[k, 0]) + ' ' +
                '{:+8.2f}'.format(casters[k, 1]) + ' ' +
                '{:+4.3f}'.format(     dh[k]))
        f.write(line + '\n')
    f.close()
    