import os
import numpy as np
import matplotlib.pyplot as plt

from eratosthenes.generic.handler_s2 import meta_S2string
from eratosthenes.generic.mapping_io import makeGeoIm, read_geo_image, \
    read_geo_info
from eratosthenes.generic.gis_tools import ll2utm, shape2raster

from eratosthenes.preprocessing.handler_multispec import create_shadow_image, \
    create_caster_casted_list_from_polygons
from eratosthenes.preprocessing.shadow_geometry import create_shadow_polygons

from eratosthenes.processing.coregistration import coregister, \
    get_coregistration, getNetworkBySunangles
from eratosthenes.processing.coupling_tools import pair_images, \
    match_shadow_casts, get_elevation_difference


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
    
    # get start and finish points of shadow edges
    conn1 = np.loadtxt(fname = dat_path+fname1+'conn.txt')
    conn2 = np.loadtxt(fname = dat_path+fname2+'conn.txt')
    
    # compensate for coregistration
    coid1 = coName.index(fname1)
    coid2 = coName.index(fname2)
    coDxy = coReg[coid1]-coReg[coid2]
    
    conn2[:,0] += coDxy[0]*10 # transform to meters?
    conn2[:,1] += coDxy[1]*10

    idxConn = pair_images(conn1, conn2)
    # connected list
    casters = conn1[idxConn[:,1],0:2]
    post1 = conn1[idxConn[:,1],2:4] # cast location in xy-coordinates for t1
    post2 = conn2[idxConn[:,0],2:4] # cast location in xy-coordinates for t1
    
    
    # refine through image matching
    (M1, crs, geoTransform1, targetprj) = read_geo_image(
            dat_path + fname1 + 'shadows.tif') # shadow transform template
    (M2, crs, geoTransform2, targetprj) = read_geo_image(
            dat_path + fname2 + 'shadows.tif') # shadow transform search space
    # castOrientation...
    post2_corr, corr_score = match_shadow_casts(M1, M2, geoTransform1, geoTransform2,
                       post1, post2)
    
    post1 = conn1[idxConn[:,1],2:4] # cast location in xy-coordinates for t1
    post2 = conn2[idxConn[:,0],2:4]
    # extract elevation change
    sun_1 = conn1[idxConn[:,1],4:6]
    sun_2 = conn2[idxConn[:,0],4:6]
    
    idx = 1
    xy_1 = post1[idx,:] # x_c = np.array([[post1[idx,0]], [post2_corr[idx,0]]])
    xy_2 = post2_corr[idx,:] # y_c = np.array([[post1[idx,1]], [post2_corr[idx,1]]]) 
    dh = get_elevation_difference(sun_1, sun_2, post1, post2_corr, casters)
    
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
        line = ('{:+8.2f}'.format(     post1[k, 0]) + ' ' + 
                '{:+8.2f}'.format(     post1[k, 1]) + ' ' +
                '{:+8.2f}'.format(post2_corr[k, 0]) + ' ' +
                '{:+8.2f}'.format(post2_corr[k, 1]) + ' ' +
                '{:+8.2f}'.format(   casters[k, 0]) + ' ' +
                '{:+8.2f}'.format(   casters[k, 1]) + ' ' +
                '{:+4.3f}'.format(        dh[k]))
        f.write(line + '\n')
    f.close()
    