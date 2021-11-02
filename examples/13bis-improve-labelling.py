# explore other label generators than just Otsu (general, does not work with shifting snowlines?)


import matplotlib.pyplot as plt

import os
import pathlib
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

from scipy import ndimage
import morphsnakes as ms

from osgeo import ogr, osr, gdal
import rioxarray

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.mapping_tools import \
    get_bbox, GDAL_transform_to_affine, pix_centers, map2pix
from eratosthenes.generic.handler_cop import mosaic_tiles
from eratosthenes.generic.handler_im import \
    rotated_sobel, bilinear_interpolation
from eratosthenes.preprocessing.image_transforms import mat_to_gray
from eratosthenes.preprocessing.shadow_geometry import \
    make_shadowing, shadow_image_to_list
from eratosthenes.preprocessing.shadow_matting import \
    closed_form_matting_with_prior, compute_confidence_through_sun_angle
from eratosthenes.postprocessing.solar_tools import \
    az_to_sun_vector, make_shading_minnaert
#fpath1 = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-25-10-2019/', \
#                      'shadow.tif')
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
                      'shadow.tif')
M_dir, M_name = os.path.split(fpath)
#dat_path = '/Users/Alten005/GO-eratosthenes/start-code/examples/'
Z_file = "COP_DEM_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/', \
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__)) 
if os.getcwd()!=dir_path: 
    os.chdir(dir_path) # change to directory where script is situated

(M, spatialRefM, geoTransformM, targetprjM) = read_geo_image(fpath)

# if it does not exist, create DEM
if not os.path.exists(os.path.join(dir_path, Z_file)):
    bbox = get_bbox(geoTransformM, M.shape[0], M.shape[1])
    dem_tiles_filenames = pathlib.Path(Z_dir).glob("**/*_DEM.tif")
    affTransformM = GDAL_transform_to_affine(geoTransformM)
    Z = mosaic_tiles(dem_tiles_filenames, bbox, spatialRefM, affTransformM)
    
    Z.rio.to_raster(os.path.join(dir_path, Z_file))
#Z = read_geo_image(os.path.join(dir_path, Z_file))[0]

#fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019-full', \
#                      'T05VMG_20191015T213531_B08.jp2')
#(I, spatialRefI, geoTransformI, targetprjI) = read_geo_image(fpath)
#
#X_grd,Y_grd = pix_centers(geoTransformM, M.shape[0], M.shape[1], make_grid=True)
#I_grd,J_grd = map2pix(geoTransformI, X_grd, Y_grd)
#
#I_sub = bilinear_interpolation(I, J_grd, I_grd)
#
#fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
#                      'B8.tif')
#make_geo_im(I_sub, geoTransformM, spatialRefM, fpath)
        
# get 10 meter bands
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
                      'B3.tif')
(B3,_,_,_) = read_geo_image(fpath)
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
                      'B4.tif')
(B4,_,_,_) = read_geo_image(fpath)
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
                      'B8.tif')
(B8,_,_,_) = read_geo_image(fpath)    
    
Zn, Az = 90-21.3, 174.2 # 15 Oct 2019 
#Zn, Az = 90-17.7, 174.8 # 25 Oct 2019 

# generate shadow image
Shw =  make_shadowing(dir_path, Z_file, M_dir, M_name, Zn=Zn, Az=Az)

B348 = np.dstack((mat_to_gray(B3, notI=B3==0),\
                  mat_to_gray(B4, notI=B4==0),\
                  mat_to_gray(B8, notI=B8==0)))

    
    
Conf = compute_confidence_through_sun_angle(Shw, Az)    
alpha = closed_form_matting_with_prior(B348, \
                                       Shw, \
                                       100*mat_to_gray(Conf), \
                                       consts_map=None)


shadow_image_to_list(M, geoTransformM, dir_path, [], method='svm', 
                     Shw=Shw, Zn=Zn, Az=Az)


Shd2 =  make_shading_minnaert(Z, Az, Zn, k=1.0)




# makeGeoIm(M_init, geoTransformM, spatialRefM, "COP_DEM_shadow_red.tif", \
#               sun_angles = f"az:{Az}-zn:{Zn}", date_created='2019-10-25')

M_init =  make_shadowing(dir_path, Z_file, M_dir, M_name, Zn=Zn, Az=Az)

sun_fil = rotated_sobel(Az, size=5).T

alpha = 10
M_g = 1/ (1 + alpha*np.abs(ndimage.convolve(M, sun_fil)))
#M_g = ms.inverse_gaussian_gradient(M, sigma=1.0)

num_iter = 10

# simple threshold
#M_init = M>=np.quantile(M,.8)


M_gac = ms.morphological_geodesic_active_contour(M_g, num_iter, M_init, \
                                             smoothing=1, threshold=1,
                                             balloon=0)
# M_gac = ms.morphological_geodesic_active_contour(M_g, num_iter, M_gac, \
#                                              smoothing=1, threshold=1,
#                                              balloon=-1)
    
makeGeoIm(M_gac, geoTransformM, spatialRefM, "COP_DEM_GAC2_red.tif", \
              sun_angles = f"az:{Az}-zn:{Zn}", date_created='2019-10-25')
