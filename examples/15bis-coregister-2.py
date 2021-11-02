import matplotlib.pyplot as plt

import os
import pathlib
import numpy as np

from osgeo import ogr, osr, gdal
from skimage.exposure import match_histograms
import rioxarray

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.mapping_tools import \
    get_bbox, GDAL_transform_to_affine, pix_centers, map2pix, pix2map, pol2cart
from eratosthenes.generic.handler_cop import mosaic_tiles
from eratosthenes.generic.handler_im import \
    rotated_sobel, bilinear_interpolation

from eratosthenes.preprocessing.read_sentinel2 import \
    read_view_angles_s2, read_detector_mask, list_central_wavelength_s2
from eratosthenes.preprocessing.image_transforms import mat_to_gray
from eratosthenes.preprocessing.shadow_geometry import \
    shadow_image_to_list, cast_orientation
from eratosthenes.preprocessing.acquisition_geometry import \
    get_template_acquisition_angles, get_template_aspect_slope
from eratosthenes.preprocessing.shadow_matting import \
    closed_form_matting_with_prior, compute_confidence_through_sun_angle
from eratosthenes.preprocessing.shadow_transforms import ica

from eratosthenes.processing.matching_tools import \
    get_coordinates_of_template_centers, pad_radius
from eratosthenes.processing.coupling_tools import \
    match_pair
from eratosthenes.processing.matching_tools_differential import hough_sinus

from eratosthenes.postprocessing.solar_tools import \
    az_to_sun_vector, make_shading, make_shadowing

boi = ['red', 'green', 'blue', 'near infrared']
s2_df = list_central_wavelength_s2()
s2_df = s2_df[s2_df['name'].isin(boi)]

s2path = '/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/'
fpath = os.path.join(s2path, 'shadow.tif')

M_dir, M_name = os.path.split(fpath)
#dat_path = '/Users/Alten005/GO-eratosthenes/start-code/examples/'
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.getcwd()!=dir_path:
    os.chdir(dir_path) # change to directory where script is situated

(M, spatialRefM, geoTransformM, targetprjM) = read_geo_image(fpath)
M *= -1

Z = read_geo_image(os.path.join(dir_path, Z_file))[0]
R = read_geo_image(os.path.join(dir_path, R_file))[0]

# create observation angles
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019-full', \
                      'T05VMG_20191015T213531_B08.jp2')
_,spatialRefI,geoTransformI,targetprjI = read_geo_image(fpath)

path_meta = '/Users/Alten005/GO-eratosthenes/start-code/examples/'+\
    'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE/'+\
    'GRANULE/L1C_T05VMG_A022534_20191015T213843/QI_DATA'

#det_stack =read_detector_mask(path_meta, (10980, 10980, len(s2_df)), s2_df, geoTransformI)
#Zn,Az = read_view_angles_s2(s2path, 'metadata.xml', det_stack, s2_df)

#X_grd,Y_grd = pix_centers(geoTransformM, M.shape[0], M.shape[1], make_grid=True)
#I_grd,J_grd = map2pix(geoTransformI, X_grd, Y_grd)

#I_sub = bilinear_interpolation(Az[:,:,0], J_grd, I_grd)

#fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
#                      'viewZn.tif')
#make_geo_im(I_sub, geoTransformM, spatialRefM, fpath)

Det = read_geo_image(os.path.join(s2path, 'detIdBlue.tif'))[0]
Zn = read_geo_image(os.path.join(s2path, 'viewZn.tif'))[0]
Az = read_geo_image(os.path.join(s2path, 'viewAz.tif'))[0]

Blue = read_geo_image(os.path.join(s2path, 'B2.tif'))[0]
Green = read_geo_image(os.path.join(s2path, 'B3.tif'))[0]
Red = read_geo_image(os.path.join(s2path, 'B4.tif'))[0]
Near = read_geo_image(os.path.join(s2path, 'B8.tif'))[0]

from eratosthenes.preprocessing.shadow_transforms import entropy_shade_removal, shadow_free_rgb
from eratosthenes.preprocessing.color_transforms import rgb2lms, rgb2hsi

Li,Mi,Si = rgb2lms(mat_to_gray(Red), mat_to_gray(Green), mat_to_gray(Blue))
RGB = shadow_free_rgb(mat_to_gray(Blue), mat_to_gray(Green),
                      mat_to_gray(Red), mat_to_gray(Near))

Si,Ri = entropy_shade_removal(mat_to_gray(Blue), mat_to_gray(Red),mat_to_gray(Near))

Si += .5
Si[Si<-.5] = -.5

window_size = 1 #2**3
sample_I, sample_J = get_coordinates_of_template_centers(Zn, window_size)

Slp,Asp = get_template_aspect_slope(pad_radius(Z,window_size),
                                    sample_I+window_size,
                                    sample_J+window_size,
                                    window_size)

Azi,Zen = get_template_acquisition_angles(pad_radius(Az,window_size), pad_radius(Zn,window_size),
                                          pad_radius(Det,window_size).astype('int64'),
                                          sample_I+window_size,
                                          sample_J+window_size,
                                          window_size)


zn, az = 90-21.3, 174.2 # 15 Oct 2019
#Zn, Az = 90-17.7, 174.8 # 25 Oct 2019

# generate shadow image
Shw =  make_shadowing(Z, az, zn)
Shd =  make_shading(Z, az, zn)

#plt.imshow(Shd, cmap='gray'), plt.show()
#plt.imshow(M, cmap='gray'), plt.show()

# filter along sun direction....?

#from eratosthenes.preprocessing.image_transforms import \
#    log_adjustment, mat_to_gray
#L = log_adjustment(mat_to_gray(M)*2**8)/2**16

#H = match_histograms(L,Shd)

F = cast_orientation(Shd, Az, indexing='xy')

# fa = rotated_sobel(Az, indexing='xy')
# from scipy import ndimage
# D = ndimage.convolve(Shd, fa)

N = cast_orientation(Si, Az, indexing='xy')

# make Mask
Stable = (R==0) & (Shw!=1)

window_size = 2**3
sample_X,sample_Y = pix2map(geoTransformM, sample_I, sample_J)
match_X,match_Y = match_pair(Shd, Si, Stable, Stable, geoTransformM, geoTransformM,
                             sample_X, sample_Y,
                             temp_radius=window_size,
                             search_radius=window_size,
                             correlator='mask_corr', subpix='svd',
                             processing='simple')

#
#from sklearn import linear_model
#lr = linear_model.LinearRegression()
#lr.fit(x, y)
#ransac = linear_model.RANSACRegressor(max_trials=1000,min_samples=300)
#ransac.fit(Si[Stable].reshape(-1, 1), Shd[Stable].reshape(-1, 1))
#Sr = ransac.predict(Si.reshape(-1, 1)).reshape(Si.shape)

#plt.scatter(asp_val,asp_med,s=1,c='black',marker='.')
#plt.scatter(asp_val,asp_num,s=1,c='black',marker='.')
#IN = asp_num>np.quantile(asp_num, .5)
#plt.scatter(asp_val[IN],asp_num[IN],s=1,c='blue',marker='.')

#Sm = match_histograms(-Si[Stable],Shd[Stable])

dI = Shd-Si #Shd + 4*Si#-L
tan_Slp = np.tan(np.radians(Slp))
dD = np.divide(dI,tan_Slp,
               out=np.zeros_like(tan_Slp), where=tan_Slp!=0)

Asp_Shd,dD_Shd = hough_sinus(np.radians(Asp[Stable]), Shd[Stable],
                             max_amp=1, sample_fraction=5000)
Asp_Si,dD_Si = hough_sinus(np.radians(Asp[Stable]), Si[Stable],
                           max_amp=1, sample_fraction=5000)

#di, dj = pol2cart(dD_H, Asp_H)

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp.flatten(), Si.flatten(),
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp[Stable], Si[Stable],
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)


fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp[Stable], Shd[Stable],
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp[Stable], Si[Stable],
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)
ax1.scatter(asp_val,asp_med,s=1,c='black',marker='.')

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.imshow(-Si, cmap=plt.cm.bone)
ax1.imshow(Shd, cmap=plt.cm.bone)

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.imshow(F, cmap=plt.cm.bone)
ax1.imshow(N, cmap=plt.cm.bone)

plt.hist2d(np.divide(Blue-Green,Blue+Green)[Stable],
           np.divide(Red-Near,Red+Near)[Stable],
           bins=100, range=[[-1, 1], [-1, 1]],
           cmap=plt.cm.gist_heat_r)


plt.hist2d(Asp[Stable], dI[Stable],
           bins=90, range=[[-180, +180], [-1, +1]],
           cmap=plt.cm.gist_heat_r)

plt.hist(Asp[Stable])


dY,dX = sample_Y-match_Y, sample_X-match_X
#plt.imshow(dY, vmin=-20, vmax=+20), plt.show()
#plt.imshow(dX, vmin=-20, vmax=+20), plt.show()

Rho, Phi = np.sqrt(dY**2,dX**2), np.arctan2(dY,dX)

dD = np.divide(Rho,np.tan(np.radians(Slp)))
dD = np.multiply(Rho,np.tan(np.radians(Slp)))

IN = dD!=0
# IN = (dX!=0) & (dY!=0)
plt.hist2d(Asp[IN], dD[IN], bins=90, range=[[-180, +180], [0, .4]], cmap=plt.cm.jet)

plt.scatter(Phi, np.divide(Rho,np.tan(np.radians(Slp))))


