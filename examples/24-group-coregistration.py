import os

import numpy as np

from dhdt.generic.mapping_io import \
    read_geo_image
from dhdt.generic.mapping_tools import pix2map, get_bbox, map2pix
from dhdt.input.read_sentinel2 import \
    read_mean_sun_angles_s2
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope
from dhdt.processing.coupling_tools import match_pair
from dhdt.processing.matching_tools import \
    get_coordinates_of_template_centers
from dhdt.postprocessing.solar_tools import \
    make_shading, make_shadowing
from dhdt.testing.matching_tools import create_shadow_caster_casted

window_size = 2**4

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
im_path = ('S2-2021-10-27',
           'S2-2021-04-05',
           'S2-2020-10-14',
           'S2-2020-11-06',
           'S2-2019-10-25',
           'S2-2019-10-15',
           'S2-2017-10-20',
           'S2-2017-10-18')
im_path = ('S2-2019-10-25',
           'S2-2019-10-15')

(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))
(R, spatialRefM, geoTransformM, targetprjM) = \
    read_geo_image(os.path.join(R_dir, R_file))

# get subset of data
bbox_R = get_bbox(geoTransformM, R.shape[0], R.shape[1])
bbox_R = bbox_R.reshape((2,2)).T
subset_i,subset_j = map2pix(geoTransformZ, bbox_R[:,0], bbox_R[:,1])
subset_i,subset_j = np.sort(subset_i.astype(int)), np.sort(subset_j.astype(int))
Z = Z[subset_i[0]:subset_i[1],subset_j[0]:subset_j[1]]

# get couple
im_1_name, im_2_name = im_path[0], im_path[1]

im_1_dir = os.path.join(im_dir, im_1_name)
im_2_dir = os.path.join(im_dir, im_2_name)

# read imagery
S_1 = read_geo_image(os.path.join(im_1_dir, 'S.tif'))[0]
V_1 = read_geo_image(os.path.join(im_1_dir, 'view.tif'))[0]
S_2 = read_geo_image(os.path.join(im_2_dir, 'S.tif'))[0]
V_2 = read_geo_image(os.path.join(im_2_dir, 'view.tif'))[0]

# get meta-data
zn_1, az_1 = read_mean_sun_angles_s2(im_1_dir, fname='MTD_TL.xml')
zn_2, az_2 = read_mean_sun_angles_s2(im_2_dir, fname='MTD_TL.xml')

Shw_t = create_shadow_caster_casted(Z, geoTransformM, az_1, zn_1, im_1_dir)

# make shadowing
Shw_1, Shd_1 = make_shadowing(Z, az_1, zn_1), make_shading(Z, az_1, zn_1)
Shw_2, Shd_2 = make_shadowing(Z, az_2, zn_2), make_shading(Z, az_2, zn_2)

# do coregistration
Stable = (R == 0) # off-glacier terrain
sample_I, sample_J = get_coordinates_of_template_centers(Z, window_size)

sample_X, sample_Y = pix2map(geoTransformM, sample_I, sample_J)
match_X, match_Y, match_score = match_pair(Shd_1, S_1[:,:,0],
                                           Stable, Stable,
                                           geoTransformM, geoTransformM,
                                           sample_X, sample_Y,
                                           temp_radius=window_size,
                                           search_radius=window_size,
                                           correlator='phas_only',
                                           subpix='moment',
                                           metric='peak_entr') # 'robu_corr'

dY, dX = sample_Y - match_Y, sample_X - match_X
IN = ~np.isnan(dX)
IN[IN] = np.remainder(dX[IN], 1) != 0

Slp, Asp = get_template_aspect_slope(Z,
                                     sample_I, sample_J,
                                     window_size)


Abs_norm = np.divide(np.sqrt(dX**2 + dY**2),np.tan(np.deg2rad(Slp)))
dDir = np.rad2deg(np.arctan2(dX,dY))
# plt.hist2d(Asp[~np.isnan(dDir)], dDir[~np.isnan(dDir)], bins=36)

PURE = np.logical_and(IN, Slp<20)
dx_coreg, dy_coreg = np.median(dX[PURE]), np.median(dY[PURE])
print(dx_coreg, dy_coreg)