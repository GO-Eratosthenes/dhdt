import os
import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import read_geo_image, read_geo_info
from dhdt.generic.mapping_tools import \
    pix2map, map2pix, bilinear_interp_excluding_nodat
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.preprocessing.acquisition_geometry import \
    slope_along_perp
from dhdt.processing.matching_tools import \
    get_coordinates_of_template_centers
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope
from dhdt.processing.coupling_tools import match_pair
from dhdt.postprocessing.mapping_io import dh_txt2shp
from dhdt.presentation.image_io import output_image

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'


im_path = ('S2-2021-04-05', #'S2-2020-10-19',
           'S2-2021-10-27')

soi = 'S.tif'
vname = 'view.tif'
window_size = 2**4
spac=10

# get (meta)-data
im_1_name, im_2_name = im_path[0], im_path[1]
fname_1 = os.path.join(base_dir, im_1_name, soi)
fname_2 = os.path.join(base_dir, im_2_name, soi)

A_1 = read_geo_image(fname_1)[0][:,:,1]
A_2 = read_geo_image(fname_2)[0][:,:,1]
geoTransform = read_geo_info(fname_1)[1]

Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
R_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')
Z = read_geo_image(os.path.join(R_dir, Z_file))[0]
R = read_geo_image(os.path.join(R_dir, R_file))[0]

V_1 = read_geo_image(os.path.join(base_dir, im_1_name, vname))[0]
S_1_perp, S_1_para = slope_along_perp(Z, V_1[:,:,-1], spac)
del V_1

Glac = np.ones_like(R)#(R != 0)
sample_I, sample_J = get_coordinates_of_template_centers(A_1, window_size)
sample_X, sample_Y = pix2map(geoTransform, sample_I, sample_J)
match_X, match_Y, match_score = match_pair(A_1, A_2,
                                           Glac, Glac,
                                           geoTransform, geoTransform,
                                           sample_X, sample_Y,
                                           temp_radius=window_size,
                                           search_radius=window_size,
                                           correlator='phas_corr',
                                           subpix='moment',
                                           metric='peak_entr') # 'robu_corr'

dY, dX = sample_Y - match_Y, sample_X - match_X
IN = ~np.isnan(dX)
IN[IN] = np.remainder(dX[IN], 1) != 0

Slp, Asp = get_template_aspect_slope(Z,
                                     sample_I, sample_J,
                                     window_size)

output_image(Asp, 'aspect-hr.jpg', cmap='twilight', compress=95)
output_image(np.arctan2(dX,dY), 'orient.jpg', cmap='twilight_r', compress=95)

Mag = mat_to_gray(np.log10(np.sqrt(dX**2+dY**2)), vmin=.25, vmax=2.0)

output_image(Slp, 'slope.jpg', cmap='turbo', compress=95) # 'viridis'
output_image(Mag, 'magnitude.jpg', cmap='turbo', compress=95) # inferno

print('jaja')