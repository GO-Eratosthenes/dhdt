import os

import matplotlib.pyplot as plt
import numpy as np

from dhdt.generic.handler_www import get_bz2_file, get_file_from_protected_www
from dhdt.generic.handler_aster import get_as_image_locations
from dhdt.generic.mapping_tools import pix2map, ll2map, get_utm_zone_simple

from dhdt.input.read_rapideye import list_central_wavelength_re, read_stack_re

from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.coupling_tools import match_pair

dat_dir = '/Users/Alten005/ICEFLOW/FreqStack' # os.getcwd()
dat_fol = ['20161031_225504_5919624_RapidEye-4',
           '20161117_225933_5919624_RapidEye-2'] # greenland
t_size = 2**5

for idx, foi in enumerate(dat_fol): # loop through to see if folders of interest exist
    re_dir = os.path.join(dat_dir, foi+os.path.sep)
    if idx==0:
        I1, spatialRef1, geoTransform1, targetprj1 = read_stack_re(re_dir)
    else:
        I2, spatialRef2, geoTransform2, targetprj2 = read_stack_re(re_dir)

grd_i,grd_j = get_coordinates_of_template_centers(I1, t_size) # //2
grd_x,grd_y = pix2map(geoTransform1,grd_i,grd_j)

#main processing
grd_x4,grd_y4, match_metric4 = match_pair(
    I1, I2, None, None, geoTransform1, geoTransform2, grd_x, grd_y,
    temp_radius=t_size, search_radius=t_size,
    correlator='ampl_comp', subpix='svd', metric='phase_fit')

grd_x1,grd_y1, match_metric1 = match_pair(
    I1[...,2], I2[...,2], None, None, geoTransform1, geoTransform2, grd_x, grd_y,
    temp_radius=t_size, search_radius=t_size,
    correlator='ampl_comp', subpix='svd', metric='phase_fit')

plt.figure()
plt.imshow(np.arctan2(grd_y1-grd_y, grd_x1-grd_x), cmap=plt.cm.twilight)

plt.figure()
plt.imshow(np.arctan2(grd_y4-grd_y, grd_x4-grd_x), cmap=plt.cm.twilight)

plt.figure()
plt.imshow(grd_x1-grd_x, vmin=-10, vmax=+10, cmap=plt.cm.RdBu), plt.colorbar()

plt.figure()
plt.imshow(grd_x4-grd_x, vmin=-10, vmax=+10, cmap=plt.cm.RdBu), plt.colorbar()

print('.')