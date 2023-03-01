import os
import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import \
    read_geo_image, read_geo_info, make_geo_im
from dhdt.generic.mapping_tools import ll2map, pix_centers, map2pix
from dhdt.generic.handler_im import bilinear_interpolation
from dhdt.generic.gis_tools import shape2raster
from dhdt.input.read_planetscope import list_central_wavelength_sd, \
    read_data_ps, read_detector_type_ps, list_central_wavelength_dr
from dhdt.preprocessing.image_transforms import \
    general_midway_equalization, high_pass_im
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope
from dhdt.preprocessing.image_transforms import mat_to_gray, gamma_adjustment
from dhdt.processing.matching_tools_differential import hough_sinus,\
    differential_stacking
from dhdt.processing.matching_tools_frequency_filters import perdecomp
from dhdt.processing.network_tools import get_network_indices_constrained

im_dir = '/Users/Alten005/SEES/Doc/Ibiza'
#im_name = '20220611_103921_72_2402_3B_AnalyticMS_SR_8b_harmonized.tif'
im_name = '20220611_103921_72_2402_3B_AnalyticMS_8b.tif'

#im_dir = '/Users/Alten005/StMaarten/Data/PSScene4Band/20210704_142134_100a/analytic_sr_udm2'
#im_name = '20210704_142134_100a_3B_AnalyticMS_SR.tif'

sat_type = read_detector_type_ps(im_dir)

df = list_central_wavelength_sd(num_bands=8)
#df = list_central_wavelength_dr()
df.sort_values(by=['bandid'], inplace=True)
df = df[df['bandid'].notna()]
df = df[2:4]
det_tim_sep = .15

w,h = 2**6, 2**6
ic,jc = 4500,1000
#ic,jc = 2500,7700

f_full = os.path.join(im_dir, im_name)
#I = read_band_ps(im_dir, im_name, no_dat=1)[0]
I = read_data_ps(f_full,df)[0]
I = I[ic:ic+h,jc:jc+w,:]
print('image read')

Ih = general_midway_equalization(I)
Ih *= 1E-3
print('imagery equalized')

# rescale
from dhdt.generic.handler_im import rescale_image
Ih_s = rescale_image(Ih.data, 0.25)

# smoothing
Ip = perdecomp(Ih)[0]
Is = Ih - high_pass_im(Ip.data, radius=5)
#plt.figure(), plt.imshow(high_pass_im(Ih.data[...,0], radius=100))



acq_ord = np.array(df['acquisition_order'], dtype=np.float64)
ids = np.array(df['bandid'], dtype=np.int64)
idxs = get_network_indices_constrained(ids,
       acq_ord, 10, 2, double_direction=False)
couple_dt = acq_ord[idxs[0,:]] - acq_ord[idxs[1,:]]
couple_dt *= det_tim_sep

# smoothing


dt_I = differential_stacking(Is, idxs[0,:], idxs[1,:], couple_dt)
print('.')

plt.figure(),
plt.hexbin(dt_I[:,0],dt_I[:,1], extent=(-.5,+.5,-1,+1), gridsize=20)
