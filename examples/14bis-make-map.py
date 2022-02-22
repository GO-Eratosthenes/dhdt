import os
import pathlib
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from osgeo import ogr, osr, gdal
import rioxarray

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.mapping_tools import get_bbox, GDAL_transform_to_affine
#from eratosthenes.preprocessing.shadow_geometry import make_shading
from eratosthenes.postprocessing.solar_tools import make_shading
from eratosthenes.presentation.terrain_tools import \
    luminance_perspective, contrast_perspective, enhanced_vertical_shading, \
    adaptive_elevation_smoothing

from eratosthenes.generic.test_tools import create_sample_image_pair, \
    construct_phase_plane, signal_to_noise
#from eratosthenes.processing.matching_tools import normalize_power_spectrum, \
#    perdecomp, phase_corr, phase_only_corr, robust_corr, \
#    local_coherence, cross_shading_filter


fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-25-10-2019/', \
                      'shadow.tif')
M_dir, M_name = os.path.split(fpath)
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__)) 

Z = read_geo_image(os.path.join(Z_dir, Z_file))[0]

Z_smh = adaptive_elevation_smoothing(Z)

Zn_1,Az_1 = 90-17.7, 130 # 25 Oct 2019 

Shd_1 =  make_shading(Z, Zn_1, Az_1, spac=10)
#Shd_1 = (Shd_1+1)/2

Zn_2,Az_2 = 90-17.7, 165 # 25 Oct 2019 

Shd_2 =  make_shading(Z, Zn_2, Az_2, spac=10)
Shd_2 = (Shd_2+1)/2

Shd_v = enhanced_vertical_shading(Z, Zn_1, Zn_2, spac=10, c=.99)

Shd_bras = luminance_perspective(Shd_1, Z, .5, k=5)
Shd_jen = contrast_perspective(Shd_1, Z, .7, 2000, v_min=0.3, k=2)

plt.imshow(Shd_jen, cmap='gray'), plt.show()



im1 = Shd_1[400:400+128,800:800+128]
im2 = Shd_2[404:404+128,801:801+128]
(im1,_), (im2,_) = perdecomp(im1), perdecomp(im2)

Q_hat = phase_corr(im1, im2) # cross-power spectrum
#Q_hat = phase_only_corr(im1, im2)
#Q_hat = robust_corr(im1, im2)
Q_hat = normalize_power_spectrum(Q_hat)


ax = plt.subplot()
im = ax.imshow(np.fft.fftshift(np.angle(Q_hat)), cmap='twilight')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

ax = plt.subplot()
im = ax.imshow(np.fft.fftshift(local_coherence(Q_hat)))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

W = cross_shading_filter(Q_hat)

Zn, Az = 90-17.7, 174.8 # 25 Oct 2019 

Shd =  make_shading(dir_path, Z_file, M_dir, M_name) #, Zn=Zn, Az=Az)
Shd = (Shd+1)/2


