import matplotlib.pyplot as plt

import os
import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.generic.handler_im import bilinear_interpolation
from eratosthenes.generic.mapping_tools import pix2map, pol2cart

from eratosthenes.preprocessing.read_sentinel2 import \
    read_view_angles_s2, read_detector_mask, list_central_wavelength_s2
from eratosthenes.preprocessing.image_transforms import mat_to_gray
from eratosthenes.preprocessing.shadow_geometry import \
    shadow_image_to_list, cast_orientation
from eratosthenes.preprocessing.acquisition_geometry import \
    get_template_acquisition_angles, get_template_aspect_slope
from eratosthenes.preprocessing.shadow_matting import \
    closed_form_matting_with_prior, compute_confidence_through_sun_angle

from eratosthenes.processing.matching_tools import \
    get_coordinates_of_template_centers, pad_radius
from eratosthenes.processing.matching_tools_differential import hough_sinus
from eratosthenes.processing.coupling_tools import \
    match_pair

from eratosthenes.postprocessing.solar_tools import \
    az_to_sun_vector, make_shading, make_shadowing

s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019/'
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

Z = read_geo_image(os.path.join(Z_dir, Z_file))[0]
R = read_geo_image(os.path.join(Z_dir, R_file))[0]

d_hat = 2*np.random.random((2))-1
Z2 = bilinear_interpolation(Z, d_hat[0], d_hat[1])

dZ = Z2-Z

window_size = 1
sample_I, sample_J = get_coordinates_of_template_centers(Z, window_size)

Slp,Asp = get_template_aspect_slope(Z,sample_I,sample_J,window_size)

tan_Slp = np.tan(np.radians(Slp))
dD = np.divide(dZ,tan_Slp,
               out=np.zeros_like(tan_Slp), where=tan_Slp!=0)
dD /= geoTransformM[1] # transform to metric

Asp_H,dD_H = hough_sinus(np.radians(Asp.flatten()), dD.flatten(), max_amp=2, sample_fraction=5000)
di, dj = pol2cart(dD_H, Asp_H)

plt.hist2d(Asp.flatten(), dD.flatten(),
           bins=90, range=[[-180, +180], [-2, 2]],
           cmap=plt.cm.jet)
plt.show()


