import matplotlib.pyplot as plt

import os
import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image

from eratosthenes.preprocessing.shadow_transforms import entropy_shade_removal
from eratosthenes.preprocessing.image_transforms import mat_to_gray
from eratosthenes.preprocessing.acquisition_geometry import \
    get_template_aspect_slope

from eratosthenes.processing.matching_tools import \
    get_coordinates_of_template_centers, pad_radius

from eratosthenes.postprocessing.solar_tools import \
    make_shading, make_shadowing

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

Z = read_geo_image(os.path.join(dir_path, Z_file))[0]
Rgi = read_geo_image(os.path.join(dir_path, R_file))[0]

# create observation angles
fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019-full', \
                      'T05VMG_20191015T213531_B08.jp2')
_,spatialRefI,geoTransformI,targetprjI = read_geo_image(fpath)

path_meta = '/Users/Alten005/GO-eratosthenes/start-code/examples/'+\
    'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE/'+\
    'GRANULE/L1C_T05VMG_A022534_20191015T213843/QI_DATA'

Det = read_geo_image(os.path.join(s2path, 'detIdBlue.tif'))[0]
Zn = read_geo_image(os.path.join(s2path, 'viewZn.tif'))[0]
Az = read_geo_image(os.path.join(s2path, 'viewAz.tif'))[0]

Blue = read_geo_image(os.path.join(s2path, 'B2.tif'))[0]
Red = read_geo_image(os.path.join(s2path, 'B4.tif'))[0]
Near = read_geo_image(os.path.join(s2path, 'B8.tif'))[0]

Si,Ri = entropy_shade_removal(mat_to_gray(Blue), mat_to_gray(Red),mat_to_gray(Near))

Si += .5
Si[Si<-.5] = -.5

window_size = 1
sample_I, sample_J = get_coordinates_of_template_centers(Zn, window_size)

Slp,Asp = get_template_aspect_slope(pad_radius(Z,window_size),
                                    sample_I+window_size,
                                    sample_J+window_size,
                                    window_size)

zn, az = 90-21.3, 174.2 # 15 Oct 2019

# generate shadow image
Shw =  make_shadowing(Z, az, zn)
Shd =  make_shading(Z, az, zn)


# make Mask
Stable = (Rgi==0) & (Shw!=1)

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp.flatten(), Si.flatten(),
           bins=90, range=[[-180, +180], [-.75, .75]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp.flatten(), Shd.flatten(),
           bins=90, range=[[-180, +180], [-.75, .75]],
           cmap=plt.cm.gist_heat_r)


fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp[Stable], Shd[Stable],
           bins=90, range=[[-180, +180], [-.75, .75]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp[Stable], Si[Stable],
           bins=90, range=[[-180, +180], [-.75, .75]],
           cmap=plt.cm.gist_heat_r)

bins = np.linspace(-.5, 1, 100)
plt.hist([Si[Shw], Si[~Shw]], bins, label=['shadow', 'enlightened'])
plt.legend(loc='upper right')
