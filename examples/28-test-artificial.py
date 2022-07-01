import os

import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import \
    read_geo_image, read_geo_info, make_geo_im
from dhdt.generic.mapping_tools import pix2map, get_bbox, map2pix
from dhdt.input.read_sentinel2 import read_mean_sun_angles_s2
from dhdt.generic.test_tools import \
    create_shadow_caster_casted, test_photohypsometric_refinement, \
    create_artificial_terrain
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_stack, get_casted_elevation_difference, clean_dh, \
    get_hypsometric_elevation_change
from dhdt.postprocessing.solar_tools import \
    make_shadowing, make_shading
window_size = 2**4

from dhdt.generic.test_tools import test_photohypsometric_coupling

test_photohypsometric_refinement(2, (400, 600))

test_photohypsometric_coupling(10, (400, 600))

#

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

conn_file = "conn-test.txt"
for im_folder in im_path:
    dat_path = os.path.join(im_dir, im_folder)

    if not os.path.exists(os.path.join(dat_path, conn_file)):
        # get meta-data
        zn, az = read_mean_sun_angles_s2(dat_path, fname='MTD_TL.xml')
        Shw_t = create_shadow_caster_casted(Z, geoTransformM, az, zn,
                    dat_path, bbox=np.concatenate((subset_i, subset_j)),
                    out_name=conn_file)

print('conn files made')
dh = read_conn_files_to_stack(im_path, conn_file=conn_file, folder_path=im_dir)
print('data loaded')

dh = clean_dh(dh)
dxyt = get_casted_elevation_difference(dh)
print('hypsometric pairs calculated')

dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransformZ)
print('dhdt created!')

print(np.quantile(dhdt['dZ_12'],[0.25, 0.5, 0.75]))
plt.figure(), plt.hist(dhdt['dZ_12'], bins=400), plt.show()
