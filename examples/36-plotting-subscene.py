import os

import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import \
    read_geo_image, read_geo_info, make_geo_im
from dhdt.generic.mapping_tools import pix2map, get_bbox, map2pix

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
im_path = ('S2-2019-10-25',
           'S2-2021-04-05',
           'S2-2020-10-14',
           'S2-2020-11-06',
           'S2-2021-10-27',
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

