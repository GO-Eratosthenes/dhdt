import os
import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import \
    read_geo_image
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_stack, get_casted_elevation_difference, clean_dh, \
    get_hypsometric_elevation_change
from dhdt.postprocessing.group_statistics import get_general_hypsometry

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

C_file = "conn.txt"
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

dh = read_conn_files_to_stack(im_path, folder_path=im_dir)
print('data loaded')

dh = clean_dh(dh)
dxyt = get_casted_elevation_difference(dh)
print('hypsometric pairs calculated')

# y,A = get_relative_hypsometric_network(dxyt, Z, geoTransformZ)

dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransformZ)

print('dhdt created!')

dt = dhdt['T_2']-dhdt['T_1']
dzdt = np.divide(dhdt['dZ_12']*365.25, dt.astype(float))

plt.figure(), plt.hexbin(dhdt['Z_12'], dzdt, extent=(0, 1000, -2, +2), gridsize=30)

elev,q_050,n_050 = get_general_hypsometry(dhdt['Z_12'], dzdt, quant=.5)
print('dhdt created!')


