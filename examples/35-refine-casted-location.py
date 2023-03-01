import os
import numpy as np

import matplotlib.pyplot as plt

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import get_mean_map_location
from dhdt.preprocessing.atmospheric_geometry import \
    get_refraction_angle, update_list_with_corr_zenith_pd
from dhdt.processing.photohypsometric_image_refinement import \
    update_casted_location, update_list_with_refined_casted
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, get_casted_elevation_difference, \
    get_hypsometric_elevation_change, update_casted_elevation, \
    update_glacier_id, clean_dxyt, read_conn_to_df, read_conn_files_to_stack
from dhdt.postprocessing.group_statistics import get_general_hypsometry

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/coreg/'
C_file = "conn.txt"
S_file = "shadow.tif"
A_file = "albedo.tif"
M_file = "classification.tif"
Z_file = "COP-DEM-05VMG.tif"
conn_new = 'conn_new.txt'

Z_dir = os.path.join('/'.join(base_dir.split('/')[:-3]), 'Cop-DEM_GLO-30')
(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))

extent = 120 # amount of slack

im_path = ('S2_2019-10-15', 'S2_2020-10-19')

for idx in range(len(im_path)):
    if os.path.exists(os.path.join(base_dir, im_path[idx],conn_new)):
        continue

    # read location
    dh = read_conn_to_df(os.path.join(base_dir, im_path[idx], C_file))

    # read imagery
    S, spatialRef, geoTransform, targetprj = read_geo_image(
        os.path.join(base_dir, im_path[idx], S_file))
    A = read_geo_image(os.path.join(base_dir, im_path[idx], A_file))[0]
    M = read_geo_image(os.path.join(base_dir, im_path[idx], M_file))[0]
    print('.')

    if not 'zenith_refrac' in dh.columns:
        x_bar, y_bar = get_mean_map_location(geoTransform, spatialRef)
        dh = update_casted_location(dh, S, M, geoTransform, extent=extent)

    dh_new = update_casted_location(dh, S, M, geoTransform, extent=extent)

    update_list_with_refined_casted(dh_new, os.path.join(base_dir, im_path[idx]),
                                    file_name=conn_new)

    print('.')
# random allocation
#cnt = np.random.randint(dh.shape[0], size=1)[0] # 16666 #

#for cnt in range(dh.shape[0]):
#    x, y = dh.iloc[cnt]['casted_X'], dh.iloc[cnt]['casted_Y']
#    x_t, y_t = dh.iloc[cnt]['caster_X'], dh.iloc[cnt]['caster_Y']
#    az = dh.iloc[cnt]['azimuth']

#    x_new,y_new,p_std = refine_cast_location(S, M, geoTransform, x,y, az,x_t,y_t,
#                                             extent=120)

print('.')

dh = read_conn_files_to_df(im_path, folder_path=base_dir,
                           conn_file='conn_new.txt')
print('data loaded')

dh = clean_dh(dh)
dxyt = get_casted_elevation_difference(dh)
print('hypsometric pairs calculated')

# y,A = get_relative_hypsometric_network(dxyt, Z, geoTransformZ)

dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransformZ)

print('dhdt created!')

dt = dhdt['T_2']-dhdt['T_1']
dzdt = np.divide(dhdt['dZ_12']*365.25, dt/np.timedelta64(1, 'Y'))

#plt.figure(), plt.hexbin(dhdt['Z_12'], dzdt, extent=(0, 1000, -2, +2), gridsize=30)

elev,q_050,n_050 = get_general_hypsometry(dhdt['Z_12'], dzdt, quant=.5)
print('dhdt created!')


