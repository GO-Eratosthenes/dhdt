import os
import numpy as np

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image, read_geo_info
from dhdt.generic.mapping_tools import get_mean_map_location
from dhdt.input.read_sentinel2 import list_central_wavelength_msi
from dhdt.preprocessing.atmospheric_geometry import \
    get_refraction_angle, update_list_with_corr_zenith_pd
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, update_caster_elevation, \
    update_glacier_id

#test_refraction_calculation()


base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/coreg/'
gis_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'

R_dir = os.path.join(gis_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(gis_dir, 'Cop-DEM_GLO-30')

S_file = 'shadow.tif'
C_file = "conn.txt"
R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))
(R, spatialRefR, geoTransformR, targetprjR) = \
    read_geo_image(os.path.join(R_dir, R_file))

im_paths = ('S2_2017-10-20', 'S2_2020-10-19')

i_bias = int((geoTransformR[3]-geoTransformZ[3])/geoTransformR[5])
j_bias = int((geoTransformR[0]-geoTransformZ[0])/geoTransformR[1])

Z = Z[i_bias:i_bias+geoTransformR[-2],j_bias:j_bias+geoTransformR[-1]]


s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd'] == 10]
central_wavelength = s2_df['center_wavelength'].to_numpy().mean() # [nm]

h = np.linspace(0, np.round(np.max(Z) / 100) * 100 + 100, 10)

for im_path in im_paths:
    # get roughly the central location of the image location
    spatialRef,geoTransform = read_geo_info(os.path.join(base_dir,
         im_path, S_file))[:2]
    x_bar, y_bar = get_mean_map_location(geoTransform, spatialRef)

    dh = read_conn_files_to_df((im_path,), folder_path=base_dir,
                               dist_thres=100.)
    dh = (dh
          .pipe(start_pipeline)
          .pipe(clean_dh)
          .pipe(update_caster_elevation, Z, geoTransform)
          .pipe(update_glacier_id, R, geoTransform)
          .pipe(get_refraction_angle, x_bar, y_bar, spatialRef,
                central_wavelength, h)
          )
    update_list_with_corr_zenith_pd(dh, os.path.join(base_dir,im_path))
    print('.')
print('.')