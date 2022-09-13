import os
import numpy as np

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image, read_geo_info
from dhdt.generic.mapping_tools import get_mean_map_location
from dhdt.input.read_sentinel2 import list_central_wavelength_msi
from dhdt.preprocessing.atmospheric_geometry import get_refraction_angle
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, update_caster_elevation

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

C_file = "conn.txt"
R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
im_paths = ('S2-2021-10-27',
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

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd'] == 10]
central_wavelength = s2_df['center_wavelength'].to_numpy().mean() # [nm]

h = np.linspace(0, np.round(np.max(Z) / 100) * 100 + 100, 10)

simple_refraction = True

for im_path in im_paths:
    # get roughly the central location of the image location
    spatialRef,geoTransform,_,_,_,_ = read_geo_info(os.path.join(im_dir,
         im_path, 'S.tif'))
    x_bar, y_bar = get_mean_map_location(geoTransform, spatialRef)

    dh = read_conn_files_to_df((im_path,), folder_path=im_dir,
                               dist_thres=100.)
    dh = (dh
          .pipe(start_pipeline)
          .pipe(clean_dh)
          .pipe(update_caster_elevation, Z, geoTransform)
          .pipe(get_refraction_angle, x_bar, y_bar, spatialRef, h,
                                         central_wavelength,
                                         simple_refraction=simple_refraction)
          )
print('.')