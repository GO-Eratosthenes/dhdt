import os
import numpy as np

import matplotlib.pyplot as plt

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, get_casted_elevation_difference, \
    get_hypsometric_elevation_change, update_casted_elevation, \
    update_glacier_id, clean_dhdt
from dhdt.postprocessing.group_statistics import get_general_hypsometry

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

C_file = "conn.txt"
R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))
(R, spatialRefM, geoTransformM, targetprjM) = \
    read_geo_image(os.path.join(R_dir, R_file))

im_paths = ('S2-2021-10-27',
           'S2-2021-04-05',
           'S2-2020-10-19',
           'S2-2017-10-18',
           'S2-2020-10-14',
           'S2-2020-11-06',
           'S2-2019-10-25',
           'S2-2019-10-15',
           'S2-2017-10-20',
           'S2-2017-10-18')
im_paths = ('S2-2019-10-25', 'S2-2019-10-15',)

dh = read_conn_files_to_df(im_paths, folder_path=im_dir, dist_thres=10.)
# preprocessing
dh = (dh
      .pipe(start_pipeline)
      .pipe(update_glacier_id, R, geoTransformM)
      .pipe(clean_dh)
      )

#plt.figure(), plt.scatter(dh['caster_X'], dh['caster_Y'])
# processing
dxyt = (dh
        .pipe(start_pipeline)
        .pipe(get_casted_elevation_difference)
        .pipe(update_casted_elevation, Z, geoTransformZ)
        )

dhdt = (dxyt
        .pipe(get_hypsometric_elevation_change, dxyt)
        .pipe(clean_dhdt)
        )

dhdt_sel = dhdt[dhdt['G_1']==dhdt['G_1'].unique()[3]]
elev,q_050,n_050 = get_general_hypsometry(dhdt_sel['Z_12'].to_numpy(),
                                          dhdt_sel['dZ_12'].to_numpy(),
                                          quant=.5)

print('hypsometric pairs calculated')



