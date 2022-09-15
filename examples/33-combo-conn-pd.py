import os
import numpy as np

import matplotlib.pyplot as plt

from dhdt.generic.debugging import start_pipeline
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, get_casted_elevation_difference, \
    get_hypsometric_elevation_change

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
im_dir = os.path.join(base_dir, 'Sentinel-2')
R_dir = os.path.join(base_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

C_file = "conn.txt"
R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
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

dh = read_conn_files_to_df(im_paths, folder_path=im_dir,
                           dist_thres=100.)
dh = (dh
      .pipe(start_pipeline)
      .pipe(clean_dh)
      )

#plt.figure(), plt.scatter(dh['caster_X'], dh['caster_Y'])

dxyt = get_casted_elevation_difference(dh)
print('hypsometric pairs calculated')

dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransformZ)
