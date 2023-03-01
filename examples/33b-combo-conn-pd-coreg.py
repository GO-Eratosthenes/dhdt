import os
import numpy as np

import matplotlib.pyplot as plt

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.preprocessing.atmospheric_geometry import \
    get_refraction_angle, update_list_with_corr_zenith_pd
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, get_casted_elevation_difference, \
    get_hypsometric_elevation_change, update_casted_elevation, \
    update_glacier_id, clean_dxyt
from dhdt.postprocessing.group_statistics import get_general_hypsometry

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/coreg/'
gis_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'

R_dir = os.path.join(gis_dir, 'Cop-DEM-GLO-30')
Z_dir = os.path.join(gis_dir, 'Cop-DEM_GLO-30')

C_file = "conn.txt"
R_file = "5VMG_RGI_red.tif"
Z_file = "COP-DEM-05VMG.tif"
(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))
(R, spatialRefM, geoTransformM, targetprjM) = \
    read_geo_image(os.path.join(R_dir, R_file))

im_path = ('S2_2017-10-20', 'S2_2020-10-19')
from dhdt.preprocessing.shadow_geometry import shadow_image_to_suntrace_list

M,_,geoTransformM,_ = \
    read_geo_image(os.path.join(base_dir, im_path[0], 'classification.tif'))

aap = shadow_image_to_suntrace_list(M, geoTransformM, 146)

dh = read_conn_files_to_df(im_path, folder_path=base_dir,
                           dist_thres=100.)





# there is an error in the conn-files... open issue?


# preprocessing
dh = (dh
      .pipe(start_pipeline)
      .pipe(update_glacier_id, R, geoTransformZ)
      .pipe(clean_dh)
      )

# refine: mask, weighted_normalized_cross_correlation

# via test function...? + noise, fading,...
# compare coordinate list...?
# -> shadow enhancement: L0 -> make_boundary
# affine warping....?

# random location to show... so debugging np.random.randi m-h n-w

# distribution test....? for autumn scene
# compare against synthetic shading from ArcticDEM, two coloured histogram

# post-processing....: filtering....?

#plt.figure(), plt.scatter(dh['caster_X'], dh['caster_Y'])



# processing
dxyt = (dh
        .pipe(start_pipeline)
        .pipe(get_casted_elevation_difference)
        .pipe(update_casted_elevation, Z, geoTransformZ)
        )

dhdt = (dxyt
        .pipe(clean_dxyt)
        .pipe(get_hypsometric_elevation_change, dxyt)
        )

dhdt_sel = dhdt[dhdt['G_1']==dhdt['G_1'].unique()[1]]
elev,q_050,n_050 = get_general_hypsometry(dhdt_sel['Z_12'].to_numpy(),
                                          dhdt_sel['dZ_12'].to_numpy(),
                                          quant=.5)

print('hypsometric pairs calculated')



