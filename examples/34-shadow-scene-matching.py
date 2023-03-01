import os
import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import read_geo_image, read_geo_info
from dhdt.generic.unit_conversion import deg2arg
from dhdt.generic.mapping_tools import \
    map2pix, bilinear_interp_excluding_nodat
from dhdt.input.read_sentinel2 import get_local_bbox_in_s2_tile
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.preprocessing.shadow_filters import \
    fade_shadow_cast, L0_smoothing
from dhdt.preprocessing.shadow_geometry import shadow_image_to_suntrace_list
from dhdt.processing.coupling_tools import match_pair
from dhdt.postprocessing.photohypsometric_tools import read_conn_to_df
from dhdt.postprocessing.mapping_io import casting_pairs_mat2shp
from dhdt.presentation.image_io import output_draw_color_lines_overlay


base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/coreg/'

im_path = ('S2_2017-10-20', 'S2_2020-10-19')

full_path = os.path.join(base_dir, im_path[0])

f_conn = 'conn.txt'
f_sdw = 'shadow.tif'
f_arti = 'shadow_artificial.tif'
f_shp = 'shadow_refine.shp'

# create shadowing
spatialRef, geoTransform, targetprj, rows, cols, bands = read_geo_info(
    os.path.join(full_path, f_arti))
S_0 = read_geo_image(os.path.join(full_path, f_arti))[0]
S_1 = read_geo_image(os.path.join(full_path, f_sdw))[0]
print('imagery read')

# create connectivity
conn_df = read_conn_to_df(os.path.join(full_path, f_conn))
#conn_df.casted_X.values, conn_df.casted_Y.values,
az = conn_df.azimuth.mean()

trace_list = shadow_image_to_suntrace_list(S_0,geoTransform,az)
# plt.figure(), plt.plot(trace_list[::40,0::2].T, trace_list[::40,1::2].T, color='r')

# fading shadowing
S_fade = fade_shadow_cast(S_0, deg2arg(az+180))
S_hard = L0_smoothing(S_1)
print('image processing done')

# refine to imagery
x_new,y_new,score = match_pair(S_fade, S_hard, None, None,
                               geoTransform, geoTransform,
                               trace_list[:,2].T, trace_list[:,3],
                               temp_radius=7, search_radius=22,
                               correlator='norm_corr', subpix='svd',
                               processing='simple',precision_estimate=False,
                               metric='peak_abs')
#
print('image matching done')
xy_0 = np.stack((conn_df.casted_X.values, conn_df.casted_Y.values), axis=1)
#xy_0 = trace_list[:,2:]
xy_1 = np.stack((x_new, y_new), axis=1)
casting_pairs_mat2shp(xy_0,xy_1, os.path.join(full_path, f_shp), spatialRef)
casting_pairs_mat2shp(np.stack((conn_df.caster_X.values, conn_df.caster_Y.values), axis=1),
                      np.stack((conn_df.casted_X.values, conn_df.casted_Y.values), axis=1),
                      os.path.join(full_path, 'conn.shp'), spatialRef)


print('.')







