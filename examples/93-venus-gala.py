import os
import numpy as np
import geopandas

from dhdt.input.read_venus import list_central_wavelength_vssc, \
    list_platform_metadata_vn, read_stack_vn
from dhdt.generic.handler_venus import get_vn_image_locations, \
    get_vn_mean_view_angles

dat_dir = '/Users/Alten005/Galapagos/Data/'
ven_dir = 'VENUS-XS_20221218-163112-000_L1C_GALAPAGO_C_V3-1'

full_path = os.path.join(dat_dir, ven_dir)

vn_df = list_central_wavelength_vssc()
vn_df = get_vn_image_locations(full_path, vn_df)
vn_df = get_vn_mean_view_angles(full_path,vn_df)

im_stack = read_stack_vn(vn_df)[0]
print('.')
#s2_dict = get_s2_dict(s2_df)