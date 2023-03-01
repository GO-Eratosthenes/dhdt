import os
import numpy as np

from dhdt.input.read_sentinel2 import \
    read_stack_s2, get_flight_path_s2, read_view_angles_s2, \
    get_timing_mask
from dhdt.input.read_sentinel2 import list_central_wavelength_msi
from dhdt.generic.handler_sentinel2 import \
    get_s2_image_locations, get_s2_granule_id, get_s2_dict
from dhdt.preprocessing.image_transforms import \
    general_midway_equalization
from dhdt.generic.mapping_io import read_geo_image

from dhdt.presentation.image_io import output_image

base_dir = '/Users/Alten005/MAIO/Atmosphere'
sname = ('S2A_MSIL1C_20191120T101321_N0208_R022_T31SGR_20191120T104456.SAFE',
         'S2B_MSIL1C_20191006T101029_N0208_R022_T31SGR_20191006T135143.SAFE')
fname = 'MTD_MSIL1C.xml'
full_path = os.path.join(base_dir, sname[0], fname)
s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==10]
s2_1, datastrip_id = get_s2_image_locations(full_path,s2_df)

im_stack, spatialRef, geoTransform, targetprj = read_stack_s2(s2_1)
