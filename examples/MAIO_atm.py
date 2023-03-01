
import os
import numpy as np

from dhdt.input.read_sentinel2 import \
    read_stack_s2, list_central_wavelength_msi
from dhdt.input.read_landsat import \
    list_central_wavelength_oli, read_stack_l8
from dhdt.generic.handler_sentinel2 import \
    get_s2_image_locations, get_s2_granule_id, get_s2_dict
from dhdt.generic.handler_landsat import get_l8_image_locations
from dhdt.preprocessing.image_transforms import \
    general_midway_equalization
from dhdt.generic.mapping_io import read_geo_image

from dhdt.presentation.image_io import output_image

base_dir = '/Users/Alten005/MAIO/Atmosphere'
sname = ('S2B_MSIL1C_20220930T100729_N0400_R022_T33UWA_20220930T121519.SAFE',
         'S2A_MSIL1C_20191120T101321_N0208_R022_T31SGR_20191120T104456.SAFE',
         'S2B_MSIL1C_20191006T101029_N0208_R022_T31SGR_20191006T135143.SAFE')
fname = 'MTD_MSIL1C.xml'
full_path = os.path.join(base_dir, sname[0], fname)
s2_df = list_central_wavelength_msi()
#s2_df = s2_df[s2_df['common_name'].str.contains("swir")]
s2_df = s2_df[s2_df['gsd']==20]
s2_df, datastrip_id = get_s2_image_locations(full_path,s2_df)

im_stack, spatialRef, geoTransform, targetprj = read_stack_s2(s2_df)


I_sub = im_stack[840:1040,1210:1410,:]

I_h = general_midway_equalization(I_sub[...,-2:])
I_n = np.divide(I_h[...,-2]-I_h[...,-1],I_h[...,-2]+I_h[...,-1])
I_d = I_h[...,-2]-I_h[...,-1]


I_m = np.logical_or(I_sub[...,-2]<1030, I_sub[...,-1]<1012)

I_p = np.ma.array(I_d, mask=I_m)
plt.imshow(I_p, vmin=-100, vmax=+100), plt.colorbar()

##
sname = ('S2A_MSIL1C_20191120T101321_N0208_R022_T31SGR_20191120T104456.SAFE',
         'S2B_MSIL1C_20191006T101029_N0208_R022_T31SGR_20191006T135143.SAFE')
fname = 'MTD_MSIL1C.xml'
full_path = os.path.join(base_dir, sname[0], fname)
s2_df = list_central_wavelength_msi()
#s2_df = s2_df[s2_df['common_name'].str.contains("swir")]
s2_df = s2_df[s2_df['gsd']==20]
s2_1, datastrip_id = get_s2_image_locations(full_path,s2_df)

im_stack, spatialRef, geoTransform, targetprj = read_stack_s2(s2_1)

#import matplotlib.pyplot as plt
#plt.figure(), plt.imshow(im_stack[4601:4800,3651:3850,1], cmap=plt.cm.gray)

I1_sub = im_stack[4550:4750,3650:3850,:]

full_path = os.path.join(base_dir, sname[1], fname)
s2_2, datastrip_id = get_s2_image_locations(full_path,s2_df)
im_stack, spatialRef, geoTransform, targetprj = read_stack_s2(s2_2)

I2_sub = im_stack[4550:4750,3650:3850,:]

I_h11 = general_midway_equalization(np.dstack((I1_sub[...,-2], I2_sub[...,-2])))
I_h12 = general_midway_equalization(np.dstack((I1_sub[...,-1], I2_sub[...,-1])))
I_h = (I_h12[...,1]-I_h12[...,0]) - (I_h11[...,1]-I_h11[...,0])

from dhdt.preprocessing.shadow_transforms import pca, ica

#i1,i2,i3,i4 = ica(I1_sub[...,-2], I1_sub[...,-1],
#                  I2_sub[...,-2], I2_sub[...,-1], min_samp=1e4)

i1,i2,i3,i4 = ica(I_h12[...,1], I_h12[...,0],
                  I_h11[...,1], I_h11[...,0], min_samp=1e4)



I1_h = general_midway_equalization(I1_sub)


# landsat
I1_m = np.mean(I1_h,axis=2)
I1_m = np.mean(I1_h[...,:4],axis=2)

I_m11 = np.mean(I_h11,axis=2)
I_m11 = np.mean(I_h11[...,:4],axis=2)



l8_df = list_central_wavelength_oli()
IN = np.zeros((11), dtype=bool)
IN[1:7] = True
l8_df = l8_df[IN]

im_stack, spatialRef, geoTransform, targetprj = read_stack_l8(base_dir, l8_df)

I_sub = im_stack[2500:2700,3250:3450,:]
I_sub = im_stack[2400:2800,3150:3550,:]


I_h = general_midway_equalization(I_sub)
# I_n = np.divide(I_h[...,-2]-I_h[...,-1],I_h[...,-2]+I_h[...,-1])

