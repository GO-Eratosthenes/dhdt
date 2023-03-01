import os
import glob
import numpy as np

from osgeo import gdal

from dhdt.generic.mapping_io import read_geo_image, make_geo_im
from dhdt.preprocessing.image_transforms import gamma_adjustment, mat_to_gray
from dhdt.input.read_sentinel2 import list_central_wavelength_msi, \
    read_stack_s2
from dhdt.generic.handler_sentinel2 import get_S2_image_locations
from dhdt.presentation.image_io import output_image

dat_dir = '/Users/Alten005/HLR/2018'
yoi = int(dat_dir.split('/')[-1]) # year of interest

if yoi==2016:
    im_list = ['S2A_MSIL1C_20160624T103022_N0204_R108_T32TLR_20160624T103721.SAFE',
               'S2A_MSIL1C_20160624T103022_N0204_R108_T32TLS_20160624T103721.SAFE',
               'S2A_MSIL1C_20160624T103022_N0204_R108_T32TMR_20160624T103721.SAFE',
               'S2A_MSIL1C_20160624T103022_N0204_R108_T32TMS_20160624T103721.SAFE']
elif yoi==2018:
    im_list = ['S2B_MSIL1C_20180709T103019_N0206_R108_T32TLR_20180709T155649.SAFE',
               'S2B_MSIL1C_20180709T103019_N0206_R108_T32TLS_20180709T155649.SAFE',
               'S2B_MSIL1C_20180709T103019_N0206_R108_T32TMR_20180709T155649.SAFE',
               'S2B_MSIL1C_20180709T103019_N0206_R108_T32TMS_20180709T155649.SAFE']
elif yoi==2020:
    im_list = ['S2B_MSIL1C_20200708T102559_N0209_R108_T32TLR_20200708T124214.SAFE',
               'S2B_MSIL1C_20200708T102559_N0209_R108_T32TLS_20200708T124214.SAFE',
               'S2B_MSIL1C_20200708T102559_N0209_R108_T32TMR_20200708T124214.SAFE',
               'S2B_MSIL1C_20200708T102559_N0209_R108_T32TMS_20200708T124214.SAFE']
elif yoi==2022:
    im_list = ['S2B_MSIL1C_20220708T102559_N0400_R108_T32TLR_20220708T123203.SAFE',
               'S2B_MSIL1C_20220708T102559_N0400_R108_T32TLS_20220708T123203.SAFE',
               'S2B_MSIL1C_20220708T102559_N0400_R108_T32TMR_20220708T123203.SAFE',
               'S2B_MSIL1C_20220708T102559_N0400_R108_T32TMS_20220708T123203.SAFE']

noi = ['red','green','blue']

vrt_name = 'multi_spec.vrt'
xml_name = 'MTD_MSIL1C.xml'

bois = [4, 3, 2]
for im_name in im_list:

    s2_df = list_central_wavelength_msi()
    s2_df = s2_df[s2_df['common_name'].isin(noi)]
    full_path = os.path.join(dat_dir, im_name, xml_name)
    s2_df, datastrip_id = get_S2_image_locations(full_path, s2_df)

    data, spatialRef, geoTransform, targetprj = read_stack_s2(s2_df)
    data = data[...,::-1] # bring colorscale correctly

    data[0,0,:], data[-1,-1,:] = 1, 2**14
    d_m = mat_to_gray(data, data==0, vmin=1, vmax=2**14)
    if yoi==2022:
        gm = 0.7
    else:
        gm = 0.5
    d_g = gamma_adjustment(d_m, gm)
    d_8 = np.round(d_g * 256).astype(np.uint8)

    make_geo_im(d_8, geoTransform, spatialRef,
                os.path.join(dat_dir, im_name[:-5]) + '.tif')
    print('*')


files_to_mosaic = []
for im_name in im_list:
    im_full = os.path.join(dat_dir, im_name[:-5]+'.tif')
    files_to_mosaic.append(im_full)

g = gdal.Warp(os.path.join(dat_dir, "mosa.tif"), files_to_mosaic, format="GTiff",
              options=["COMPRESS=LZW", "TILED=NO"]) # if you want
g = None # Close file and flush to disk

for tile_path in files_to_mosaic:
    os.remove(tile_path)

root_dir = '/'+os.path.join(*dat_dir.split('/')[:-1])
data = read_geo_image(os.path.join(dat_dir, "mosa.tif"))[0]
output_image(data[8500:13500,2000:12000,:],
             os.path.join(root_dir, 'hlr'+str(yoi)+'.jpg'))
os.remove(os.path.join(dat_dir, "mosa.tif"))
print('+')