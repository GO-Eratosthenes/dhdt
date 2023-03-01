import os
import glob
import numpy as np

from osgeo import gdal

from dhdt.generic.mapping_io import read_geo_image, make_geo_im
from dhdt.preprocessing.image_transforms import gamma_adjustment, mat_to_gray
from dhdt.input.read_sentinel2 import list_central_wavelength_msi
from dhdt.generic.handler_sentinel2 import get_S2_image_locations
from dhdt.presentation.image_io import output_image

dat_dir = '/Users/Alten005/SEES/Data/S2'
im_list = ['S2A_MSIL1C_20220709T122711_N0400_R052_T33XWF_20220709T175050',
           'S2A_MSIL1C_20220709T122711_N0400_R052_T33XWG_20220709T175050',
           'S2A_MSIL1C_20220709T122711_N0400_R052_T33XWH_20220709T175050',
           'S2A_MSIL1C_20220710T115651_N0400_R066_T33XXG_20220710T154125',
           'S2A_MSIL1C_20220710T115651_N0400_R066_T33XXG_20220710T172110',
           'S2A_MSIL1C_20220710T115651_N0400_R066_T33XXH_20220710T154125']

boi = [2, 3, 4, 8]

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['common_name'].isin(boi)]
vrt_name = 'multi_spec.vrt'

bois = [4, 3, 2]
for im_name in im_list:
    for boi in bois:
        fname = glob.glob(os.path.join(dat_dir, im_name, 'analytic',
                                       '*0'+str(boi)+'.jp2'))[0]
        data, spatialRef, geoTransform, targetprj  = read_geo_image(fname)
        np.putmask(data, data==0, np.unique(data)[1])
        d_g = gamma_adjustment(data, 0.75)
        d_m = mat_to_gray(d_g, data==0)
        d_8 = np.round(d_m*256).astype(np.uint8)
        if boi==bois[0]:
            imstack = d_8[...,np.newaxis].copy()
        else:
            imstack = np.dstack((imstack, d_8[...,np.newaxis]))

    output_image(imstack, os.path.join(dat_dir, im_name,
                 fname.split('/')[-1][:15])+'.jpg' )

#    make_geo_im(imstack,geoTransform,spatialRef, os.path.join(dat_dir, im_name,
#                fname.split('/')[-1][:15])+'.tif' ))
    print('*')


for im_name in im_list:
#    b2_path = os.path.join(dat_dir, im_name, 'analytic', '*02.jp2')
    b3_path = os.path.join(dat_dir, im_name, 'analytic', '*03.jp2')
    b4_path = os.path.join(dat_dir, im_name, 'analytic', '*04.jp2')
    b8_path = os.path.join(dat_dir, im_name, 'analytic', '*08.jp2')

    band_list = [glob.glob(b2_path)[0], glob.glob(b3_path)[0],
                 glob.glob(b4_path)[0], glob.glob(b8_path)[0]]

    ffull = os.path.join(dat_dir, im_name, vrt_name)
    vrt_options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour,
                                       addAlpha=False,
                                       separate=True,
                                       srcNodata=0)
    my_vrt = gdal.BuildVRT(ffull, band_list,
                           options=vrt_options)
    my_vrt = None

