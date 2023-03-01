import os
import numpy as np

from dhdt.generic.mapping_io import \
    read_geo_info, read_geo_image, make_geo_im, make_multispectral_vrt
from dhdt.generic.mapping_tools import make_same_size
from dhdt.generic.handler_sentinel2 import \
    get_S2_image_locations, meta_s2string
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, \
    read_detector_mask, read_view_angles_s2, read_sun_angles_s2

s2_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'

boi = ['red', 'green', 'blue', 'nir']

sun_name = 'sun.tif'
det_name = 'det.tif'
view_name = 'view.tif'

# get wanted resolution
ref_path = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'+
                        'Sentinel-2/S2-15-10-2019',
                        'B2.tif')
spatialRef,refTransform,targetprj,rows,cols,_ = read_geo_info(ref_path)

im_path = ('S2A_MSIL1C_20211027T214621_N0301_R129_T05VMG_20211027T220333.SAFE',
           'S2B_MSIL1C_20210405T214529_N0300_R129_T05VMG_20210405T232759.SAFE',
           'S2A_MSIL1C_20191025T213601_N0208_R086_T05VMG_20191025T230149.SAFE',
           'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE',
           'S2B_MSIL1C_20171020T213519_N0205_R086_T05VMG_20171020T213515.SAFE',
           'S2A_MSIL1C_20171018T214531_N0205_R129_T05VMG_20171018T214529.SAFE',
           'S2B_MSIL1C_20201106T214709_N0209_R129_T05VMG_20201106T220311.SAFE',
           'S2B_MSIL1C_20201014T213529_N0209_R086_T05VMG_20201014T220451.SAFE',
           'S2A_MSIL1C_20201019T213531_N0209_R086_T05VMG_20201019T220042.SAFE')

for im_folder in im_path:
    im_dir = os.path.join(s2_dir, im_folder)

    s2_df = list_central_wavelength_msi()
    s2_df = s2_df[s2_df['common_name'].isin(boi)]
    s2_df, datastrip_id = get_S2_image_locations(os.path.join(im_dir,
          'MTD_MSIL1C.xml'), s2_df)


    S2time, S2orbit, S2tile = meta_s2string(im_folder)
    # make folder
    im_dest = os.path.join(s2_dir, 'S2-'+S2time[1:])
    if not os.path.exists(im_dest):
        os.makedirs(im_dest)

    # copy meta-data
    meta_dir = '/'.join(s2_df['filepath'][0].split('/')[0:-2])
    meta_path = os.path.join(meta_dir, 'MTD_TL.xml')
    meta_dump = os.path.join(im_dest, 'MTD_TL.xml')
    os.system('cp '+meta_path+' '+meta_dump)

    meta_path = os.path.join(im_dir, 'MTD_MSIL1C.xml')
    meta_dump = os.path.join(im_dest, 'MTD_MSIL1C.xml')
    os.system('cp '+meta_path+' '+meta_dump)

    print('reading imagery and adjusting its size')
    for band_path in s2_df['filepath']:
        # read full scene
        (I,_,geoTransform,_) = read_geo_image(band_path+'.jp2')

        sub_I = make_same_size(I,geoTransform, refTransform, rows, cols)
        band_name = band_path[-3:]+'.tif'
        # export to map
        make_geo_im(sub_I, refTransform, spatialRef,
                    os.path.join(im_dest, band_name))

    make_multispectral_vrt(im_dest, s2_df,fname='test.vrt')
    # make meta-data imagery
    print('reading spatial metadata at pixel level')
    # get sun angles
    split_path = s2_df['filepath'][0].split('/')  # get location of imagery meta data
    path_meta = os.path.join(*split_path[:-2])
    if s2_df['filepath'][0][0] == '/':  # if absolute path, include dealing slash
        path_meta = ''.join(('/', path_meta))
    (sun_zn, sun_az) = read_sun_angles_s2(path_meta)
    sun_ang = make_same_size(np.dstack((sun_zn, sun_az)),geoTransform,
                             refTransform, rows, cols)
    make_geo_im(sun_ang, refTransform, spatialRef,
                os.path.join(im_dest, sun_name))

    # get sensor configuration
    det_meta = os.path.join(path_meta, 'QI_DATA')
    det_stack = read_detector_mask(det_meta, s2_df, geoTransform)
    det_id = make_same_size(det_stack, geoTransform, refTransform, rows, cols)
    make_geo_im(det_id, refTransform, spatialRef,
                os.path.join(im_dest, det_name))


    # get sensor viewing angles
    view_zn, view_az = read_view_angles_s2(path_meta, fname='MTD_TL.xml',
                                           det_stack=det_stack, boi=s2_df)
    del det_stack, det_id
    view_ang = make_same_size(np.dstack((view_zn, view_az)), geoTransform,
                              refTransform, rows, cols)
    make_geo_im(view_ang, refTransform, spatialRef,
                os.path.join(im_dest, view_name))
    print('done with: '+S2time[1:])