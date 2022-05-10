import os
import numpy as np

from dhdt.generic.mapping_io import \
    read_geo_info, read_geo_image, make_geo_im
from dhdt.generic.mapping_tools import pix2map, map2pix
from dhdt.generic.handler_sentinel2 import \
    get_S2_image_locations, meta_S2string, get_s2_dict
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, read_stack_s2, \
    read_detector_mask, read_view_angles_s2, read_sun_angles_s2, \
    s2_dn2toa
from dhdt.preprocessing.shadow_transforms import \
    entropy_shade_removal
from dhdt.postprocessing.solar_tools import \
    make_shading, make_shadowing
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope, compensate_ortho_offset
from dhdt.processing.coupling_tools import match_pair
from dhdt.processing.matching_tools import \
    get_coordinates_of_template_centers


boi = ['red', 'green', 'blue', 'nir']
window_size = 2**4

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/'
s2_dir = os.path.join(base_dir, 'Sentinel-2')
Z_dir = os.path.join(base_dir, 'Cop-DEM_GLO-30')

im_path = ('S2A_MSIL1C_20201019T213531_N0209_R086_T05VMG_20201019T220042.SAFE',)
# 'S2B_MSIL1C_20210405T214529_N0300_R129_T05VMG_20210405T232759.SAFE',
#            'S2A_MSIL1C_20211027T214621_N0301_R129_T05VMG_20211027T220333.SAFE',
# 'S2A_MSIL1C_20191025T213601_N0208_R086_T05VMG_20191025T230149.SAFE',
#            'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE',
# 'S2B_MSIL1C_20171020T213519_N0205_R086_T05VMG_20171020T213515.SAFE',
# 'S2A_MSIL1C_20171018T214531_N0205_R129_T05VMG_20171018T214529.SAFE',
# 'S2B_MSIL1C_20201106T214709_N0209_R129_T05VMG_20201106T220311.SAFE',
#            'S2B_MSIL1C_20201014T213529_N0209_R086_T05VMG_20201014T220451.SAFE',
#
Z_file = "COP-DEM-05VMG.tif"
(Z, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(Z_dir, Z_file))

sample_I, sample_J = get_coordinates_of_template_centers(Z, window_size)

for im_folder in im_path:
    im_dir = os.path.join(s2_dir, im_folder)

    s2_df = list_central_wavelength_msi()
    s2_df = s2_df[s2_df['common_name'].isin(boi)]
    s2_df, datastrip_id = get_S2_image_locations(os.path.join(im_dir,
          'MTD_MSIL1C.xml'), s2_df)
    s2_dict = get_s2_dict(s2_df)

    S2time, S2orbit, S2tile = meta_S2string(im_folder)

    print('reading imagery')
    im_stack, spatialRef, geoTransform, targetprj = read_stack_s2(s2_df)
    im_stack = s2_dn2toa(im_stack)

    print('reading spatial metadata at pixel level')
    # get sun angles
    sun_zn, sun_az = read_sun_angles_s2(s2_dict['MTD_TL_path'])

    # get sensor configuration
    det_stack = read_detector_mask(s2_dict['QI_DATA_path'], s2_df, geoTransform)

    # get sensor viewing angles
    view_zn, view_az = read_view_angles_s2(s2_dict['MTD_TL_path'],
                                           fname='MTD_TL.xml',
                                           det_stack=det_stack, boi=s2_df)
    view_zn, view_az = view_zn[:,:,2], view_az[:,:,2]

    # create shadow enhanced imagery
    idx_blue = np.flatnonzero(s2_df['common_name']=='blue')[0]
    idx_red = np.flatnonzero(s2_df['common_name']=='red')[0]
    idx_nir = np.flatnonzero(s2_df['common_name'] == 'nir')[0]
    Si,Ri = entropy_shade_removal(im_stack[:,:,idx_blue], im_stack[:,:,idx_red],
                                  im_stack[:, :, idx_nir], a=138)
    del im_stack

    # construct shadowing and shading
    print('creating shading')
#    Shw = make_shadowing(Z, sun_az, sun_zn)
    Shd = make_shading(Z, sun_az, sun_zn)

    Stable = Z!=0

    # matching
    print('start matching')
    sample_X, sample_Y = pix2map(geoTransform, sample_I, sample_J)
    match_X, match_Y, match_score = match_pair(Shd, Si,
                                               Stable, Stable,
                                               geoTransform, geoTransform,
                                               sample_X, sample_Y,
                                               temp_radius=window_size,
                                               search_radius=window_size,
                                               correlator='phas_only',
                                               subpix='moment',
                                               metric='peak_entr')  # 'robu_corr'

    dY, dX = sample_Y - match_Y, sample_X - match_X
    IN = ~np.isnan(dX)

    Slp, Asp = get_template_aspect_slope(Z,
                                         sample_I, sample_J,
                                         window_size)

    PURE = np.logical_and(IN, Slp<20)
    dx_coreg, dy_coreg = np.median(dX[PURE]), np.median(dY[PURE])
    print(dx_coreg, dy_coreg)

    Si_ort = compensate_ortho_offset(Si, Z, dx_coreg, dy_coreg,
                                     view_az, view_zn, geoTransform)
    print('scene compensated')

    # write out results
    make_geo_im(Si, geoTransform, spatialRef,
                os.path.join(base_dir, 'shade-raw-'+S2time[1:]+'.tif'))
    make_geo_im(Si_ort, geoTransform, spatialRef,
                os.path.join(base_dir, 'shade-cor-2-'+S2time[1:]+'.tif'))
    print('scene written')

print('done processing')