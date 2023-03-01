import os
import numpy as np
import morphsnakes as ms

from scipy import ndimage

from dhdt.generic.mapping_io import \
    read_geo_image, read_geo_info, make_geo_im
from dhdt.generic.mapping_tools import pix2map, get_bbox
from dhdt.generic.gis_tools import get_mask_boundary
from dhdt.generic.handler_im import bilinear_interpolation
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, \
    read_mean_sun_angles_s2, get_local_bbox_in_s2_tile
from dhdt.preprocessing.shadow_transforms import \
    entropy_shade_removal, normalized_range_shadow_index
from dhdt.preprocessing.image_transforms import \
    mat_to_gray, normalize_histogram, gamma_adjustment
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope, slope_along_perp
from dhdt.preprocessing.shadow_geometry import shadow_image_to_list
from dhdt.processing.coupling_tools import match_pair
from dhdt.processing.matching_tools import \
    get_coordinates_of_template_centers
from dhdt.postprocessing.solar_tools import \
    make_shading, make_shadowing
from dhdt.presentation.image_io import \
    output_image, output_glacier_map_background, output_mask, \
    output_cast_lines_from_conn_txt

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'

im_path = ('S2-2019-10-25',
           'S2-2021-10-27',
           'S2-2021-04-05',
           'S2-2020-10-19',
           'S2-2017-10-18',
           'S2-2020-10-14',
           'S2-2020-11-06',
           'S2-2019-10-15',
           'S2-2017-10-20',
           'S2-2017-10-18')

#im_path = ('S2-2020-10-19',
#           'S2-2017-10-18')

counts = 50 # iterations of the snake
window_size = 2**4

boi = ['red', 'green', 'blue', 'nir']
s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['common_name'].isin(boi)]

# import elevation and glacier data
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.getcwd()!=dir_path:
    os.chdir(dir_path) # change to directory where script is situated

(Z,_,geoTransform,crs) = read_geo_image(os.path.join(Z_dir, Z_file))
spatialRef,_,_,rows,cols,_ = read_geo_info(os.path.join(Z_dir, Z_file))
R = read_geo_image(os.path.join(Z_dir, R_file))[0]

#from dhdt.postprocessing.group_statistics import get_normalized_hypsometry
#a = get_normalized_hypsometry(R, Z, np.multiply(Z,np.random.random(Z.shape)),
#                              bins=20)


for im_folder in im_path:
    print('working on '+im_folder)
    im_dir = os.path.join(base_dir, im_folder)
    # create bbox in local Sentinel-2 image frame
    bbox_ij = get_local_bbox_in_s2_tile(os.path.join(im_dir, 'B02.tif'),
                                        im_dir)

    date = np.datetime64(im_folder[3:], 'ns')
    bbox_xy = get_bbox(geoTransform, rows, cols)
    zn, az = read_mean_sun_angles_s2(im_dir, fname='MTD_TL.xml')

    if not os.path.exists(os.path.join(im_dir, "S.tif")):

        # read imagery
        Blue = read_geo_image(os.path.join(im_dir, 'B02.tif'))[0]
        Green = read_geo_image(os.path.join(im_dir, 'B03.tif'))[0]
        Red = read_geo_image(os.path.join(im_dir, 'B04.tif'))[0]
        Near = read_geo_image(os.path.join(im_dir, 'B08.tif'))[0]
        View = read_geo_image(os.path.join(im_dir, 'view.tif'))[0]

        # make RGB imagery
        Sn = normalized_range_shadow_index(mat_to_gray(Red),
                                           mat_to_gray(Green),
                                           mat_to_gray(Blue))
        Sn = normalize_histogram(-Sn)
        RGB = np.dstack((gamma_adjustment(mat_to_gray(Red), gamma=.25),
                         gamma_adjustment(mat_to_gray(Green), gamma=.25),
                         gamma_adjustment(mat_to_gray(Blue), gamma=.25)))

        RGB = mat_to_gray(1.5*RGB)
        RGB[RGB<.3] = .3

        Si,Ri = entropy_shade_removal(mat_to_gray(Blue),
                                      mat_to_gray(Red),
                                      mat_to_gray(Near), a=138)

        # construct shadowing and shading
        Shw = make_shadowing(Z, az, zn)
        Shd = make_shading(Z, az, zn)

        # do coregistration
        Stable = (R == 0) # off-glacier terrain
        sample_I, sample_J = get_coordinates_of_template_centers(Z, window_size)
        sample_X, sample_Y = pix2map(geoTransform, sample_I, sample_J)
        match_X, match_Y, match_score = match_pair(Shd, Si,
                                                   Stable, Stable,
                                                   geoTransform, geoTransform,
                                                   sample_X, sample_Y,
                                                   temp_radius=window_size,
                                                   search_radius=window_size,
                                                   correlator='phas_only',
                                                   subpix='moment',
                                                   metric='peak_entr') # 'robu_corr'

        dY, dX = sample_Y - match_Y, sample_X - match_X
        IN = ~np.isnan(dX)
        IN[IN] = np.remainder(dX[IN], 1) != 0

        Slp, Asp = get_template_aspect_slope(Z,
                                             sample_I, sample_J,
                                             window_size)

        # observation angles...
        Slp_along, Slp_perp = slope_along_perp(Z, View[:,:,4], spac=10)

        PURE = np.logical_and(IN, Slp<20)
        dx_coreg, dy_coreg = np.median(dX[PURE]), np.median(dY[PURE])
        print(dx_coreg, dy_coreg)

        di_coreg = +1 * dy_coreg / geoTransform[5]
        dj_coreg = +1 * dx_coreg / geoTransform[1]

        Sn = bilinear_interpolation(Sn, di_coreg, dj_coreg)
        Si = bilinear_interpolation(Si, di_coreg, dj_coreg)
        Ri = bilinear_interpolation(Ri, di_coreg, dj_coreg)
        RGB= bilinear_interpolation(RGB,di_coreg, dj_coreg)

        # export
        make_geo_im(np.dstack((Si,Ri)), geoTransform, spatialRef,
                    os.path.join(im_dir, "S.tif"),
                    sun_angles='az:'+str(int(az))+'-zn:'+str(int(zn)),
                    date_created='+'+im_folder[3:])
    else:
        SR = read_geo_image(os.path.join(im_dir, "S.tif"))[0]
        Si, Ri = np.squeeze(SR[...,0]), np.squeeze(SR[...,1])
        Shw = make_shadowing(Z, az, zn)

    # estimate polygon
    M_acwe_si = ms.morphological_chan_vese(Si, counts, init_level_set=Shw,
                                          smoothing=0, lambda1=1, lambda2=1,
                                          albedo=Ri)
    M_acwe_op = ndimage.binary_opening(M_acwe_si)
    M_acwe_cl = ndimage.binary_closing(M_acwe_op)
    Bnd = get_mask_boundary(M_acwe_cl)

    if 1==1:
        # output .png s
        output_mask(Bnd, os.path.join(im_dir, 'Red-shadow-polygon-snk-' + im_folder
                                      +'.png'))
        output_mask(get_mask_boundary(Shw), os.path.join(im_dir,
                    'Red-shadow-polygon-dem-' + im_folder +'.png'),
                    color=np.array([153, 51, 255]))
        # output .jpg s
        output_image(Sn, os.path.join(im_dir, 'Red-normalized-range-shadow-index-'
                                      + im_folder + '.jpg'), cmap='gray')
        output_image(normalize_histogram(Si),
                     os.path.join(im_dir, 'Red-entropy-shade-removal-' +
                                      im_folder +'.jpg'), cmap='gray')
        output_image(normalize_histogram(Ri),
                     os.path.join(im_dir, 'Red-entropy-albedo-'+
                                      im_folder +'.jpg'), cmap='gray')
        output_image(RGB, os.path.join(im_dir,'Red-rgb-gamma-'+im_folder+'.jpg'))
        output_glacier_map_background(Z, R, os.path.join(im_dir,
                                                         'Red-background.jpg'),
                                      az=az, zn=zn, compress=95)

    # construct connection files
    shadow_image_to_list(M_acwe_cl, geoTransform, im_dir, Zn=zn, Az=az,
                         bbox=bbox_ij)
    # create casting plot
    output_cast_lines_from_conn_txt(im_dir, geoTransform, R, step=1)