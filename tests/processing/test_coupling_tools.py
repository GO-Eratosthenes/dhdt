import os
import tempfile

import numpy as np

from dhdt.generic.handler_im import bilinear_interpolation
from dhdt.generic.mapping_io import read_geo_info, read_geo_image
from dhdt.generic.mapping_tools import map2pix, pix2map
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_stack, clean_locations_with_no_id, \
    get_casted_elevation_difference, get_hypsometric_elevation_change
from dhdt.processing.coupling_tools import couple_pair, match_pair
from dhdt.processing.network_tools import get_network_indices
from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.matching_tools_organization import \
    list_spatial_correlators, list_peak_estimators, list_phase_estimators, \
    list_frequency_correlators

# artificial creation functions
from dhdt.testing.matching_tools import \
    create_sample_image_pair, _test_subpixel_localization
from dhdt.testing.terrain_tools import \
    create_artificial_terrain, create_artifical_sun_angles, \
    create_shadow_caster_casted
from dhdt.testing.mapping_tools import \
    create_local_crs, create_artificial_geoTransform

# testing functions
def test_match_pair(im_size=2**7,temp_size=2**4, tolerance=10):
    # create artificial data
    I = np.ones((im_size, im_size))
    geoTransform = create_artificial_geoTransform(I, spac=1.)
    I_samp, J_samp = get_coordinates_of_template_centers(I, temp_size//2)
    X_samp, Y_samp = pix2map(geoTransform, I_samp, J_samp)
    del I
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=im_size,
             max_range=3, integer=False, ndim=bands)

        # test frequency correlators
        _test_frequency_correlators(im1, im2, geoTransform, X_samp, Y_samp,
                                  temp_size, di, dj, tolerance=tolerance)
        #todo: test phase plane estimators

        if bands==1:
            # test spatial correlators
            _test_spatial_correlators(im1, im2, geoTransform, X_samp, Y_samp,
                                      temp_size, di, dj, tolerance=tolerance)
            #todo: test peak estimators

    return

def _test_spatial_correlators(im1,im2,geoTransform,X_samp,Y_samp,temp_size,
                              di, dj, tolerance=0.5):
    corr_list = list_spatial_correlators(unique=True)
    msk1, msk2 = np.invert(np.isnan(im1)), np.invert(np.isnan(im2))
    for correlator in corr_list:
        print('testing : ' + correlator)
        X_disp,Y_disp,_ = match_pair(im1, im2, msk1, msk2,
                                     geoTransform, geoTransform, X_samp, Y_samp,
                                     temp_radius=temp_size,
                                     search_radius=2*temp_size,
                                     correlator=correlator)
        dX, dY = X_samp-X_disp, Y_samp-Y_disp
        dX_hat, dY_hat = np.nanmedian(dX.ravel()), np.nanmedian(dY.ravel())
        di_hat,dj_hat = map2pix(geoTransform, dX_hat, dY_hat)
        _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
        print('passed')
    return

def _test_frequency_correlators(im1,im2,geoTransform,X_samp,Y_samp,temp_size,
                                di, dj, tolerance=0.5):
    corr_list = list_frequency_correlators()
    msk1, msk2 = np.invert(np.isnan(im1)), np.invert(np.isnan(im2))
    for correlator in corr_list:
        print('testing : ' + correlator)
        X_disp,Y_disp,_ = match_pair(im1, im2, msk1, msk2,
                                     geoTransform, geoTransform, X_samp, Y_samp,
                                     temp_radius=temp_size,
                                     search_radius=temp_size,
                                     correlator=correlator)
        dX, dY = X_disp-X_samp, Y_disp-Y_samp
        dX_hat, dY_hat = np.nanmedian(dX.ravel()), np.nanmedian(dY.ravel())
        di_hat,dj_hat = map2pix(geoTransform, dX_hat, dY_hat)
        _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
        print('passed')
    return

def test_photohypsometric_coupling(N=10, Z_shape=(400,600), tolerance=20):
    Z, geoTransform = create_artificial_terrain(Z_shape[0], Z_shape[1])
    az,zn = create_artifical_sun_angles(N)
    crs = create_local_crs()

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        for i in range(N):
            f_name = "conn-"+str(i).zfill(3)+".txt"
            create_shadow_caster_casted(Z, geoTransform, az[i], zn[i],
                dump_path, out_name=f_name, incl_image=False, crs=crs)

        conn_list = tuple("conn-"+str(i).zfill(3)+".txt" for i in range(N))
        dh = read_conn_files_to_stack(None, conn_file=conn_list,
                                      folder_path=dump_path)

    dh = clean_locations_with_no_id(dh)
    dxyt = get_casted_elevation_difference(dh)
    dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransform)

    assert np.isclose(0, np.quantile(dhdt['dZ_12'], 0.5), atol=tolerance)

def test_photohypsometric_refinement_by_same(Z_shape=(400,600), tolerance=1.0,
                                             weight=False):
    Z, geoTransform = create_artificial_terrain(Z_shape[0], Z_shape[1])
    az, zn = create_artifical_sun_angles(1)

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        create_shadow_caster_casted(Z, geoTransform,
                                    az, zn, dump_path, out_name="conn.txt",
                                    incl_image=True, incl_wght=weight)
        # read and refine through image matching
        if weight:
             post_1,post_2_new,caster,dh,score,caster_new,_,_,_ = \
                 couple_pair(os.path.join(dump_path, "conn.tif"),
                             os.path.join(dump_path, "conn.tif"),
                             wght_1=os.path.join(dump_path, "connwgt.tif"),
                             wght_2=os.path.join(dump_path, "connwgt.tif"),
                             temp_radius=5, search_radius=9, rect=None,
                             match='wght_corr')
        else:
            post_1,post_2_new,caster,dh,score,caster_new,_,_,_ = \
                couple_pair(os.path.join(dump_path, "conn.tif"), \
                            os.path.join(dump_path, "conn.tif"),
                            rect=None)
    pix_dispersion = np.nanmedian(np.hypot(post_1[:,0]-post_2_new[:,0],
                                           post_1[:,1]-post_2_new[:,1]))
    assert np.isclose(0, pix_dispersion, atol=tolerance)

def _test_photohypsometric_refinement(N, Z_shape, tolerance=0.1):
    Z, geoTransform = create_artificial_terrain(Z_shape[0], Z_shape[1])
    az,zn = create_artifical_sun_angles(N, az_min=179.9, az_max=180.1,
                                        zn_min=60., zn_max=65.)

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        for i in range(N):
            f_name = "conn-"+str(i).zfill(3)+".txt"
            create_shadow_caster_casted(Z, geoTransform,
                                        az[i], zn[i], dump_path, out_name=f_name, incl_image=True,
                                        incl_wght=True)
        # read and refine through image matching
        print('+')
        match_idxs = get_network_indices(N)
        for idx in range(match_idxs.shape[1]):
            fname_1 = "conn-"+str(match_idxs[0][idx]).zfill(3)+".tif"
            fname_2 = "conn-"+str(match_idxs[1][idx]).zfill(3)+".tif"
            file_1 = os.path.join(dump_path,fname_1)
            file_2 = os.path.join(dump_path,fname_2)
            w_1 = file_1.split('.')[0]+"wgt.tif"
            w_2 = file_2.split('.')[0]+"wgt.tif"

            post_1,post_2_new,caster,dh,score,caster_new,_,_,_ = \
                couple_pair(file_1, file_2, wght_1=w_1, wght_2=w_2,
                            temp_radius=5, search_radius=9, rect="metadata",
                            match='wght_corr')
            geoTransform = read_geo_info(file_1)[1]
            i_1,j_1 = map2pix(geoTransform, post_1[:,0], post_1[:,1])
            i_2,j_2 = map2pix(geoTransform, post_2_new[:,0], post_2_new[:,1])
            h_1 = bilinear_interpolation(Z, i_1, j_1)
            h_2 = bilinear_interpolation(Z, i_2, j_2)
            dh_12 = h_2 - h_1

            I_1, I_2 = read_geo_image(file_1)[0], read_geo_image(file_2)[0]

            cnt = 1320
            w,h = 11,11
            i_idx,j_idx = np.floor(i_1[cnt]).astype(int), \
                          np.floor(j_1[cnt]).astype(int)
            Isub_1 = I_1[i_idx-w:i_idx+w,j_idx-w:j_idx+w]
            i_idx, j_idx = np.floor(i_2[cnt]).astype(int), \
                           np.floor(j_2[cnt]).astype(int)
            Isub_2 = I_2[i_idx-w:i_idx+w, j_idx-w:j_idx+w]

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(Isub_1)
            ax1.scatter(np.mod(j_1[cnt],1)+w, np.mod(i_1[cnt],1)+h, marker='+')
            ax2.imshow(Isub_2)
            ax2.scatter(np.mod(j_2[cnt],1)+w, np.mod(i_2[cnt],1)+h, marker='+')

            print('.')
    return