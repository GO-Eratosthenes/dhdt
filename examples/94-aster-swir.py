import os

import matplotlib.pyplot as plt
import numpy as np

from dhdt.generic.handler_aster import \
    get_as_image_locations, list_platform_metadata_tr
from dhdt.generic.mapping_tools import pix2map, ref_rotate
from dhdt.generic.orbit_tools import calculate_correct_mapping, \
    remap_observation_angles

from dhdt.input.read_aster import list_central_wavelength_as, \
    read_stack_as, get_flight_path_as, read_view_angles_as, \
    read_lattice_as
from dhdt.preprocessing.multispec_transforms import bispectral_minimum
from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.coupling_tools import match_pair
from dhdt.testing.mapping_tools import create_artificial_geoTransform

dat_dir = '/Users/Alten005/jitter/Scherler' # os.getcwd()
dat_fol = ['AST_L1A_00311102004045819_20230228095904_28538',
           'AST_L1A_00311292005045840_20230228083802_15377'] # khumbu
t_size = 2**5
roi = 30. # resolution of interest
bid_min, bid_max = 7, 9
as_level = 1


# we are only interested in the high resolution imagery, for now
for idx, foi in enumerate(dat_fol): # loop through to see if folders of interest exist
    as_df = list_central_wavelength_as()
    as_dir = os.path.join(dat_dir, foi+os.path.sep)

    as_dict = list_platform_metadata_tr()
    sat_time, sat_xyz, sat_uvw = get_flight_path_as(as_dir)

    as_df = as_df[np.logical_and(as_df['bandid']>=bid_min,
                                 as_df['bandid']<=bid_max)]

    # get metadata
    as_df = get_as_image_locations(as_dir, as_df)
    Zn_grd, Az_grd = read_view_angles_as(as_dir, boi_df=as_df)
    Lat_grd, Lon_grd, I_grd, J_grd = read_lattice_as(as_dir, boi_df=as_df)

    Im, crs, geoTransform,_ = read_stack_as(as_dir, as_df, level=as_level)

    # make a nice displacement map
    geoDummy = create_artificial_geoTransform(Im, spac=roi)
    I_samp, J_samp = get_coordinates_of_template_centers(Im, t_size//4)
    X_samp, Y_samp = pix2map(geoDummy, I_samp, J_samp)
    Msk = np.ones_like(Im, dtype=bool)

    # fig.7 in Iwasaki
    X_disp,Y_disp,sc78 = match_pair(Im[...,0], Im[...,1], Msk[...,0], Msk[...,1],
                                    geoDummy, geoDummy,
                                    X_samp, Y_samp,
                                    temp_radius=t_size, search_radius=t_size,
                                    correlator='phas_corr', metric='peak_ratio')

    dX78, dY78 = X_disp - X_samp, Y_disp - Y_samp
    dX78_hat, dY78_hat = np.nanmedian(dX78.ravel()), np.nanmedian(dY78.ravel())

    X_disp,Y_disp,sc89 = match_pair(Im[...,1], Im[...,2], Msk[...,1], Msk[...,2],
                                    geoDummy, geoDummy,
                                    X_samp, Y_samp,
                                    temp_radius=t_size, search_radius=t_size,
                                    correlator='phas_corr', metric='peak_ratio')
    dX89, dY89 = X_disp - X_samp, Y_disp - Y_samp
    dX89_hat, dY89_hat = np.nanmedian(dX89.ravel()), np.nanmedian(dY89.ravel())

    dAcc = (dX78 - dX78_hat) - (dX89 - dX89_hat)
    dAll = (dY78 - dY78_hat) - (dY89 - dY89_hat)

    sc_th = 4.
    IN = np.logical_and(sc78>=sc_th, sc89>=sc_th)

    dAcc[~IN] = np.nan
    dAll[~IN] = np.nan

    v_ext = 5

    dAcc_vert, dAcc_horz = np.nanmean(dAcc, axis=0), np.nanmean(dAcc, axis=1)
    dAll_vert, dAll_horz = np.nanmean(dAll, axis=0), np.nanmean(dAll, axis=1)

    h, w = plt.figaspect(1)
    fig = plt.figure(figsize=(h, w))
    grid = fig.add_gridspec(nrows=2, ncols=2,
                            hspace=0, wspace=0, width_ratios=[2, 1],
                            height_ratios=[1, 2])
    ax = fig.add_subplot(grid[1, 0])
    ax.set_aspect('equal')
    ay = fig.add_subplot(grid[0, 0], sharex=ax)
    az = fig.add_subplot(grid[1, 1], sharey=ax)
    plt.setp(ay.get_xticklabels(), visible=False)
    plt.setp(az.get_yticklabels(), visible=False)

    ax.imshow(dAcc, vmin=-v_ext, vmax=+v_ext,
               cmap=plt.cm.RdBu)
    az.plot(dAcc_horz, range(len(dAcc_horz)))
    ay.plot(range(len(dAcc_vert)), dAcc_vert)
    az.set_xlim(-4,4)
    ay.set_ylim(-2, 2)
    # Add this for square image
    plt.show()


    h, w = plt.figaspect(1)
    fig = plt.figure(figsize=(h, w))
    grid = fig.add_gridspec(nrows=2, ncols=2,
                            hspace=0, wspace=0, width_ratios=[2, 1],
                            height_ratios=[1, 2])
    ax = fig.add_subplot(grid[1, 0])
    ax.set_aspect('equal')
    ay = fig.add_subplot(grid[0, 0], sharex=ax)
    az = fig.add_subplot(grid[1, 1], sharey=ax)
    plt.setp(ay.get_xticklabels(), visible=False)
    plt.setp(az.get_yticklabels(), visible=False)

    ax.imshow(dAll, vmin=-v_ext, vmax=+v_ext,
               cmap=plt.cm.RdBu)
    az.plot(dAll_horz, range(len(dAll_horz)))
    ay.plot(range(len(dAll_vert)), dAll_vert)
    az.set_xlim(-4,4)
    ay.set_ylim(-2, 2)
    plt.show()

    print('.')
dX_hat, dY_hat = np.nanmedian(dX.ravel()), np.nanmedian(dY.ravel())

