import os
import glob
import math
import warnings
warnings.filterwarnings('once')
from tqdm import tqdm

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import stack_arrays, merge_arrays

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from skimage import measure

from ..generic.mapping_tools import pix2map, cast_orientation, ref_trans, \
    aff_trans_template_coord, rot_trans_template_coord
from ..generic.mapping_io import read_geo_image, read_geo_info
from ..generic.handler_im import bilinear_interpolation, select_boi_from_stack

from ..input.read_sentinel2 import \
    read_sun_angles_s2, get_local_bbox_in_s2_tile

from .matching_tools import \
    pad_images_and_filter_coord_list, pad_radius, get_integer_peak_location, \
    get_data_and_mask
from .matching_tools_organization import \
    list_spatial_correlators, list_peak_estimators, list_phase_estimators, \
    match_translation_of_two_subsets, estimate_subpixel, estimate_precision, \
    estimate_translation_of_two_subsets, list_frequency_correlators, \
    estimate_match_metric
from .matching_tools_differential import \
    affine_optical_flow, simple_optical_flow
from .matching_tools_binairy_boundaries import \
    beam_angle_statistics, cast_angle_neighbours, neighbouring_cast_distances


# simple image matching
def match_pair(I1, I2, L1, L2, geoTransform1, geoTransform2, X_grd, Y_grd,
               temp_radius=7, search_radius=22,
               correlator='robu_corr', subpix='moment',
               processing='simple', boi=np.array([]), precision_estimate=False,
               metric='peak_abs', **kwargs):
    """ simple high level image matching routine

    Parameters
    ----------
    I1 : {numpy.array, numpy.ma}, size=(m,n,b), dtype=float, ndim={2,3}
        first image array, can be complex.
    I2 : {numpy.array, numpy.ma}, size=(m,n,b), dtype=float, ndim={2,3}
        second image array, can be complex.
    L1 : numpy.array, size=(m,n), dtype=bool
        labelled image of the first image (I1), hence False's are neglected.
    L2 : numpy.array, size=(m,n), dtype=bool
        labelled image of the second image (I2), hence False's are neglected.
    geoTransform1 : tuple
        affine transformation coefficients of array M1
    geoTransform2 : tuple
        affine transformation coefficients of array M2
    X_grd : numpy.array, size=(k,l), type=float
        horizontal map coordinates of matching centers of array M1
    Y_grd : numpy.array, size=(k,l), type=float
        vertical map coordinates of matching centers of array M2
    temp_radius: integer
        amount of pixels from the center
    search_radius: integer
        amount of pixels from the center
    correlator : {'norm_corr' (default), 'aff_of', 'cumu_corr',\
                  'sq_diff', 'sad_diff',\
                  'cosi_corr', 'phas_only', 'symm_phas', 'ampl_comp',\
                  'orie_corr', 'mask_corr', 'bina_phas', 'wind_corr',\
                  'gaus_phas', 'upsp_corr', 'cros_corr', 'robu_corr',\
                  'lucas_kan', 'lucas_aff'}
        Specifies which matching method to apply, see
        "list_spatial_correlators", "list_frequency_correlators" or
        "list_differential_correlators" for more information
    subpix : {'tpss' (default), 'svd', 'radon', 'hough', 'ransac', 'wpca',\
              'pca', 'lsq', 'diff', 'gauss_1', 'parab_1', 'moment', 'mass',\
              'centroid', 'blais', 'ren', 'birch', 'eqang', 'trian', 'esinc',\
              'gauss_2', 'parab_2', 'optical_flow'}
        Method used to estimate the sub-pixel location, see
        "list_peak_estimators", list_phase_estimators" for more information
    metric : {'peak_abs' (default), 'peak_ratio', 'peak_rms', 'peak_ener',
              'peak_nois', 'peak_conf', 'peak_entr'}
        Metric to be used to describe the matching score.
    precision_estimate : boolean, default=False
        estimate the precision of the match, by modelling the correlation
        function through a Gaussian, see also [1] for more details.
    processing : {'simple' (default), 'refine', 'stacking'}
        Specifies which procssing strategy to apply to the imagery:

          * 'simple' : direct single pair processing
          * 'refine' : move template to first estimate, and refine matching
          * 'stacking' : matching several bands and combine the result, see
                         also [2] for more details
    boi : np.array
        list of bands of interest

    Returns
    -------
    X2_grd, Y2_grd : numpy.array, size=(k,l), type=float
        horizontal and vertical map coordinates of the refined matching centers
        of array I2
    match_metric : numpy.array, size=(k,l), type=float
        matching score of individual matches

    See Also
    --------
    .matching_tools_organization.list_differential_correlators,
    .matching_tools_organization.list_spatial_correlators,
    .matching_tools_organization.list_frequency_correlators,
    .matching_tools_organization.list_peak_estimators,
    .matching_tools_organization.list_phase_estimators

    References
    ----------
    .. [1] Altena et al. "Correlation dispersion as a measure to better estimate
       uncertainty of remotely sensed glacier displacements", The Crysophere,
       vol.16(6) pp.2285–2300, 2022.
    .. [2] Altena & Leinss, "Improved surface displacement estimation through
       stacking cross-correlation spectra from multi-channel imagery", Science
       of remote sensing, vol.6 pp.100070, 2022.
    """
    # combating import loops
    from .matching_tools_organization import \
        list_differential_correlators, list_peak_estimators, \
        list_frequency_correlators

    # init
    assert type(I1) in (np.ma.core.MaskedArray, np.ndarray), \
         ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.ndarray), \
         ("please provide an array")
    if (type(I1) not in (np.ma.core.MaskedArray,)) and (L1 is None):
        L1 = np.ones_like(I1)
    elif L1 is None: L1 = np.array([])
    if (type(I2) not in (np.ma.core.MaskedArray,)) and (L2 is None):
        L2 = np.ones_like(I2)
    elif L2 is None: L2=np.array([])
    assert isinstance(L1, np.ndarray), "please provide an array"
    assert isinstance(L2, np.ndarray), "please provide an array"
    assert isinstance(geoTransform1, tuple), 'geoTransform should be a tuple'
    assert isinstance(geoTransform2, tuple), 'geoTransform should be a tuple'
    assert X_grd.shape == Y_grd.shape  # should be of the same size
    assert temp_radius <= search_radius, 'given search radius is too small'

    b=1 if kwargs.get('num_estimates') == None else kwargs.get('num_estimates')

    I1, L1 = get_data_and_mask(I1, L1)
    I2, L2 = get_data_and_mask(I2, L2)

    if X_grd.ndim==1: # sometimes a collumn is given instead of a grid
        X_grd,Y_grd = X_grd[:,np.newaxis], Y_grd[:,np.newaxis]

    correlator = correlator.lower()
    if (subpix is not None): subpix=subpix.lower()

    # create lists with the different match approaches
    frequency_based = list_frequency_correlators()
    differential_based = list_differential_correlators()
    peak_based = list_peak_estimators()

    # prepare
    I1,I2 = select_boi_from_stack(I1,boi), select_boi_from_stack(I2,boi)
    I1,I2,i1,j1,i2,j2,IN = pad_images_and_filter_coord_list(
        I1, I2, geoTransform1, geoTransform2, X_grd, Y_grd,
        temp_radius, search_radius, same=True)
    geoTransformPad2 = ref_trans(geoTransform2, -search_radius, -search_radius)

    L1,L2 = pad_radius(L1,temp_radius, cval=False), \
            pad_radius(L2, search_radius, cval=False)

    (m,n) = X_grd.shape
    X2_grd,Y2_grd = np.zeros((m,n,b)), np.zeros((m,n,b))
    if precision_estimate: # also include error estimate
        match_metric = np.zeros((m,n,4))
    else:
        match_metric = np.zeros((m,n))
    grd_new = np.where(IN)

    # processing
    for counter in tqdm(range(len(i1))): # loop through all posts
        # create templates
        if correlator in frequency_based:
            I1_sub = create_template_off_center(I1,i1[counter],j1[counter],
                                                2*temp_radius)
            I2_sub = create_template_off_center(I2,i2[counter],j2[counter],
                                                2*search_radius)
            L1_sub = create_template_off_center(L1,i1[counter],j1[counter],
                                                2*temp_radius)
            L2_sub = create_template_off_center(L2,i2[counter],j2[counter],
                                                2*search_radius)
        else: # differential or spatial based
            I1_sub = create_template_at_center(I1,i1[counter],j1[counter],
                                                temp_radius)
            I2_sub = create_template_at_center(I2,i2[counter],j2[counter],
                                                search_radius)
            L1_sub = create_template_at_center(L1,i1[counter],j1[counter],
                                                temp_radius)
            L2_sub = create_template_at_center(L2,i2[counter],j2[counter],
                                                search_radius)

        if np.all(L1_sub==0) or np.all(L2_sub==0):
            di, dj, score = np.nan*np.zeros(b), np.nan*np.zeros(b), 0
            continue

        if correlator in differential_based:
            di,dj,rms = estimate_translation_of_two_subsets(I1_sub, I2_sub,
                 L1_sub, L2_sub, correlator, **kwargs)
            score = np.sqrt(np.nansum(rms**2)) # euclidean distance of rms
        else:
            QC = match_translation_of_two_subsets(I1_sub,I2_sub,
                 correlator, subpix, L1_sub,L2_sub)
            if (subpix in peak_based) or (subpix is None):
                di,dj = get_integer_peak_location(QC, metric=None)[0:2]
            else: # integer estimation in phase space is not straight forward
                di,dj = np.zeros(b), np.zeros(b)
            m0 = np.array([di, dj])

#            if np.logical_and(np.all(I1_sub!=0), np.all(I2_sub!=0)):
#                print('all data in arrays')

            if processing in ['refine']: # reposition second template
                if correlator in frequency_based:
                    I2_new = create_template_off_center(I2,
                        i2[counter]-di, j2[counter]-dj, 2*temp_radius)
                    L2_new = create_template_off_center(L2,
                        i2[counter]-di, j2[counter]-dj, 2*temp_radius)
                else:
                    I2_new = create_template_at_center(I2,
                        i2[counter]-di, j2[counter]-dj, temp_radius)
                    L2_new = create_template_at_center(L2,
                        i2[counter]-di, j2[counter]-dj, temp_radius)

                QC = match_translation_of_two_subsets(I1_sub, I2_new,
                        correlator,subpix, L1_sub,L2_new)
            if subpix is None:
                ddi,ddj = 0, 0
            else:
                ddi,ddj = estimate_subpixel(QC, subpix, m0=m0)

            di += ddi
            dj += ddj

            # get the similarity metrics
            if precision_estimate:
                si,sj,rho = estimate_precision(QC, di,dj,
                                               method='gaussian')
            score = estimate_match_metric(QC, di=di, dj=dj, metric=metric)
#            if score>.7:
#                print('.')

        # transform from local image to metric map system
        x2_new,y2_new = pix2map(geoTransformPad2,
                                i2[counter] + di, j2[counter] + dj)

        # write results in arrays
        idx_grd = np.unravel_index(grd_new[0][counter], (m,n), 'C')
        if b>1:
            for bands in range(b):
                X2_grd[idx_grd[0],idx_grd[1],bands] = x2_new[bands]
                Y2_grd[idx_grd[0],idx_grd[1],bands] = y2_new[bands]
        else:
            X2_grd[idx_grd[0],idx_grd[1]] = x2_new
            Y2_grd[idx_grd[0],idx_grd[1]] = y2_new
        if precision_estimate:
            match_metric[idx_grd[0],idx_grd[1],0] = score
            match_metric[idx_grd[0],idx_grd[1],1] = si
            match_metric[idx_grd[0],idx_grd[1],2] = sj
            match_metric[idx_grd[0],idx_grd[1],3] = rho
        else:
            match_metric[idx_grd[0],idx_grd[1]] = score

    if X2_grd.shape[2]==1:
        X2_grd,Y2_grd = np.squeeze(X2_grd), np.squeeze(Y2_grd)
    return X2_grd, Y2_grd, match_metric

def couple_pair(file_1, file_2, bbox=None, rgi_id=None,
                wght_1=None, wght_2=None,
                rect='metadata', prepro='bandpass',
                match='norm_corr', boi=np.array([]),
                temp_radius=3, search_radius=7,
                processing=None, subpix='moment', metric='peak_entr',
                shw_fname="shadow.tif", label_fname="labelPolygons.tif"):
    """ refine and couple two shadow images together via matching

    Parameters
    ----------
    file_1, file_2 : string
        path to first and second image(ry)
    bbox : {list, numpy.array}, size=(4,1)
        bounding box of region of interest
    rgi_id : string
        code of the glacier of interest following the Randolph Glacier Inventory
    rect : {"binary", "metadata", None}
        rectify the imagery so shear and elongation is reduced, done by
            * binary - asdsa
            * metadata - estimate scale and shear from geometry and metadata
            else: do not include affine corrections
    prepro : string
        #todo
    match: {'norm_corr' (default), 'aff_of', 'cumu_corr', 'sq_diff',
    'sad_diff', 'cosi_corr', 'phas_only', 'symm_phas', 'ampl_comp', 'orie_corr',
    'mask_corr', 'bina_phas', 'wind_corr', 'gaus_phas', 'upsp_corr',
    'cros_corr', 'robu_corr', 'lucas_kan', 'lucas_aff'}
        Specifies which matching method to apply, see
            "list_spatial_correlators", "list_frequency_correlators" or
            "list_differential_correlators" for more information
    boi : numpy.array
        list of bands of interest
    temp_radius: integer
        amount of pixels from the center
    search_radius: integer
        amount of pixels from the center
    processing : {"stacking", "shadow", None}
        matching can be done on a single image pair, or by matching several
        bands and combine the result (stacking)
            * stacking - using multiple bands in the matching, see [1]
            * shadow - using the shadow transfrom in the given folder
            else - using the image files provided in 'file1' & 'file2'
    subpix : {'tpss' (default), 'svd', 'radon', 'hough', 'ransac', 'wpca',\
            'pca', 'lsq', 'diff', 'gauss_1', 'parab_1', 'moment', 'mass',\
            'centroid', 'blais', 'ren', 'birch', 'eqang', 'trian', 'esinc',\
            'gauss_2', 'parab_2', 'optical_flow'}
        Method used to estimate the sub-pixel location, see
        "list_peak_estimators", list_phase_estimators" for more information
    shw_fname : str, default='shadow.tif'
        image name of the shadow transform/enhancement
    label_fname : str, default="labelPolygons.tif"
        filename of the numbered image with all the shadow polygons

    Returns
    -------
    post_1 : numpy.array, size=(k,2)
        locations of shadow ridge at instance 1
    post_2_corr : numpy.array, size=(k,2)
        locations of shadow ridge at instance 2
    casters : numpy.array, size=(k,2)
        common point where the shadow originates
    dh : numpy.array, size=(k,1)
        elevation differences between "post_1" and "post_2_corr"

    References
    ----------
    .. [1] Altena & Leinss, "Improved surface displacement estimation through
       stacking cross-correlation spectra from multi-channel imagery", Science
       of remote sensing, vol.6 pp.100070, 2022.
    """
    # admin
    assert os.path.exists(file_1), ('file does not seem to exist')
    assert os.path.exists(file_2), ('file does not seem to exist')
    dir_im1, name_im1 = os.path.split(file_1)
    dir_im2, name_im2 = os.path.split(file_2)

    crs,geoTransform1,_,rows1,cols1,_ = read_geo_info(file_1)
    _,geoTransform2,_,rows2,cols2,_ = read_geo_info(file_2)

    if bbox is None:  bbox = get_local_bbox_in_s2_tile(file_1, dir_im1)

    # text file listing coordinates of shadow casting and casted points
    if os.path.exists(file_1[:-4]+'.tif'): # testing dataset is created
        conn_fname1 = name_im1[:-4]+'.txt'
        conn_fname2 = name_im2[:-4]+'.txt'
    elif rgi_id is not None:
        conn_fname1 = f"conn-rgi{rgi_id[0]:08d}.txt"
        conn_fname2 = conn_fname1
    else:
        conn_fname1 = "conn.txt"
        conn_fname2 = conn_fname1

    conn_1 = np.loadtxt(fname=os.path.join(dir_im1, conn_fname1))
    conn_2 = np.loadtxt(fname=os.path.join(dir_im2, conn_fname2))

    idxConn = pair_images(conn_1, conn_2) # connected list
    caster = conn_1[idxConn[:,1],0:2]
    # cast location in xy-coordinates
    post_1,post_2 = conn_1[idxConn[:,1],2:4].copy(),\
                     conn_2[idxConn[:,0],2:4].copy()

    if processing in ['shadow']:
        I1 = read_geo_image(os.path.join(dir_im1, shw_fname), boi=boi)[0]
        I2 = read_geo_image(os.path.join(dir_im2, shw_fname), boi=boi)[0]
    elif processing in ['stacking']: # multi-spectral stacking
        #todo
        I1 = get_stack_out_of_dir(dir_im1,boi,bbox)[0]
        I2 = get_stack_out_of_dir(dir_im2,boi,bbox)[0]
    else: # load the files given
        I1 = read_geo_image(file_1, boi=boi)[0]
        I2 = read_geo_image(file_2, boi=boi)[0]

    # handle masking
    if wght_1 is not None:
        if isinstance(wght_1, str):
            L1 = read_geo_image(wght_1)[0].data
        elif type(wght_1) in (np.ndarray,):
            L1 = wght_1
        I1 = get_data_and_mask(I1)[0]
    else:
        I1, L1 = get_data_and_mask(I1)
    if wght_2 is not None:
        if isinstance(wght_2, str):
            L2 = read_geo_image(wght_2)[0].data
        elif type(wght_2) in (np.ndarray,):
            L2 = wght_2
        I2 = get_data_and_mask(I2)[0]
    else:
       I2, L2 = get_data_and_mask(I2)

    # handle affine compensation
    if rect in ['metadata']:
        az_1, az_2 = conn_1[idxConn[:, 1], 4], conn_2[idxConn[:, 0], 4]
        scale_12, simple_sh = get_shadow_rectification(caster,
           post_1, post_2, az_1, az_2)
    else: # start with rigid displacement
        scale_12 = np.ones((np.shape(post_1)[0]))
        simple_sh = np.zeros((np.shape(post_1)[0]))

    post_2_new, score = match_shadow_casts(I1, I2, L1, L2,
       geoTransform1, geoTransform2, post_1, post_2, scale_12, simple_sh,
       temp_radius=temp_radius, search_radius=search_radius,
       prepro=prepro, correlator=match, subpix=subpix, metric='peak_abs')

    # cast location in xy-coordinates for t_1
    post_1,post_2 = conn_1[idxConn[:,1],2:4], conn_2[idxConn[:,0],2:4]
    # extract elevation change
    sun_1,sun_2 = conn_1[idxConn[:,1],4:6], conn_2[idxConn[:,0],4:6]
    # calculate new intersection

    caster_new = get_caster(sun_1[:,0], sun_2[:,0], post_1, post_2_new)

    dh_old = get_elevation_difference(sun_1, sun_2, post_1, post_2, caster)
    dh_new = get_elevation_difference(sun_1, sun_2, post_1, post_2_new, caster)
    return post_1, post_2_new, caster, dh_new, score, caster_new, dh_old, \
           post_2, crs

def create_template_at_center(I, i,j, radius, filling='random'):
    """ get sub template of data array, at a certain location

    Parameters
    ----------
    I : numpy.array, size=(m,n) or (m,n,b)
        data array
    i,j : integer
        horizontal, vertical coordinate of the template center
    radius : integer
        radius of the template
    filling : {'random', 'NaN', 'zero'}, default='random'
        sometimes the location is outside the domain or to close to the border.
        then the template is filled up with either:
           * 'random' : random numbers, ranging from 0...1, 0...2**bit
                        dependend on the data type of the array
           * 'nan' : not a number values
           * 'zero' : all zeros

    Returns
    -------
    I_sub : numpy.array, size={(m,n),(m,n,b)}
        template of data array

    See Also
    --------
    create_template_off_center : similar function, but creates even template

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    i,j = np.round(i).astype(int), np.round(j).astype(int)
    if not type(radius) is tuple: radius = (radius, radius)

    if I.ndim==2:
        sub_shape = (2*radius[0]+1, 2*radius[1]+1)
    elif I.ndim==3:
        sub_shape = (2*radius[0]+1, 2*radius[1]+1, I.shape[2])

    # create sub template
    if filling in ('random', ):
        quant = 0
        if I.dtype.type == np.uint8:
            quant = 8
        elif I.dtype.type == np.uint16:
            quant = 16

        if quant==0:
            I_sub = np.random.random_sample(sub_shape)
        else:
            I_sub = np.random.randint(2**quant, size=sub_shape)
    elif filling in ('nan', ):
        I_sub = np.nan * np.zeros(sub_shape)
    else:
        I_sub = np.zeros(sub_shape)

    if I.dtype.type == np.uint8:
        I_sub = I_sub.astype(np.uint8)
    elif I.dtype.type == np.uint16:
        I_sub = I_sub.astype(np.uint16)

    m,n = I.shape[0], I.shape[1]
    min_0, max_0 = np.maximum(0, i-radius[0]), np.minimum(m, i+radius[0]+1)
    min_1, max_1 = np.maximum(0, j-radius[1]), np.minimum(n, j+radius[1]+1)

    if np.logical_or(min_0>=max_0, min_1>=max_1):
        return I_sub
    sub_min_0 = np.abs(np.minimum(i-radius[0], 0))
    sub_max_0 = (2*radius[0]+1) + m-np.maximum(i+radius[0]+1, m)
    sub_min_1 = np.abs(np.minimum(j-radius[0], 0))
    sub_max_1 = (2*radius[1]+1) + n-np.maximum(j+radius[1]+1, n)
    if I.ndim==3:
        I_sub[sub_min_0:sub_max_0,sub_min_1:sub_max_1,:] = \
            I[min_0:max_0, min_1:max_1, :]
    else:
        I_sub[sub_min_0:sub_max_0,sub_min_1:sub_max_1] = \
            I[min_0:max_0, min_1:max_1]
    return I_sub

def create_template_off_center(I, i,j, width, filling='random'):
    """ get sub template of data array, at a certain location

    Parameters
    ----------
    I : numpy.array, size={(m,n), (m,n,b)}
        data array
    i : integer
        vertical coordinate of the template center
    j : integer
        horizontal coordinate of the template center
    width : {integer, tuple}
        dimension of the template
    filling : {'random', 'NaN', 'zero'}
        sometimes the location is outside the domain or to close to the border.
        then the template is filled up with either:
           * 'random' : random numbers, ranging from 0...1, 0...2**bit
                        dependend on the data type of the array
           * 'nan' : not a number values
           * 'zero' : all zeros

    Returns
    -------
    I_sub : numpy.array, size={(m,n),(m,n,b)}
        template of data array

    See Also
    --------
    create_template_at_center : similar function, but creates odd template

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |

    """
    # make sure the input are integer loctions
    i,j = np.round(i).astype(int), np.round(j).astype(int)

    if not type(width) is tuple: width = (width, width)
    radius = (width[0] //2, width[1] //2)
    if I.ndim==3:
        I_sub = I[i-radius[0]:i+radius[1], j-radius[0]:j+radius[1], :]
    else:
        I_sub = I[i-radius[0]:i+radius[1], j-radius[0]:j+radius[1]]
    return I_sub

def make_time_pairing(acq_time,
                      t_min=np.timedelta64(300, 'ms').astype('timedelta64[ns]'),
                      t_max=np.timedelta64(3, 's').astype('timedelta64[ns]')):
    """ construct pairing based upon time difference

    Parameters
    ----------
    acq_time : numpy.array, size=(_,1), type=np.timedelta64[ns]
        acquisition times of different recordings
    t_min : integer, type=np.timedelta64[ns]
        minimal time difference to pair
    t_max : integer, type=np.timedelta64[ns]
        maximum time difference to pair

    Returns
    -------
    idx_1 : TYPE
        index of first couple, based on the "acq_time" as reference
    idx_2 : TYPE
        index of second couple, based on the "acq_time" as reference
    couple_dt : numpy.array, size=(_,1), type=np.timedelta64[ns]
        time difference of the pair
    """
    t_1,t_2 = np.meshgrid(acq_time, acq_time,
                          sparse=False, indexing='ij')
    dt = t_1 - t_2 # time difference matrix

    # only select in one direction
    idx_1,idx_2 = np.where((dt > t_min) & (dt < t_max))

    couple_dt = dt[idx_1,idx_2]
    idx_sort = np.argsort(couple_dt) # sort by short to long duration

    return idx_1[idx_sort], idx_2[idx_sort], couple_dt[idx_sort]

def pair_images(conn_1, conn_2, thres=20):
    """ Find the closest correspnding points in two lists, within a certain
    bound (given by thres)

    Parameters
    ----------
    conn_1 : numpy.array, size=(k,>=2)
        list with spatial coordinates or any other index
    conn_2 : numpy.array, size=(k,>=2)
        list with spatial coordinates or any other index
    thres : {integer,float}
        search range to consider

    Returns
    -------
    idxConn : numpy.array, size=(l,2)
        array of indices of pairs

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> for p in range(1006,2020):
    >>> plt.plot(np.array([conn_1[idxConn[p,0],0], conn_1[idxConn[p,0],2]]),
                 np.array([conn_1[idxConn[p,0],1], conn_1[idxConn[p,0],3]]) )
    >>> plt.plot(np.array([conn_2[idxConn[p,1],0], conn_1[idxConn[p,1],2]]),
                 np.array([conn_2[idxConn[p,1],1], conn_1[idxConn[p,1],3]]) )
    >>> plt.show()

    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(conn_1[:,0:2])
    distances, indices = nbrs.kneighbors(conn_2[:,0:2])
    IN = distances<thres
    idxConn = np.transpose(np.vstack((np.where(IN)[0], indices[IN])))
    return idxConn

def pair_posts(xy_1, xy_2, thres=20.):
    """ find unique closest pair of two point sets

    Parameters
    ----------
    xy_1 : numpy.array, size=(m,_), dtype=float
        array with coordinates
    xy_2 : numpy.array, size=(m,_), dtype=float
        array with coordinates
    thres : float
        upper limit to still include a pairing

    Returns
    -------
    idx_conn : numpy.array, size=(k,2)
        array with connectivity, based upon the index of xy_1 and xy_2
    """
    def closest_post_index(node, nodes):
        D = cdist([node], nodes, 'sqeuclidean').squeeze()
        idx = np.argmin(D)
        dist = D[idx]
        return idx, dist

    idx_conn = []
    for idx,arr in enumerate(xy_2):
        if xy_1.size==0:
            break
        i,d = closest_post_index(arr, xy_1)
        if d <= thres:
            idx_conn.append([i, idx])
            xy_1[i, ...] = np.inf
    idx_conn = np.array(idx_conn)
    return idx_conn

def merge_ids(dh_mother, dh_child, idx_uni):
    """ merge two collections of shadow edge locations, through their mutual
    identification numbers

    Parameters
    ----------
    dh_mother : {pandas.DataFrame, numpy.recordarray}
        stack of coordinates and times, having at least the following collumns:
            - timestamp : unit=days, date stamp
            - caster_x,caster_y : unit=meter, map coordinate of start of shadow
            - casted_x,casted_y : unit=meter, map coordinate of end of shadow
            - azimuth : unit=degrees, sun orientation
            - zenith : unit=degrees, overhead angle of the sun
            - id : identification
    dh_child : {pandas.DataFrame, numpy.recordarray}
        stack of coordinates and times, having at least the following collumns:
            - timestamp : unit=days, date stamp
            - caster_x,caster_y : unit=meter, map coordinate of start of shadow
            - casted_x,casted_y : unit=meter, map coordinate of end of shadow
            - azimuth : unit=degrees, sun orientation
            - zenith : unit=degrees, overhead angle of the sun
            - id : identification
    idx_uni : numpy.array, size=(m,2)
        list of pairs connection both collections

    Returns
    -------
    dh_stack : {pandas.DataFrame, numpy.recordarray}
        stack of coordinates and times, having at least the following collumns:
            - timestamp : unit=days, date stamp
            - caster_x,caster_y : unit=meter, map coordinate of start of shadow
            - casted_x,casted_y : unit=meter, map coordinate of end of shadow
            - azimuth : unit=degrees, sun orientation
            - zenith : unit=degrees, overhead angle of the sun
            - id : identification
    """
    if type(dh_mother) in (pd.core.frame.DataFrame, ):
        dh_stack = merge_ids_pd(dh_mother, dh_child, idx_uni)
    elif type(dh_mother) in (np.recarray, ):
        dh_stack = merge_ids_rec(dh_mother, dh_child, idx_uni)
    return dh_stack

def merge_ids_rec(dh_mother, dh_child, idx_uni):
    id_present = True if 'id' in dh_mother.dtype.names else False

    id_counter = 1
    if id_present:
        id_counter = np.max(dh_mother['id']) + 1
        id_child = np.zeros(dh_child.shape[0], dtype=int)

        # assign to older id's
        NEW = dh_mother['id'][idx_uni[:, 0]] == 0
        id_child[idx_uni[np.invert(NEW), 1]] = \
            dh_mother['id'][idx_uni[np.invert(NEW), 0]]
        new_ids = id_counter + np.arange(0, np.sum(NEW)).astype(int)
        # create new ids
        id_child[idx_uni[NEW, 1]] = new_ids
        dh_mother['id'][idx_uni[NEW, 0]] = new_ids
        id_child = np.rec.fromarrays([id_child],
                                  dtype=np.dtype([('id', int)]))
        dh_child = merge_arrays((dh_child, id_child),
                                flatten=True, usemask=False)
    else:
        id_mother = np.zeros(dh_mother.shape[0], dtype=int)
        id_child = np.zeros(dh_child.shape[0], dtype=int)
        id_mother[idx_uni[:, 0]] = \
            id_counter + np.arange(0, idx_uni.shape[0]).astype(int)
        id_child[idx_uni[:, 1]] = \
            id_counter + np.arange(0, idx_uni.shape[0]).astype(int)

        id_child = np.rec.fromarrays([id_child],
                                     dtype=np.dtype([('id', int)]))
        dh_child = merge_arrays((dh_child, id_child),
                                flatten=True, usemask=False)
        id_mother = np.rec.fromarrays([id_mother],
                                      dtype=np.dtype([('id', int)]))
        dh_mother = merge_arrays((dh_mother, id_mother),
                                flatten=True, usemask=False)
    dh_stack = stack_arrays((dh_mother, dh_child),
                            asrecarray=True, usemask=False)
    return dh_stack

def merge_ids_pd(dh_mother, dh_child, idx_uni):
    id_present = True if 'id' in dh_mother else False

    id_counter = 1
    if id_present:
        id_counter = np.max(dh_mother['id']) + 1
        id_child = np.zeros(dh_child.shape[0], dtype=int)

        # assign to older id's
        NEW = dh_mother['id'].to_numpy()[idx_uni[:, 0]] == 0
        id_child[idx_uni[np.invert(NEW), 1]] = \
            dh_mother['id'].to_numpy()[idx_uni[np.invert(NEW), 0]]
        dh_child.insert(len(dh_child.columns), 'id', id_child)
        # update mother
        df_header = list(dh_mother)
        df_id = [df_header.index(i) for i in df_header if 'id' in i][0]
        dh_mother.iloc[idx_uni[NEW, 0], [df_id]] = id_counter + \
                    np.arange(0, np.sum(NEW)).astype(int)
    else:
        id_mother = np.zeros(dh_mother.shape[0], dtype=int)
        id_child = np.zeros(dh_child.shape[0], dtype=int)
        id_mother[idx_uni[:, 0]] = \
            id_counter + np.arange(0, idx_uni.shape[0]).astype(int)
        id_child[idx_uni[:, 1]] = \
            id_counter + np.arange(0, idx_uni.shape[0]).astype(int)

        dh_mother.insert(len(dh_mother.columns), 'id', id_mother)
        dh_child.insert(len(dh_child.columns), 'id', id_child)

    dh_stack = pd.concat([dh_mother, dh_child])
    return dh_stack

def match_shadow_ridge(M1, M2, Az1, Az2, geoTransform1, geoTransform2,
                        xy1, xy2, describ='BAS', K=5):

    M1,M2,i1,j1,i2,j2 = pad_images_and_filter_coord_list(M1,M2,
         geoTransform1, geoTransform2, np.vstack((xy1[:,0],xy2[:,0])).T,
         np.vstack((xy1[:,1],xy2[:,1])).T, K, 2*K, same=False)

    if describ in ['CAN', 'NCD']: # sun angles are needed
        Az1 = np.pad(Az1, K,'constant', constant_values=0)
        Az2 = np.pad(Az2, 2*K,'constant', constant_values=0)

    ij2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    xy2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    corr_score = np.zeros((i1.shape[0],1)).astype(np.float16)
    for counter in range(len(i1)):
        M1_sub = M1[i1[counter] - K:i1[counter] + K + 1,
                j1[counter] - K:j1[counter] + K + 1]
        M2_sub = M2[i2[counter] - 2*K:i2[counter] + 2*K + 1,
                j2[counter] - 2*K:j2[counter] + 2*K + 1]

        if describ in ['CAN', 'NCD']:
            Az1_sub = Az1[i1[counter],j1[counter]]
            Az2_sub = Az2[i2[counter],j2[counter]]

        contours = measure.find_contours(M1_sub!=0, level=0.5,
                                        fully_connected='low',
                                        positive_orientation='low', mask=None)
        # find nearest contour and edge point
        idx = 2*K # input a randomly large number
        for i in range(0,len(contours)):
            post_dist = np.sum((contours[i]-K+1)**2, axis=1)**.5
            min_dist = np.min(post_dist)
            if i == 0:
                near = i
                idx = np.argmin(post_dist)
                near_dist = min_dist
            elif min_dist < near_dist:
                near = i
                idx = np.argmin(post_dist)
                near_dist = min_dist
        contour = contours[near] # contour of interest
        if describ == 'BAS':
            doi = beam_angle_statistics(contour[:,0], contour[:,1],
                                        K, xy_id=idx) # point describtor
        elif describ in ['CAN', 'NCD']:
            local_sun = np.array([np.cos(np.deg2rad(Az1_sub)),
                                  np.sin(np.deg2rad(Az1_sub))])
            if describ == 'CAN':
                doi = cast_angle_neighbours(contour[:,0], contour[:,1],
                                            local_sun, K, xy_id=idx) # point describtor
            elif describ == 'NCD':
                doi = neighbouring_cast_distances(contour[:,0], contour[:,1],
                                                  local_sun, K, xy_id=idx)

        contours = measure.find_contours(M2_sub!=0, level=0.5,
                                        fully_connected='low',
                                        positive_orientation='low', mask=None)

        # loop through candidate contours and build describtor, then match
        for i in range(0,len(contours)-1):
            track = contours[i]
            if describ == 'BAS':
                toi = beam_angle_statistics(track[:,0], track[:,1], K)
            elif describ in ['CAN', 'NCD']:
                local_sun = np.array([np.cos(np.deg2rad(Az2_sub)),
                                      np.sin(np.deg2rad(Az2_sub))])
                if describ == 'CAN':
                    toi = cast_angle_neighbours(track[:,0], track[:,1],
                                                local_sun, K)
                elif describ == 'NCD':
                    toi = neighbouring_cast_distances(track[:,0], track[:,1],
                                                      local_sun, K)

            descr_dis = np.sum((toi-doi)**2, axis=1)**.5

            min_dist = np.min(descr_dis)/len(doi)
            if i == 0:
                sim_score = min_dist
                idx = np.argmin(descr_dis)
                di,dj = track[idx,0], track[idx,1]
            elif min_dist<sim_score:
                sim_score = min_dist
                idx = np.argmin(descr_dis)
                di, dj = track[idx,0], track[idx,1]

        corr_score[counter] = sim_score
        ij2_corr[counter,0] = i2[counter] - di
        ij2_corr[counter,1] = j2[counter] - dj

        ij2_corr[counter,0] += 2*K # 2*K compensate for padding
        ij2_corr[counter,1] += 2*K

    xy2_corr[:,0], xy2_corr[:,1] = pix2map(geoTransform1, ij2_corr[:,0], ij2_corr[:,1])
    return xy2_corr, corr_score
# create file with approx shadow cast locations

# refine by matching, and include correlation score
def match_shadow_casts(M1, M2, L1, L2, geoTransform1, geoTransform2,
                       xy1, xy2, scale_12, simple_sh,
                       temp_radius=7, search_radius=22,
                       prepro='bandpass',correlator='cosicorr', subpix='moment',
                       metric='peak_nois'):
    """
    Redirecting arrays to

    Parameters
    ----------
    M1 : numpy.array, size=(m,n,b), dtype=float
        first image array, can be complex.
    M2 : numpy.array, size=(m,n,b), dtype=float
        second image array, can be complex.
    L1 : numpy.array, size=(m,n), dtype=bool
        label of the first image, where True denote elements to ignore.
    L2 : numpy.array, size=(m,n), dtype=bool
        label of the second image, where True denote elements to ignore.
    geoTransform1 : tuple, size={(6,) (8,)}
        affine transformation coefficients of array M1
    geoTransform2 : tuple, size={(6,) (8,)}
        affine transformation coefficients of array M2
    xy1 : np.array, size=(_,2), type=float
        map coordinates of matching centers of array M1
    xy2 : np.array, size=(_,2), type=float
        map coordinates of matching centers of array M2
    scale_12 : float
        estimate of scale change between array M1 & M2
    shear_12 : float
        estimate of shear between array M1 & M2
    temp_radius: integer
        amount of pixels from the center
    search_radius: integer
        amount of pixels from the center
    prepro : {'bandpass' (default), 'raiscos', 'highpass'}
        Specifies which pre-procssing procedure to apply to the imagery:

          * 'bandpass' : bandpass filter
          * 'raiscos' : raised cosine
          * 'highpass' : high pass filter
    correlator : {'norm_corr' (default), 'aff_of', 'cumu_corr',\
                  'sq_diff', 'sad_diff',\
                  'cosi_corr', 'phas_only', 'symm_phas', 'ampl_comp',\
                  'orie_corr', 'mask_corr', 'bina_phas', 'wind_corr',\
                  'gaus_phas', 'upsp_corr', 'cros_corr', 'robu_corr'}
       Method to be used to do the correlation/similarity using either spatial
       measures for the options see "list_spatial_correlators" and
       "list_frequency_correlators"
    subpix : {'tpss' (default), 'svd', 'radon', 'hough', 'ransac', 'wpca',\
              'pca', 'lsq', 'diff', 'gauss_1', 'parab_1', 'moment', 'mass',\
              'centroid', 'blais', 'ren', 'birch', 'eqang', 'trian', 'esinc',\
              'gauss_2', 'parab_2', 'optical_flow'}
        Abbreviation for the sub-pixel localization to be used, for the options
        see "list_phase_estimators" and "list_peak_estimators"
    metric : {'peak_abs' (default), 'peak_ratio', 'peak_rms', 'peak_ener',
              'peak_nois', 'peak_conf', 'peak_entr'}
        Abbreviation for the metric type to be calculated, for the options see
        "list_matching_metrics" for the options

    Returns
    -------
    xy2_corr : np.array, size=(_,2), type=float
        map coordinates of the refined matching centers of array M2
    xy2_corr : np.array, size=(_,2), type=float
        associated scoring metric of xy2_corr
    """
    correlator, subpix = correlator.lower(), subpix.lower()
    frequency_based = list_frequency_correlators()
    spatial_based = list_spatial_correlators()

    # some sub-pixel methods use the peak of the correlation surface,
    # while others need the phase of the cross-spectrum
    phase_based, peak_based = list_phase_estimators(), list_peak_estimators()

    if correlator=='aff_of': # optical flow needs to be of the same size
        search_radius = np.copy(temp_radius)

    M1,M2,i1,j1,i2,j2,IN = pad_images_and_filter_coord_list(M1, M2,
         geoTransform1, geoTransform2, np.vstack((xy1[:,0],xy2[:,0])).T,
         np.vstack((xy1[:,1],xy2[:,1])).T, temp_radius, search_radius,
         same=False)
    L1,L2 = pad_radius(L1, temp_radius), pad_radius(L2, search_radius)
    geoTransformPad2 = ref_trans(geoTransform2, -search_radius, -search_radius)


    ij2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    snr_score = np.zeros((i1.shape[0],1)).astype(np.float16)

    for idx in tqdm(range(len(i1))): # loop through all posts
        A = np.array([[1, simple_sh[idx]],
                      [0, scale_12[idx]]])
        A_inv = np.linalg.inv(A)

        # create templates
        if correlator in frequency_based:
            X_one, Y_one = aff_trans_template_coord(np.eye(2), temp_radius,
                                                    fourier=True)
            X_aff,Y_aff = aff_trans_template_coord(A_inv, search_radius,
                                                    fourier=True)
        else:
            X_one, Y_one = aff_trans_template_coord(np.eye(2), temp_radius,
                                                    fourier=False)
            X_aff, Y_aff = aff_trans_template_coord(A_inv, search_radius,
                                                    fourier=False)

        M1_sub = bilinear_interpolation(M1, i1[idx]+Y_one, j1[idx]+X_one)
        L1_sub = bilinear_interpolation(L1, i1[idx]+Y_one, j1[idx]+X_one)
        M2_sub = bilinear_interpolation(M2, i2[idx]+Y_aff, j2[idx]+X_aff)
        L2_sub = bilinear_interpolation(L2, i2[idx]+Y_aff, j2[idx]+X_aff)

        if correlator in ['aff_of']:
            (di,dj,Aff,snr) = affine_optical_flow(M1_sub, M2_sub)
        else:
            QC = match_translation_of_two_subsets(M1_sub, M2_sub,
                                                  correlator,subpix,
                                                  L1_sub,L2_sub)
        if (subpix in peak_based) or (subpix is None):
            di, dj, score, _ = get_integer_peak_location(QC, metric=metric)
        else:
            di, dj, score = np.zeros((1)), np.zeros((1)), np.zeros((1))
        m0 = np.array([di, dj])

        if subpix in ['optical_flow']:
            # reposition second template
            M2_new = create_template_at_center(M2,
                   i2[idx]-di, j2[idx]-dj,temp_radius)
            # optical flow refinement
            ddi,ddj = simple_optical_flow(M1_sub, M2_new)
            if (abs(ddi)>0.5) or (abs(ddj)>0.5): # divergence
                ddi,ddj = 0,0
        else:
            ddi,ddj = estimate_subpixel(QC, subpix, m0=m0)

        if abs(ddi)<2: di += ddi
        if abs(ddj)<2: dj += ddj

        # adjust for affine transform
        if 'Aff' in locals(): # affine transform is also estimated
            Anew = A*Aff
            dj_rig = dj*Anew[0,0] + di*Anew[0,1]
            di_rig = dj*Anew[1,0] + di*Anew[1,1]
        else:
            dj_rig = dj*A[0,0] + di*A[0,1]
            di_rig = dj*A[1,0] + di*A[1,1]

        snr_score[idx] = score

        ij2_corr[idx,0] = i2[idx] - di_rig
        ij2_corr[idx,1] = j2[idx] - dj_rig

    x2_corr,y2_corr = pix2map(geoTransformPad2, ij2_corr[:,0], ij2_corr[:,1])
    xy2_corr = np.nan * np.ones((IN.shape[0],2))
    xy2_corr[IN,0], xy2_corr[IN,1] = x2_corr, y2_corr
    return xy2_corr, snr_score

def angles2unit(azimuth):
    """ Transforms arguments in degrees to unit vector, in planar coordinates

    Parameters
    ----------
    azimuth : dtype=float, unit=degrees
        argument with clockwise angle direction

    Returns
    -------
    xy : numpy.array, size=(2,1), dtype=float
        unit vector in map coordinates, that is East and Nordwards direction

    Notes
    -----
    The angle(s) are declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

    """
    assert isinstance(azimuth, (float, np.ndarray))

    x, y = np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))

    if x.ndim==1:
        xy = np.stack((x[:,np.newaxis], y[:,np.newaxis]), axis=1)
    elif x.ndim==2:
        xy = np.dstack((x, y))
    return xy

def get_intersection(xy_1, xy_2, xy_3, xy_4):
    """ given two points per line, estimate the intersection

    Parameters
    ----------
    xy_1,xy_2 : numpy.array, size=(m,2)
        array with 2D coordinate of start and end from the first line.
    xy_3, xy_4 : numpy.array, size=(m,2)
        array with 2D coordinate of start and end from the second line.

    Returns
    -------
    xy : numpy.array, size=(m,2)
        array with 2D coordinate of intesection

    Notes
    -----
    The geometry can be seen like this:

    .. code-block:: text

                 + xy_1
                 |
                 |    + xy_4
                 |   /
                 |  /
                 | /
                 |/
                 x xy
                /|
               / |
              /  +xy_2
             /
            + xy_3
    """
    assert (isinstance(xy_1, (float, np.ndarray)) and
            isinstance(xy_2, (float, np.ndarray)))
    assert (isinstance(xy_3, (float, np.ndarray)) and
            isinstance(xy_4, (float, np.ndarray)))
    if type(xy_1) in (np.ndarray,):
        assert len(set({xy_1.shape[0], xy_2.shape[0],
                        xy_3.shape[0], xy_4.shape[0], })) == 1, \
            ('please provide arrays of the same size')

    x_1,x_2,x_3,x_4 = xy_1[:,0], xy_2[:,0], xy_3[:,0], xy_4[:,0]
    y_1,y_2,y_3,y_4 = xy_1[:,1], xy_2[:,1], xy_3[:,1], xy_4[:,1]

    numer_1, numer_2 = x_1*y_2 - y_1*x_2, x_3*y_4 - y_3*x_4
    dx_12, dy_12 = x_1 - x_2, y_1 - y_2
    dx_34, dy_34 = x_3 - x_4, y_3 - y_4
    denom = (dx_12*dy_34 - dy_12*dx_34)

    x = np.divide( numer_1 * dx_34 - dx_12 * numer_2, denom,
                   where=denom!=0, out=np.zeros_like(denom))
    y = np.divide( numer_1 * dy_34 - dy_12 * numer_2, denom,
                   where=denom!=0, out=np.zeros_like(denom))

    xy = np.stack((x,y)).T
    return xy

def _get_zenith_from_sun(sun):
    if type(sun) in (float,): return np.deg2rad(sun)
    if sun.size==1: # only zenith angles are given
        zn = np.deg2rad(sun)
    elif sun.shape[1]==2: # both azimuth an zenith are given
        zn = np.deg2rad(sun[:,1])
    return zn

def get_elevation_difference(sun_1, sun_2, xy_1, xy_2, xy_t):
    """ use geometry to get the relative elevation difference

    Parameters
    ----------
    sun_1 : numpy.array, size=(k,2), unit=degrees
        sun vector in array of first instance
    sun_2 : numpy.array, size=(k,2), unit=degrees
        sun vector in array of second instance
    xy_1 : numpy.array, size=(k,2), unit=meters
        map location of shadow cast at first instance
    xy_2 : numpy.array, size=(k,2), unit=meters
        map location of shadow cast at second instance
    xy_t : numpy.array, size=(k,2), unit=meters
        map location of common shadow caster

    Returns
    -------
    dh : numpy.array, size=(k,1), unit=meters
        elevation difference between coordinates in "xy_1" & "xy_2"
    """
    assert len(set({sun_1.shape[0], sun_2.shape[0], xy_t.shape[0],
                    xy_1.shape[0],  xy_2.shape[0],}))==1, \
        ('please provide arrays of the same size')

    zn_1 = _get_zenith_from_sun(sun_1)
    zn_2 = _get_zenith_from_sun(sun_2)

    # get planar distance between
    dist_1 = np.hypot(xy_1[:,0]-xy_t[:,0], xy_1[:,1]-xy_t[:,1])
    dist_2 = np.hypot(xy_2[:,0]-xy_t[:,0], xy_2[:,1]-xy_t[:,1])

    # get elevation difference
    dh = np.tan(zn_1)*dist_1 - np.tan(zn_2)*dist_2
    return dh

def get_caster(sun_1, sun_2, xy_1, xy_2):
    """ estimate the location of the common caster

    Parameters
    ----------
    sun_1 : numpy.array, size=(m,), units=degrees
        illumination direction at instance 1
    sun_2 : numpy.array, size=(m,), units=degrees
        illumination direction at instance 2
    xy_1 : numpy.array, size=(m,2)
        spatial location at instance 1
    xy_2 : numpy.array, size=(m,2)
        spatial location at instance 1

    Returns
    -------
    caster_new : np.array, size=(m,2)
        estimated spatial location of common caster
    """
    assert len(set({sun_1.shape[0], sun_2.shape[0],
                    xy_1.shape[0],  xy_2.shape[0],}))==1, \
        ('please provide arrays of the same size')
    sun_1, sun_2 = np.deg2rad(sun_1), np.deg2rad(sun_2)

    caster_new = get_intersection(xy_1, xy_1+np.stack((np.sin(sun_1),
                                                       np.cos(sun_1)), axis=1),
                                  xy_2, xy_2+np.stack((np.sin(sun_2),
                                                       np.cos(sun_2)), axis=1))
    return caster_new

def get_shadow_rectification(casters, post_1, post_2, az_1, az_2):
    """
    estimate relative scale and shear of templates, assuming a planar surface

    Parameters
    ----------
    casters : numpy.array, size=(k,2), unit=meters
        coordinates of the locations that casts shadow
    post_1 : numpy.array, size=(k,2), unit=meters
        coordinates of the locations that receives shadow at instance 1
    post_2 : numpy.array, size=(k,2), unit=meters
        coordinates of the locations that receives shadow at instance 2
    az_1 : numpy.array, size=(k,1), unit=degrees
        illumination orientation of the caster at instance 1
    az_2 : numpy.array, size=(k,1), unit=degrees
        illumination orientation of the caster at instance 2

    Returns
    -------
    scale_12 : numpy.array, size=(k,1), unit=[]
        relative scale difference between instance 1 & 2
    simple_sh : numpy.array, size=(k,1), unit=[]
        relative shear between instance 1 & 2
    """
    assert len(set({post_1.shape[0], post_2.shape[0], casters.shape[0],
                    az_1.shape[0], az_2.shape[0],}))==1, \
        ('please provide arrays of the same size')

    # get planar distance between
    dist_1 = np.hypot(post_1[:,0] - casters[:,0], post_1[:,1] - casters[:,1])
    dist_2 = np.hypot(post_2[:,0] - casters[:,0], post_2[:,1] - casters[:,1])
    scale_12 = np.divide(dist_2, dist_1, where=dist_1!=0)

    # sometimes caster and casted are on the same location,
    # causing very large scaling, just reduce this
    scale_12[scale_12 < .2] = 1.
    scale_12[scale_12 > 5] = 1.

    simple_sh = np.tan(np.radians(az_1 - az_2))
    return scale_12, simple_sh
