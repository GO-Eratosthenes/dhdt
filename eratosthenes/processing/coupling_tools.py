import os
import glob
import math
import warnings
warnings.filterwarnings('once')

import numpy as np

from sklearn.neighbors import NearestNeighbors
from skimage import measure

from ..generic.mapping_tools import map2pix, pix2map, rotMat, \
    castOrientation, ref_trans
from ..generic.mapping_io import read_geo_image
from ..generic.handler_im import bilinear_interpolation

from ..preprocessing.read_s2 import read_sun_angles_s2

from .matching_tools import \
    affine_optical_flow, simple_optical_flow, \
    normalized_cross_corr, sum_sq_diff, cumulative_cross_corr, \
    perdecomp, \
    cosi_corr, phase_only_corr, symmetric_phase_corr, amplitude_comp_corr, \
    orientation_corr, phase_corr, cross_corr, masked_cosine_corr, \
    tpss, hoge, \
    get_top_gaussian, get_top_parabolic, get_top_weighted, \
    binary_orientation_corr, masked_corr, \
    beam_angle_statistics, cast_angle_neighbours, neighbouring_cast_distances

def make_time_pairing(acq_time, 
                      t_min=np.timedelta64(300, 'ms').astype('timedelta64[ns]'),
                      t_max=np.timedelta64(3, 's').astype('timedelta64[ns]')):        
    """ construct pairing based upon time difference

    Parameters
    ----------
    acq_time : np.array, size=(_,1), type=np.timedelta64[ns]
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
    couple_dt : np.array, size=(_,1), type=np.timedelta64[ns]
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

def couple_pair(file1, file2, bbox, co_name, co_reg, rgi_id=None, 
                reg='binary', prepro='bandpass', \
                match='cosicorr', boi=np.array([4]), \
                processing='stacking',\
                subpix='tpss'):
    """
    

    :param dat_path:    STRING
        directory to dump results
    :param fname1:      STRING
        path to first image(ry)
    :param fname2:      STRING
        path to second image(ry)
    :param bbox:        NP.ARRAY [4x1]
    :param co_name:
    :param co_reg:
    :param rgi_id:      STRING
        code of the glacier of interest following the Randolph Glacier Inventory
    :param reg:         STRING
        asdasdasd
          :options : binary - asdsa
                     metadata - estimate scale and shear from geometry 
                     and metadata
               else: don't include affine corrections
    :param prepro:
          :options :
    :param match:
          :options : norm_corr - normalized cross correlation
                     aff_of - affine optical flow
                     cumu_corr - cumulative cross correlation
                     sq_diff - sum of squared difference
                     cosi_corr - cosicorr
                     phas_only - phase only correlation
                     symm_phas - symmetric phase correlation
                     ampl_comp - 
                     orie_corr - orientation correlation
                     mask_corr - masked normalized cross correlation
                     bina_phas - 
    :param boi:        NP.ARRAY 
        list of bands of interest
    :param processing: STRING
        matching can be done on a single image pair, or by matching several 
        bands and combine the result (stacking)
          :options : stacking - using multiple bands in the matching
                     shadow - using the shadow transfrom in the given folder
          :else :    using the image files provided in 'file1' & 'file2'
    :param subpix:     STRING
        matching is done at pixel resolution, but subpixel localization is
        possible to increase this resolution
          :options : tpss - 
                     svd - 
                     gauss_1 - 1D Gaussian fitting
                     parab_1 - 1D parabolic fitting
                     weighted - weighted polynomial fitting
                     optical_flow - optical flow refinement
                     
    :return post1:
    :return post2_corr:
    :return casters:
    :return dh:
    """
    
    # get start and finish points of shadow edges
    im1_dir, im1_name = os.path.split(file1)
    im2_dir, im2_name = os.path.split(file2)
    
    # naming of the files
    shw_fname = "shadow.tif"           # image with shadow transform / enhancement
    label_fname = "labelPolygons.tif"  # numbered image with all the shadow polygons
    if rgi_id == None:
        conn_fname = "conn.txt"        # text file listing coordinates of shadow casting and casted points
    else:
        conn_fname = f"conn-rgi{rgi_id[0]:08d}.txt"
    conn1 = np.loadtxt(fname = os.path.join(im1_dir, conn_fname))
    conn2 = np.loadtxt(fname = os.path.join(im2_dir, conn_fname))
    
    # compensate for coregistration
    if co_name == None:
        print('no coregistration is used or applied')
    else:
        coid1,coid2 = co_name.index(im1_name), co_name.index(im2_name)
        coDxy = co_reg[coid1]-co_reg[coid2]
        
        conn2[:,0] += coDxy[0]*10 # OBS! transform to meters?
        conn2[:,1] += coDxy[1]*10

    idxConn = pair_images(conn1, conn2)
    # connected list
    casters = conn1[idxConn[:,1],0:2]
    post1 = conn1[idxConn[:,1],2:4].copy() # cast location in xy-coordinates for t1
    post2 = conn2[idxConn[:,0],2:4].copy() # cast location in xy-coordinates for t2
    # caster2 = conn2[idxConn[:,0],0:2]
    
    # refine through feature descriptors 
    if match in ['feature']: # using the shadow polygons
        
    
        (L1, crs, geoTransform1, targetprj) = read_geo_image(
                os.path.join(im1_dir, label_fname))
        (L2, crs, geoTransform2, targetprj) = read_geo_image(
                os.path.join(im2_dir, label_fname))
        
        # use polygon descriptors
        #describ='NCD' # BAS
        if subpix in ['CAN', 'NCD']: # sun angle needed?
            (sun1_Zn,sun1_Az) = read_sun_angles_s2(os.path.join(im1_dir, im1_name))
            (sun2_Zn,sun2_Az) = read_sun_angles_s2(os.path.join(im2_dir, im1_name))
            
        else:
            sun1_Az, sun2_Az = [], []
            
        post2_new, euc_score = match_shadow_ridge(L1, L2, sun1_Az, sun2_Az,
                                              geoTransform1, geoTransform2, 
                                              post1, post2, subpix, K=10)                
    else: 
        # refine through pattern matching
        
        if reg in ['binary']: # get shadow classification if needed        
            (L1, crs, geoTransform1, targetprj) = read_geo_image(
                    os.path.join(im1_dir, label_fname))
            (L2, crs, geoTransform2, targetprj) = read_geo_image(
                    os.path.join(im2_dir, label_fname))
        else:
            L1, L2 = [], []
        
        if processing in ['shadow']:
            (M1, crs, geoTransform1, targetprj) = read_geo_image(
                    os.path.join(im1_dir, shw_fname)) # shadow transform template
            (M2, crs, geoTransform2, targetprj) = read_geo_image(
                    os.path.join(im2_dir, shw_fname)) # shadow transform search space
        elif processing in ['stacking']: # multi-spectral stacking
        
            M1,geoTransform1 = get_stack_out_of_dir(im1_dir,boi,bbox)
            M2,geoTransform2 = get_stack_out_of_dir(im2_dir,boi,bbox)
        else: # load the files given
            (M1, crs, geoTransform1, targetprj) = read_geo_image(
                    os.path.join(im1_dir, im1_name))
            (M2, crs, geoTransform2, targetprj) = read_geo_image(
                    os.path.join(im2_dir, im2_name))
        
        if (reg in ['binary']) | (prepro in ['sundir']): # get sun angles if needed
            (sun1_Zn, sun1_Az) = read_sun_angles_s2(os.path.join(dat_path, fname1))
            (sun2_Zn, sun2_Az) = read_sun_angles_s2(os.path.join(dat_path, fname2))
            if bbox is not None: # clip when sub-set of imagery is used
                sun1_Zn = sun1_Zn[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                sun1_Az = sun1_Az[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                sun2_Zn = sun2_Zn[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                sun2_Az = sun2_Az[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                
            if prepro in ['sundir']: # put emphasis on sun direction in the imagery
                if M1.ndim==3: # multi-spectral
                    for i in range(M1.ndim):
                        M1[:,:,i] = castOrientation(M1[:,:,i], \
                                                    sun1_Az[bbox[0]:bbox[1], \
                                                            bbox[2]:bbox[3]])
                else:
                    M1 = castOrientation(M1, sun1_Az[bbox[0]:bbox[1], \
                                                     bbox[2]:bbox[3]])    
                if M2.ndim==3: # multi-spectral
                    for i in range(M2.ndim):
                        M2[:,:,i] = castOrientation(M2[:,:,i], \
                                                    sun2_Az[bbox[0]:bbox[1], \
                                                            bbox[2]:bbox[3]])
                else:
                    M2 = castOrientation(M2, sun2_Az[bbox[0]:bbox[1], \
                                                     bbox[2]:bbox[3]])
        else:
            sun1_Az, sun2_Az = [], []
        
        if reg in ['metadata']:
            # estimate relative scale of templates
            # get planar distance between 
            dist_1 = np.sqrt((post1[:,0]-casters[:,0])**2 + \
                             (post1[:,1]-casters[:,1])**2)
            dist_2 = np.sqrt((post2[:,0]-casters[:,0])**2 + \
                             (post2[:,1]-casters[:,1])**2)
            scale_12 = np.divide(dist_2, dist_1)
            
            # sometimes caster and casted are on the same location, 
            # causing very large scalings, just reduce this
            scale_12[scale_12<.2] = 1.
            scale_12[scale_12>5] = 1.
            
            
            az_1, az_2 = conn1[idxConn[:,1],4], conn2[idxConn[:,0],4]
            simple_sh = np.tan(np.radians(az_1-az_2))
            
            #breakpoint()
        else: # start with rigid displacement
            scale_12 = np.ones((np.shape(post1)[0]))
            simple_sh = np.zeros((np.shape(post1)[0]))
        
        temp_radius=7 
        search_radius=22
        post2_corr, corr_score = match_shadow_casts(M1, M2, L1, L2,
                                                    sun1_Az, sun2_Az,
                                                    geoTransform1, geoTransform2, 
                                                    post1, post2,
                                                    scale_12, simple_sh,
                                                    temp_radius, search_radius,
                                                    reg, prepro,
                                                    match, processing,
                                                    subpix)
    post1 = conn1[idxConn[:,1],2:4] # cast location in xy-coordinates for t1
    post2 = conn2[idxConn[:,0],2:4]
    # extract elevation change
    sun_1 = conn1[idxConn[:,1],4:6]
    sun_2 = conn2[idxConn[:,0],4:6]

    if match in ['feature']:
        IN = euc_score<0.7
        post2_corr = post2_new[IN.flatten(),:]
    else:    
        IN = corr_score>0.8
        post2_corr = post2_corr[IN.flatten(),:]

    post1 = post1[IN.flatten(),:]
    post2 = post2[IN.flatten(),:]
    sun_1 = sun_1[IN.flatten(),:]
    sun_2 = sun_2[IN.flatten(),:]
    casters = casters[IN.flatten(),:]
    
    dh = get_elevation_difference(sun_1, sun_2, post1, post2_corr, casters)
    return post1, post2_corr, casters, dh, crs
    
def pair_images(conn1, conn2, thres=20):
    """
    Find the closest correspnding points in two lists, 
    within a certain bound (given by thres)
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(conn1[:,0:2])
    distances, indices = nbrs.kneighbors(conn2[:,0:2])
    IN = distances<thres
    idxConn = np.transpose(np.vstack((np.where(IN)[0], indices[IN])))
    
    # import matplotlib.pyplot as plt
    # plt.plot(conn1[:,0],conn1[:,1]), plt.plot(conn2[:,0],conn2[:,1]), plt.show()
    
    # for p in range(1006,2020):
    #     plt.plot(np.array([conn1[idxConn[p,0],0], conn1[idxConn[p,0],2]]), \
    #              np.array([conn1[idxConn[p,0],1], conn1[idxConn[p,0],3]]) )
    #     plt.plot(np.array([conn2[idxConn[p,1],0], conn1[idxConn[p,1],2]]), \
    #              np.array([conn2[idxConn[p,1],1], conn1[idxConn[p,1],3]]) )
    # plt.show()
    return idxConn

def match_shadow_ridge(M1, M2, Az1, Az2, geoTransform1, geoTransform2, 
                        xy1, xy2, describ='BAS', K=5):
    i1, j1 = map2pix(geoTransform1, xy1[:,0], xy1[:,1])
    i2, j2 = map2pix(geoTransform2, xy2[:,0], xy2[:,1])
    
    i1, j1 = np.round(i1).astype(np.int64), np.round(j1).astype(np.int64)
    i2, j2 = np.round(i2).astype(np.int64), np.round(j2).astype(np.int64)
    
    # remove posts outside image   
    IN = np.logical_and.reduce((i1>=0, i1<=M1.shape[0], j1>=0, j1<=M1.shape[1],
                                i2>=0, i2<=M2.shape[0], j2>=0, j2<=M2.shape[1]))
    
    i1, j1, i2, j2 = i1[IN], j1[IN], i2[IN], j2[IN]
    
    # extend image size, so search regions at the border can be used as well
    M1 = np.pad(M1, K,'constant', constant_values=0)
    i1 += K
    j1 += K
    M2 = np.pad(M2, 2*K,'constant', constant_values=0)
    i2 += 2*K
    j2 += 2*K
       
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

def match_shadow_casts(M1, M2, L1, L2, sun1_Az, sun2_Az,
                       geoTransform1, geoTransform2,
                       xy1, xy2, 
                       scale_12, simple_sh,
                       temp_radius=7, search_radius=22, 
                       reg='binary', prepro='bandpass',
                       match='cosicorr', processing='stacking',
                       subpix='tpss'):
    """
    Redirecting arrays to 

    :param M1:          NP.ARRAY [_,_,b]
    :param M2:          NP.ARRAY [_,_,b]
    :param L1:          NP.ARRAY [_,_]
    :param L2:          NP.ARRAY [_,_] 
    :param M1:          NP.ARRAY [_,_]
    :param M1:          NP.ARRAY [_,_]  
    :param geoTransform1: NP.ARRAY [1,6]
        coordinate metadata of array no.1    
    :param geoTransform2: NP.ARRAY [1,6]
        coordinate metadata of array no.2  
    :param xy1:         NP.ARRAY [_,2]
        map coordinates of matching centers of array no.1
    :param xy1:         NP.ARRAY [_,2]
        map coordinates of matching centers of array no.2
    :param scale_12:    FLOAT
        estimate of scale change between array no.1&2
    :param shear_12:    FLOAT
        estimate of shear between array no.1&2        
    :param temp_radius: INTEGER
        amount of pixels from the center
    :param search_radius: INTEGER
        amount of pixels from the center    
    :param reg:         STRING
        asdasdasd
          :options : binary - asdsa
               else: estimate scale and shear from geometry and metadata
    :param prepro:
          :options : bandpass - bandpass filter
                     raiscos - raised cosine
                     highpass - high pass filter
    :param match:
          :options : norm_corr - normalized cross correlation
                     aff_of - affine optical flow
                     cumu_corr - cumulative cross correlation
                     sq_diff - sum of squared difference
                     cosi_corr - cosicorr
                     phas_only - phase only correlation
                     symm_phas - symmetric phase correlation
                     ampl_comp - 
                     orie_corr - 
                     mask_corr -
                     bina_phas - 
    :param subpix:     STRING
        matching is done at pixel resolution, but subpixel localization is
        possible to increase this resolution
          :options : tpss - 
                     svd - 
                     gauss_1 - 1D Gaussian fitting
                     parab_1 - 1D parabolic fitting
                     weighted - weighted polynomial fitting
                     optical_flow - optical flow refinement
                     
    :return xy2_corr:
    :return snr_score:
    """
    match,subpix = match.lower(), subpix.lower()
    
    # some sub-pixel methods use the peak of the correlation surface, 
    # while others need the cross-spectrum
    peak_calc = ['norm_corr', 'cumu_corr', 'sq_diff', 'mask_corr']
    phase_based = ['tpss','svd'] 
    peak_based = ['gauss_1', 'parab_1', 'weighted']
    
    if (subpix in phase_based) & (match in peak_calc):
         # some combinations are not possible, reset to default
         warnings.warn('given combination of matching-subpix is not possible, using default')
         match = 'cosicorr'
    if match=='aff_of': # optical flow needs to be of the same size
        search_radius = np.copy(temp_radius)
       
    i1, j1 = map2pix(geoTransform1, xy1[:,0], xy1[:,1])
    i2, j2 = map2pix(geoTransform2, xy2[:,0], xy2[:,1])
    
    i1, j1 = np.round(i1).astype(np.int64), np.round(j1).astype(np.int64)
    i2, j2 = np.round(i2).astype(np.int64), np.round(j2).astype(np.int64)
    
    # remove posts outside image   
    IN = np.logical_and.reduce((i1>=0, i1<=M1.shape[0], 
                                j1>=0, j1<=M1.shape[1],
                                i2>=0, i2<=M2.shape[0], 
                                j2>=0, j2<=M2.shape[1]))
    
    i1, j1, i2, j2 = i1[IN], j1[IN], i2[IN], j2[IN]
    
    # extend image size, so search regions at the border can be used as well
    M1 = pad_radius(M1, temp_radius)
    i1 += temp_radius
    j1 += temp_radius
    M2 = pad_radius(M2, search_radius)
    i2 += search_radius
    j2 += search_radius

    
    ij2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    xy2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    snr_score = np.zeros((i1.shape[0],1)).astype(np.float16)
    
    for counter in range(len(i1)): # [1234]: # loop through all posts
       
        # deformation compensation
        if reg in ['binary']:
            # construct binary templates
            az1_sub = sun1_Az[i1[counter],j1[counter]] 
            d_elong = 0 # move center of template into the direction of the caster
            i1_c = np.round(i1[counter] + \
                            (np.cos(np.degrees(az1_sub))*d_elong)).astype('int')
            j1_c = np.round(j1[counter] - \
                            (np.sin(np.degrees(az1_sub))*d_elong)).astype('int')
            
            L1_sub = L1[i1_c-search_radius : i1_c+search_radius+1,
                    j1_c-search_radius : j1_c+search_radius+1]
            
            
            x = np.arange(-search_radius, search_radius+1)
            y = np.arange(-search_radius, search_radius+1)
            
            X = np.repeat(x[np.newaxis,:], y.shape[0],axis=0)
            Y = np.repeat(y[:,np.newaxis], x.shape[0], axis=1)
            
            Xr = +np.cos(np.degrees(az1_sub))*X +np.sin(np.degrees(az1_sub))*Y
            Yr = -np.sin(np.degrees(az1_sub))*X +np.cos(np.degrees(az1_sub))*Y
            sun_along = np.abs(Xr)<(temp_radius/2) # sides lobe
            sun_top = Yr<=0 # upper part of the template
            
            B1_sub = (sun_along & sun_top & L1_sub)
            B1_sub = np.logical_or(B1_sub, np.flip(np.flip(B1_sub, 0),1)) 
            
            az2_sub = sun2_Az[i2[counter],j2[counter]] 
            i2_c = np.round(i2[counter] + \
                            (np.cos(np.degrees(az2_sub))*d_elong)).astype('int')
            j2_c = np.round(j2[counter] - \
                            (np.sin(np.degrees(az2_sub))*d_elong)).astype('int')
            
            L2_sub = L2[i2_c-search_radius : i2_c+search_radius+1,
                    j2_c-search_radius : j2_c+search_radius+1]
            
            Xr = +np.cos(np.degrees(az2_sub))*X +np.sin(np.degrees(az2_sub))*Y
            Yr = -np.sin(np.degrees(az2_sub))*X +np.cos(np.degrees(az2_sub))*Y
            sun_along = np.abs(Xr)<(temp_radius/2) # sides lobe
            sun_top = Yr<=0 # upper part of the template
            
            B2_sub = (sun_along & sun_top & L2_sub)
            B2_sub = np.logical_or(B2_sub, np.flip(np.flip(B2_sub, 0),1)) 
            
            
            breakpoint()
        else: # estimate scale and shear from geometry
            A = np.array([[1, simple_sh[counter]], 
                          [0, scale_12[counter]]]) 
            A_inv = np.linalg.inv(A)
            x = np.arange(-search_radius, search_radius+1)
            y = np.arange(-search_radius, search_radius+1)
            
            X = np.repeat(x[np.newaxis,:], y.shape[0],axis=0)
            Y = np.repeat(y[:,np.newaxis], x.shape[0], axis=1)
            
            X_aff = X*A_inv[0,0] + Y*A_inv[0,1]
            Y_aff = X*A_inv[1,0] + Y*A_inv[1,1]
        
        # create templates
        M1_sub = create_template(M1,i1[counter],j1[counter],temp_radius)
        M2_sub = bilinear_interpolation(M2, \
                                        j2[counter]+X_aff, \
                                        i2[counter]+Y_aff)
        # reduce edge effects in frequency space
        if match in ['cosi_corr', 'phas_only', 'symm_phas', 'orie_corr', \
                     'mask_corr', 'bina_phas']:
            (M1_sub,_) = perdecomp(M1_sub)
            (M2_sub,_) = perdecomp(M2_sub)
        
        # translational matching/estimation
        if match in ['norm_corr']:
            C = normalized_cross_corr(M1_sub, M2_sub)    
        elif match in ['cumu_corr']:
            C = cumulative_cross_corr(M1_sub, M2_sub)
        elif match in ['sq_diff']:
            C = sum_sq_diff(M1_sub, M2_sub)
        elif match in ['cosi_corr']:
            (Q, W, m0) = cosi_corr(M1_sub, M2_sub)
        elif match in ['phas_only']:
            Q = phase_only_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['symm_phas']:
            Q = symmetric_phase_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['ampl_comp']:
            Q = amplitude_comp_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['orie_corr']:
            Q = orientation_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['mask_corr']:
            C = masked_corr(M1_sub, M2_sub, L1_sub, L2_sub)
        elif match in ['bina_phas']:
            Q = binary_orientation_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['aff_of']:
            (di,dj,Aff,snr) = affine_optical_flow(M1_sub, M2_sub)

        if 'C' in locals():
            max_score = np.amax(C)
            snr = max_score/np.mean(C)
            ij = np.unravel_index(np.argmax(C), C.shape)
            di, dj = ij[::-1]
            m0 = np.array([di, dj])
        
        if match!='aff_of':
            # sub-pixel estimation    
            if subpix in ['tpss']:
                if 'W' not in locals():
                    breakpoint()
                (m,snr) = tpss(Q, W, m0)
                ddi, ddj = m[0], m[1]
            elif subpix in ['svd']:
                if 'W' not in locals():
                    breakpoint()
                (ddi,ddj) = hoge(Q, W)
            elif subpix in ['gauss_1']:
                (ddi, ddj) = get_top_gaussian(C, top=m0)
            elif subpix in ['parab_1']:
                (ddi, ddj) = get_top_parabolic(C, top=m0)    
            elif subpix in ['weighted']:
                (ddi, ddj) = get_top_weighted(C, ds=1, top=m0)   
            elif subpix in ['optical_flow']:
                max_score = np.amax(C)
                snr = max_score/np.mean(C)
                ij = np.unravel_index(np.argmax(C), C.shape)
                di, dj = ij[::-1]
                
                # reposition second template
                M2_new = create_template(M2,i2[counter]-di,
                         j1[counter]-dj,temp_radius)
                
                # optical flow refinement
                ddi,ddj = simple_optical_flow(M1_sub, M2_new)
                
                if (abs(ddi)>0.5) or (abs(ddj)>0.5): # divergence
                    ddi = 0
                    ddj = 0               
            else: # simple highest score extaction
                max_score = np.amax(C)
                snr = max_score/np.mean(C)
                ij = np.unravel_index(np.argmax(C), C.shape)
                di, dj = ij[::-1]
        
        # remove padding if cross correlation is used> is already done, see below
        # if 'C' in locals():    
        #     di -= (search_radius - temp_radius)
        #     dj -= (search_radius - temp_radius)
        
        if abs(ddi)<2:
            di += ddi
        if abs(ddj)<2:
            dj += ddj

        # adjust for affine transform
        if 'Aff' in locals(): # affine transform is also estimated
            Anew = A*Aff
            dj_rig = dj*Anew[0,0] + di*Anew[0,1]
            di_rig = dj*Anew[1,0] + di*Anew[1,1]
        else:
            dj_rig = dj*A[0,0] + di*A[0,1]
            di_rig = dj*A[1,0] + di*A[1,1]                   
            
        snr_score[counter] = snr
            
        ij2_corr[counter,0] = i2[counter] - di_rig
        ij2_corr[counter,1] = j2[counter] - dj_rig
        
        #ij2_corr[counter,0] -= search_radius # compensate for padding
        #ij2_corr[counter,1] -= search_radius
        
    xy2_corr[:,0], xy2_corr[:,1] = pix2map(geoTransform1, ij2_corr[:,0], ij2_corr[:,1])
    
    return xy2_corr, snr_score

def get_stack_out_of_dir(im_dir,boi,bbox):
    """
    Create a stack of images from different bands, with at specific bounding box
    
    :param im_dir:        STRING 
        path to the directory of interest
    :param boi:           NP.ARRAY 
        list of bands of interest
    :param bbox:          NP.ARRAY  [1,4]
        list with outer coordinates
        
    :return M:            NP.ARRAY [_,_,boi]
        array with data from images in folder
    :return geoTransform: NP.ARRAY [1,6]
        coordinate metadata of array
    """
    # create empty stack
    M = np.zeros((bbox[1]-bbox[0], bbox[3]-bbox[2], len(boi))) 
    
    for i in range(len(boi)):
        # find image
        find_str = os.path.join(im_dir, f'*{boi[i]:02d}.jp2')
        im_file = glob.glob(find_str)
        if len(im_file)==0: # imagery might be tiff instead
            find_str = os.path.join(im_dir, f'*{boi[i]:02d}.tif')
            im_file = glob.glob(find_str)
            
        # read image
        if len(im_file)!=0:
            (im_bnd,crs, geoTransform, targetprj) = read_geo_image(
                im_file[0])
        
        # stack image    
        M[:,:,i] = im_bnd[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    geoTransform = ref_trans(geoTransform,bbox[0],bbox[2])    
    return M, geoTransform

def create_template(I,i,j,radius):
    if I.ndim==3:
        I_sub = I[i-radius : i+radius, j-radius:j+radius, :]
    else:                        
        I_sub = I[i-radius : i+radius, j-radius:j+radius]
    return I_sub

def pad_radius(I, radius):
    '''
    add extra boundary to array, so templates can be extraxted 
    '''
    if I.ndim==3:
        I = np.pad(I, \
                   ((radius,radius), (radius,radius), (0,0)), \
                   'constant', constant_values=0)
    else:
        I = np.pad(I, \
            ((radius,radius), (radius,radius)), \
            'constant', constant_values=0)
    return I

# simple image matching
def match_pair(I1, I2, M1, M2, geoTransform1, geoTransform2, \
               temp_radius, search_radius, Xgrd, Ygrd, \
               kernel='kroon', \
               prepro='bandpass', match='cosicorr', subpix='tpss', \
               processing='stacking', boi=np.array([]) ):
    """
    simple image mathcing routine

    :param I1:          NP.ARRAY [_,_] | [_,_,_]
        intensity array,  to be used for the matching
    :param I2:          NP.ARRAY [_,_] | [_,_,_]
        intensity array,  to be used for the matching
    :param M1:          NP.ARRAY [_,_]
        mask array, highlighting where the information is in I1
    :param M2:          NP.ARRAY [_,_]
        mask array, highlighting where the information is in I2
    :param geoTransform1: NP.ARRAY [1,6]
        georeference transform of I1
    :param geoTransform2: NP.ARRAY [1,6]
        georeference transform of I2
    :param kernel:
          :options :
                     sobel - 
                     kroon - 
                     scharr - Scharr, 2000 in "Optimal operators in digital 
                     image processing"
                     robinson- 
                     kayyali -
                     prewitt -
                     simoncelli - from Farid and Simoncelli, 1997 in "Optimally 
                     rotation-equivariant directional derivative kernels"
    :param prepro:
          :options : bandpass - bandpass filter
                     raiscos - raised cosine
                     highpass - high pass filter
    :param match:
          :options : norm_corr - normalized cross correlation
                     aff_of - affine optical flow
                     cumu_corr - cumulative cross correlation
                     sq_diff - sum of squared difference
                     cosi_corr - cosicorr
                     phas_only - phase only correlation
                     symm_phas - symmetric phase correlation
                     ampl_comp - 
                     orie_corr - orientation correlation
                     mask_corr - masked normalized cross correlation
                     bina_phas - 
    :param processing: STRING
        matching can be done on a single image pair, or by matching several 
        bands and combine the result (stacking)
          :options : stacking - using multiple bands in the matching
                     shadow - using the shadow transfrom in the given folder
          :else :    using the image files provided in 'file1' & 'file2'
    :param boi:        NP.ARRAY 
        list of bands of interest
    :param subpix:     STRING
        matching is done at pixel resolution, but subpixel localization is
        possible to increase this resolution
          :options : tpss - 
                     svd - 
                     gauss_1 - 1D Gaussian fitting
                     parab_1 - 1D parabolic fitting
                     weighted - weighted polynomial fitting
                     optical_flow - optical flow refinement
                     
    :return post1:
    :return post2_corr:
    :return casters:
    :return dh:
    """
    match,subpix = match.lower(), subpix.lower()
    
    # some sub-pixel methods use the peak of the correlation surface, 
    # while others need the cross-spectrum
    peak_calc = ['norm_corr', 'cumu_corr', 'sq_diff', 'mask_corr']
    phase_based = ['tpss','svd'] 
    peak_based = ['gauss_1', 'parab_1', 'weighted']
    
    if (subpix in phase_based) & (match in peak_calc):
         # some combinations are not possible, reset to default
         warnings.warn('given combination of matching-subpix is not possible, using default')
         match = 'cosicorr'
    if match=='aff_of': # optical flow needs to be of the same size
        search_radius = np.copy(temp_radius)
       
    i1, j1 = map2pix(geoTransform1, Xgrd, Ygrd)
    i2, j2 = map2pix(geoTransform2, Xgrd, Ygrd)
    
    i1, j1 = np.round(i1).astype(np.int64), np.round(j1).astype(np.int64)
    i2, j2 = np.round(i2).astype(np.int64), np.round(j2).astype(np.int64)
    
    # remove posts outside image   
    IN = np.logical_and.reduce((i1>=0, i1<=M1.shape[0], 
                                j1>=0, j1<=M1.shape[1],
                                i2>=0, i2<=M2.shape[0], 
                                j2>=0, j2<=M2.shape[1]))
    
    i1, j1, i2, j2 = i1[IN], j1[IN], i2[IN], j2[IN]
    
    # extend image size, so search regions at the border can be used as well
    I1, M1 = pad_radius(I1, temp_radius), pad_radius(M1, temp_radius)
    i1 += temp_radius
    j1 += temp_radius
    I2, M2 = pad_radius(I2, search_radius), pad_radius(M2, search_radius)
    i2 += search_radius
    j2 += search_radius
    
    ij2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    xy2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    snr_score = np.zeros((i1.shape[0],1)).astype(np.float16)
    
    for counter in range(len(i1)): # loop through all posts
        # create templates
        I1_sub = create_template(I1,i1[counter],j1[counter],temp_radius)
        I2_sub = create_template(I2,i2[counter],j2[counter],search_radius)
        M1_sub = create_template(M1,i1[counter],j1[counter],temp_radius)
        M2_sub = create_template(M2,i2[counter],j2[counter],search_radius)

        Q = masked_cosine_corr(I1_sub, I2_sub, M1_sub, M2_sub)

        # reduce edge effects in frequency space
        if match in ['cosi_corr', 'phas_only', 'symm_phas', 'orie_corr', \
                     'mask_corr', 'bina_phas']:
            (I1_sub,_) = perdecomp(I1_sub)
            (I2_sub,_) = perdecomp(I2_sub)
        
        # translational matching/estimation
        if match in ['norm_corr']:
            C = normalized_cross_corr(I1_sub, I2_sub)    
        elif match in ['cumu_corr']:
            C = cumulative_cross_corr(I1_sub, I2_sub)
        elif match in ['sq_diff']:
            C = sum_sq_diff(I1_sub, I2_sub)
        elif match in ['cosi_corr']:
            (Q, W, m0) = cosi_corr(I1_sub, I2_sub)
        elif match in ['phas_only']:
            Q = phase_only_corr(I1_sub, I2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['symm_phas']:
            Q = symmetric_phase_corr(I1_sub, I2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['ampl_comp']:
            Q = amplitude_comp_corr(I1_sub, I2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['orie_corr']:
            Q = orientation_corr(I1_sub, I2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['mask_corr']:
            C = masked_corr(I1_sub, I2_sub, M1_sub, M2_sub)
        elif match in ['bina_phas']:
            Q = binary_orientation_corr(M1_sub, M2_sub)
            if subpix in peak_based:
                C = np.fft.ifft2(Q)
        elif match in ['aff_of']:
            (di,dj,Aff,snr) = affine_optical_flow(M1_sub, M2_sub)

        if 'C' in locals():
            max_score = np.amax(C)
            snr = max_score/np.mean(C)
            ij = np.unravel_index(np.argmax(C), C.shape)
            di, dj = ij[::-1]
            m0 = np.array([di, dj])
        
        if match!='aff_of':
            # sub-pixel estimation    
            if subpix in ['tpss']:
                if 'W' not in locals():
                    breakpoint()
                (m,snr) = tpss(Q, W, m0)
                ddi, ddj = m[0], m[1]
            elif subpix in ['svd']:
                if 'W' not in locals():
                    breakpoint()
                (ddi,ddj) = hoge(Q, W)
            elif subpix in ['gauss_1']:
                (ddi, ddj) = get_top_gaussian(C, top=m0)
            elif subpix in ['parab_1']:
                (ddi, ddj) = get_top_parabolic(C, top=m0)    
            elif subpix in ['weighted']:
                (ddi, ddj) = get_top_weighted(C, ds=1, top=m0)   
            elif subpix in ['optical_flow']:
                max_score = np.amax(C)
                snr = max_score/np.mean(C)
                ij = np.unravel_index(np.argmax(C), C.shape)
                di, dj = ij[::-1]
                # reposition second template
                I2_sub = create_template(I2,i2[counter]-di,
                                         j1[counter]-dj,temp_radius)
                   
                # optical flow refinement
                ddi,ddj = simple_optical_flow(M1_sub, M2_new)
                
                if (abs(ddi)>0.5) or (abs(ddj)>0.5): # divergence
                    ddi = 0
                    ddj = 0               
            else: # simple highest score extaction
                max_score = np.amax(C)
                snr = max_score/np.mean(C)
                ij = np.unravel_index(np.argmax(C), C.shape)
                di, dj = ij[::-1]
        
        # remove padding if cross correlation is used> is already done, see below
        #if 'C' in locals():    
        #    di -= (search_radius - temp_radius)
        #    dj -= (search_radius - temp_radius)
        
        if abs(ddi)<2:
            di += ddi
        if abs(ddj)<2:
            dj += ddj

        # adjust for affine transform
        if 'Aff' in locals(): # affine transform is also estimated
            Anew = A*Aff
            dj_rig = dj*Anew[0,0] + di*Anew[0,1]
            di_rig = dj*Anew[1,0] + di*Anew[1,1]
        else:
            dj_rig = dj*A[0,0] + di*A[0,1]
            di_rig = dj*A[1,0] + di*A[1,1]                   
            
        snr_score[counter] = snr
            
        ij2_corr[counter,0] = i2[counter] - di_rig
        ij2_corr[counter,1] = j2[counter] - dj_rig
        
        ij2_corr[counter,0] -= search_radius # compensate for padding
        ij2_corr[counter,1] -= search_radius
        
    xy2_corr[:,0], xy2_corr[:,1] = pix2map(geoTransform1, ij2_corr[:,0], ij2_corr[:,1])
    
    return xy2_corr, snr_score    



# establish dH between locations

# 

def angles2unit(azimuth):
    """
    Transforms arguments in degrees to unit vector, in planar coordinates
    
    :param azimuth:     FLOAT 
        argument with clockwise angle direction
    
    :return xy:         FLOAT
        x mapping coordinate, in Eastwards direction
        y mapping coordinate, in Nordwards direction
    """
    x = np.sin(np.radians(azimuth))
    y = np.cos(np.radians(azimuth))
    
    xy = np.stack((x,y)).T
    
    return xy
    
def get_intersection(xy_1, xy_2, xy_3, xy_4):
    """
    given two points per line, estimate the intersection
    """
    
    x_1 = xy_1[:,0]
    y_1 = xy_1[:,1]
    x_2 = xy_2[:,0]
    y_2 = xy_2[:,1]
    x_3 = xy_3[:,0]
    y_3 = xy_3[:,1]
    x_4 = xy_4[:,0]
    y_4 = xy_4[:,1]    
    
    numer_1 = x_1*y_2 - y_1*x_2
    numer_2 = x_3*y_4 - y_3*x_4
    dx_12 = x_1 - x_2
    dx_34 = x_3 - x_4
    dy_12 = y_1 - y_2
    dy_34 = y_3 - y_4
    denom = (dx_12*dy_34 - dy_12*dx_34)
    
    x = ( numer_1 * dx_34 - dx_12 * numer_2) / denom
    y = ( numer_1 * dy_34 - dy_12 * numer_2) / denom 
    
    xy = np.stack((x,y)).T
    return xy

def get_elevation_difference(sun_1, sun_2, xy_1, xy_2, xy_t):
    """
    input:   azimuth_t      array (2 x 1)     argument in degrees of the 
                                              occluder
             zenit_t        array (n x m)     zenith angle in degrees of the 
                                              sun at the location of the 
                                              occluder
             x_c            array (2 x 1)     map coordinates of casted shadow
             y_c            array (2 x 1)     map coordinates of casted shadow
    output:  dh             array (n x m)     elevation difference estimate
             xy_t           array (n x m)     caster location
    """
    # dxy_1 = angles2unit(sun_1[:,0])*1e3
    # dxy_2 = angles2unit(sun_2[:,0])*1e3

    # # get the location of the casting point
    # xy_t = get_intersection(xy_1, xy_1 + dxy_1, 
    #                              xy_2, xy_2 + dxy_2)

      
    # get planar distance between 
    dist_1 = np.sqrt((xy_1[:,0]-xy_t[:,0])**2 + (xy_1[:,1]-xy_t[:,1])**2)
    dist_2 = np.sqrt((xy_2[:,0]-xy_t[:,0])**2 + (xy_2[:,1]-xy_t[:,1])**2)
    
    # get elevation difference 
    dh = np.tan(np.radians(sun_1[:,1]))*dist_1 - \
        np.tan(np.radians(sun_2[:,1]))*dist_2
        
    return dh #, xy_t

# for i in range(len(xy_t[:,1])):   
#     plt.plot([xy_1[i,0], xy_t[i,0]], [xy_1[i,1], xy_t[i,1]], c='red')
# plt.show()
# for i in range(len(xy_t[:,1])):   
#     plt.plot([xy_2[i,0], xy_t[i,0]], [xy_2[i,1], xy_t[i,1]], c='blue')
# plt.show()


# def transform_to_elevation_difference(sun_1, sun_2, xy_1, xy_2):
#     """    
#     input:   azimuth_t      array (2 x 1)     argument in degrees of the 
#                                               occluder
#              zenit_t        array (n x m)     zenith angle in degrees of the 
#                                               sun at the location of the 
#                                               occluder
#              x_c            array (2 x 1)     map coordinates of casted shadow
#              y_c            array (2 x 1)     map coordinates of casted shadow
#     output:  dh             array (n x m)     elevation difference
#     """    
    
#     # translate coordinate system to the middle
#     xy_mean = np.stack((xy_1[:,0]/2+xy_2[:,0]/2,xy_1[:,1]/2+xy_2[:,1]/2)).T
#     xy_1_tilde = xy_1 - xy_mean
#     xy_2_tilde = xy_2 - xy_mean
#     del xy_mean
    
#     # rotate coordinate system towards mean argument
#     azi_mean = np.arctan2(0.5* (np.sin(np.deg2rad(sun_1[:,0])) + 
#                                 np.sin(np.deg2rad(sun_2[:,0]))) , 
#                           0.5* (np.cos(np.deg2rad(sun_2[:,0])) + 
#                                 np.cos(np.deg2rad(sun_2[:,0])) )
#                           ) # in radians
#     xy_1_tilde = np.stack((+ np.cos(azi_mean) * xy_1_tilde[:,0]
#                            - np.sin(azi_mean) * xy_1_tilde[:,1], 
#                            + np.sin(azi_mean) * xy_1_tilde[:,0]
#                            + np.cos(azi_mean) * xy_1_tilde[:,1])
#                           ).T
#     xy_2_tilde = np.stack((+ np.cos(azi_mean) * xy_2_tilde[:,0]
#                            - np.sin(azi_mean) * xy_2_tilde[:,1], 
#                            + np.sin(azi_mean) * xy_2_tilde[:,0]
#                            + np.cos(azi_mean) * xy_2_tilde[:,1])
#                           ).T
#     del axi_mean
    
#     # extract length
#     dxy_tilde = xy_1_tilde - xy_2_tilde
#     dzenit = sun_1[:,1] - sun_2[:,1]

#     # translate to elevation difference
    
    
    
#     return dh
    
    