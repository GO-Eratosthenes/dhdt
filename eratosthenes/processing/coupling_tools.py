import os
import glob
import math

import numpy as np

from sklearn.neighbors import NearestNeighbors
from skimage import measure

from eratosthenes.generic.mapping_tools import map2pix, pix2map, rotMat, castOrientation, ref_trans
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.generic.handler_im import bilinear_interpolation

from eratosthenes.preprocessing.read_s2 import read_sun_angles_s2

from eratosthenes.processing.matching_tools import normalized_cross_corr, cumulative_cross_corr, lucas_kanade_single, beam_angle_statistics, cast_angle_neighbours, neighbouring_cast_distances, perdecomp, cosi_corr, tpss

def couple_pair(dat_path, fname1, fname2, bbox, co_name, co_reg, rgi_id=None, 
                reg='binary', prepro='bandpass', \
                match='cosicorr', boi=np.array([4]), \
                processing='stacking',\
                subpix='tpss'):
    # get start and finish points of shadow edges
    if rgi_id == None:
        conn1 = np.loadtxt(fname = dat_path+fname1+'conn.txt')
        conn2 = np.loadtxt(fname = dat_path+fname2+'conn.txt')
    else:
        conn1 = np.loadtxt(fname = dat_path + fname1 + 'conn-rgi' + 
                           '{:08d}'.format(rgi_id[0]) + '.txt')
        conn2 = np.loadtxt(fname = dat_path + fname2 + 'conn-rgi' + 
                           '{:08d}'.format(rgi_id[0]) + '.txt')        
    
    # compensate for coregistration
    coid1 = co_name.index(fname1)
    coid2 = co_name.index(fname2)
    coDxy = co_reg[coid1]-co_reg[coid2]
    
    conn2[:,0] += coDxy[0]*10 # transform to meters?
    conn2[:,1] += coDxy[1]*10

    idxConn = pair_images(conn1, conn2)
    # connected list
    casters = conn1[idxConn[:,1],0:2]
    post1 = conn1[idxConn[:,1],2:4].copy() # cast location in xy-coordinates for t1
    post2 = conn2[idxConn[:,0],2:4].copy() # cast location in xy-coordinates for t2
    
    
    # refine through feature descriptors 
    if match in ['feature']: # using the shadow polygons
        (L1, crs, geoTransform1, targetprj) = read_geo_image(
                dat_path + fname1 + 'labelPolygons.tif')
        (L2, crs, geoTransform2, targetprj) = read_geo_image(
                dat_path + fname2 + 'labelPolygons.tif')
        
        # use polygon descriptors
        #describ='NCD' # BAS
        if subpix in ['CAN', 'NCD']: # sun angle needed?
            (sun1_Zn,sun1_Az) = read_sun_angles_s2(os.path.join(dat_path, fname1))
            (sun2_Zn,sun2_Az) = read_sun_angles_s2(os.path.join(dat_path, fname2))
            
        else:
            sun1_Az, sun2_Az = [], []
        post2_new, euc_score = match_shadow_ridge(L1, L2, sun1_Az, sun2_Az,
                                              geoTransform1, geoTransform2, 
                                              post1, post2, subpix, K=10)                
    else: 
        # refine through pattern matching
        
        if reg in ['binary']: # get shadow classification if needed        
            (L1, crs, geoTransform1, targetprj) = read_geo_image(
                    dat_path + fname1 + 'labelPolygons.tif')
            (L2, crs, geoTransform2, targetprj) = read_geo_image(
                    dat_path + fname2 + 'labelPolygons.tif')
        else:
            L1, L2 = [], []
        
        if processing in ['shadow']:
            (M1, crs, geoTransform1, targetprj) = read_geo_image(
                    dat_path + fname1 + 'shadows.tif') # shadow transform template
            (M2, crs, geoTransform2, targetprj) = read_geo_image(
                    dat_path + fname2 + 'shadows.tif') # shadow transform search space
        elif processing in ['stacking']: # multi-spectral stacking
            M1 = np.zeros((bbox[1]-bbox[0], bbox[3]-bbox[2], \
                                 len(boi))) # create empty stack
            im1_dir = dat_path + fname1
            for i in range(len(boi)):
                # find image
                im_file = glob.glob(im1_dir+'*'+'{:02d}'.format(boi[i])+'.jp2')
                if len(im_file)==0:
                    im_file = glob.glob(im1_dir+'*'+'{:02d}'.format(boi[i])+'.tif')
                if len(im_file)!=0:
                    (im1_bnd,crs, geoTransform1, targetprj) = read_geo_image(
                        im_file[0])
                M1[:,:,i] = im1_bnd[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            geoTransform1 = ref_trans(geoTransform1,bbox[0],bbox[2])    
            del im1_dir, im1_bnd, im_file

            M2 = np.zeros((bbox[1]-bbox[0], bbox[3]-bbox[2], \
                                 len(boi))) # create empty stack
            im2_dir = dat_path + fname2
            for i in range(len(boi)):
                # find image
                im_file = glob.glob(im2_dir+'*'+'{:02d}'.format(boi[i])+'.jp2')
                if len(im_file)==0:
                    im_file = glob.glob(im2_dir+'*'+'{:02d}'.format(boi[i])+'.tif')
                if len(im_file)!=0:
                    (im2_bnd,_,geoTransform2,_) = read_geo_image(
                        im_file[0])
                M2[:,:,i] = im2_bnd[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            geoTransform2 = ref_trans(geoTransform2,bbox[0],bbox[2])
            del im2_dir, im2_bnd, im_file
            
        else:
            breakpoint()
        
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
        
        if not(reg in ['binary']):
            # estimate relative scale of templates
            # get planar distance between 
            dist_1 = np.sqrt((post1[:,0]-casters[:,0])**2 + \
                             (post1[:,1]-casters[:,1])**2)
            dist_2 = np.sqrt((post2[:,0]-casters[:,0])**2 + \
                             (post2[:,1]-casters[:,1])**2)
            scale_12 = np.divide(dist_2, dist_1)
            
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
    return post1, post2_corr, casters, dh
    
def pair_images(conn1, conn2, thres=20):
    """
    Find the closest correspnding points in two lists, 
    within a certain bound (given by thres)
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(conn1[:,0:2])
    distances, indices = nbrs.kneighbors(conn2[:,0:2])
    IN = distances<thres
    idxConn = np.transpose(np.vstack((np.where(IN)[0], indices[IN])))
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
    
    """
    
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
    if M1.ndim==3:
        M1 = np.pad(M1, \
                    ((temp_radius,temp_radius), \
                     (temp_radius,temp_radius), \
                     (0,0)), \
                    'constant', constant_values=0)
    else:
        M1 = np.pad(M1, temp_radius,'constant', constant_values=0)
    i1 += temp_radius
    j1 += temp_radius
    if M2.ndim ==3:
        M1 = np.pad(M1, \
                    ((search_radius,search_radius), \
                     (search_radius,search_radius), \
                         (0,0)), \
                    'constant', constant_values=0)
    else:
        M2 = np.pad(M2, search_radius,'constant', constant_values=0)
    i2 += search_radius
    j2 += search_radius
    
    ij2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    xy2_corr = np.zeros((i1.shape[0],2)).astype(np.float64)
    snr_score = np.zeros((i1.shape[0],1)).astype(np.float16)
    
    for counter in range(len(i1)): # loop through all posts
       
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
        if M1.ndim==3:
            M1_sub = M1[i1[counter]-temp_radius : i1[counter]+temp_radius+1,
                    j1[counter]-temp_radius : j1[counter]+temp_radius+1, :]
        else:                        
            M1_sub = M1[i1[counter]-temp_radius : i1[counter]+temp_radius+1,
                    j1[counter]-temp_radius : j1[counter]+temp_radius+1]
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
            (Q, WS, m0) = cosi_corr(M1_sub, M2_sub)
        elif match in ['phas_only']:
            Q = phase_only_corr(M1_sub, M2_sub)
        elif match in ['symm_phas']:
            Q = symmetric_phase_corr(M1_sub, M2_sub)
        elif match in ['ampl_comp']:
            Q = amplitude_comp_corr(M1_sub, M2_sub)
        elif match in ['orie_corr']:
            Q = orientation_corr(M1_sub, M2_sub)
        elif match in ['mask_corr']:
            C = masked_corr(M1_sub, M2_sub, L1_sub, L2_sub)
        elif match in ['bina_phas']:
            Q = binary_orientation_corr(M1_sub, M2_sub)
        
        # sub-pixel estimation    
        if subpix in ['tpss']:
            (m, snr) = tpss(Q, WS, m0)
            di, dj = np.round(m0) + m
        elif subpix in ['gauss_1']:
            (m,snr) = get_top_gaussian(C)
            di, dj = m[0], m[1] 
        elif subpix in ['parab_1']:
            (m,snr) = get_top_parabolic(C)
            di, dj = m[0], m[1]     
        elif subpix in ['weighted']:
            (m,snr) = get_top_weighted(C, ds=1)
            di, dj = m[0], m[1]     
        elif subpix in ['optical_flow']:
            max_score = np.amax(C)
            snr = max_score/np.mean(C)
            ij = np.unravel_index(np.argmax(C), C.shape)
            di, dj = ij[::-1]
                       
            if M2.ndim==3: # construct new image
                M2_new = M2[i2[counter]-di - temp_radius:
                            i2[counter]-di + temp_radius + 1,
                            j2[counter]-dj - temp_radius:
                            j2[counter]-dj + temp_radius + 1, :]
            else:
                M2_new = M2[i2[counter]-di - temp_radius:
                            i2[counter]-di + temp_radius + 1,
                            j2[counter]-dj - temp_radius:
                            j2[counter]-dj + temp_radius + 1]
            
            # optical flow refinement
            ddi,ddj = lucas_kanade_single(M1_sub, M2_new)
            
            if (abs(ddi)>0.5) or (abs(ddj)>0.5): # divergence
                ddi = 0
                ddj = 0
            
        else: # simple highest score extaction
            max_score = np.amax(C)
            snr = max_score/np.mean(C)
            ij = np.unravel_index(np.argmax(C), C.shape)
            di, dj = ij[::-1]
        
        # remove padding if cross correlation is used
        
        # is correct?
        if 'C' in locals():    
            di -= (search_radius - temp_radius)
            dj -= (search_radius - temp_radius)
            
        di += ddi
        dj += ddj
        
        # adjust for affine transform
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
    input:
    azimuth argument with clockwise angle direction
    output:
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
    
    