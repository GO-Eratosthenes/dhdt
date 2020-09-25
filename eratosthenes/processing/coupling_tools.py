import math

import numpy as np

from sklearn.neighbors import NearestNeighbors

from eratosthenes.generic.mapping_tools import map2pix, pix2map 
from eratosthenes.processing.matching_tools import normalized_cross_corr

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



# create file with approx shadow cast locations

# refine by matching, and include correlation score

def match_shadow_casts(M1, M2, geoTransform1, geoTransform2,
                       post1, post2, temp_radius=7, search_radius=22):
    """
    
    """
    i1, j1 = map2pix(geoTransform1, post1[:,0], post1[:,1])
    i2, j2 = map2pix(geoTransform2, post2[:,0], post2[:,1])
    
    i1 = np.round(i1).astype(np.int64)
    j1 = np.round(j1).astype(np.int64)
    i2 = np.round(i2).astype(np.int64)
    j2 = np.round(j2).astype(np.int64)
    
    ij2_corr = np.zeros(i1.shape[0],2).astype(np.float64)
    post2_corr = np.zeros(i1.shape[0],2).astype(np.float64)
    corr_score = np.zeros(i1.shape[0],1).astype(np.float16)
    for counter in range(len(i1)):
        M1_sub = M1[i1[counter] - temp_radius:i1[counter] + temp_radius + 1,
                j1[counter] - temp_radius:j1[counter] + temp_radius + 1]
        M2_sub = M2[i2[counter] - search_radius:i2[counter] + search_radius + 1,
                j2[counter] - search_radius:j2[counter] + search_radius + 1]
        # match
        di,dj,corr = normalized_cross_corr(M1_sub, M2_sub)
        corr_score[counter] = corr
        ij2_corr[counter,0] = di-search_radius
        ij2_corr[counter,1] = dj-search_radius
    post2_corr[:,0], post2_corr[:,1] = pix2map(geoTransform1, post1[:,0], post1[:,1])
    
    return post2_corr, corr_score


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
    x = math.sin(math.radians(azimuth))
    y = math.cos(math.radians(azimuth))
    return x, y
    
def get_intersection(azimuth_t, x_c, y_c):
    """
    given the illumination directions, and cast locations.
    estimate the occluder
    """
    
    A = math.sin(math.radians(azimuth_t))
    y = x_c
    
    x_t = np.linalg.lstsq(A, y, rcond=None)
    
    A = math.cos(math.radians(azimuth_t))
    y = y_c
    y_t = np.linalg.lstsq(A, y, rcond=None)
         
    return x_t, y_t

def get_elevation_difference(azimuth_t, zenit_t, x_c, y_c):
    """
    input:   azimuth_t      array (2 x 1)     argument in degrees of the 
                                              occluder
             zenit_t        array (n x m)     zenith angle in degrees of the 
                                              sun at the location of the 
                                              occluder
             x_c            array (2 x 1)     map coordinates of casted shadow
             y_c            array (2 x 1)     map coordinates of casted shadow
    output:  u              array (n x m)     displacement estimate
             v              array (n x m)     displacement estimate
    """
    
    # get the location of the casting point
    (x_t,y_t) = get_intersection(azimuth_t, x_c, y_c)
       
    # get planar distance between 
    dist = np.sqrt((x_c-x_t)**2 + (y_c-y_t)**2)
    
    # get elevation difference 
    dh = math.tann(math.radians(zenit_t[0]))*dist[0] - \
        math.tann(math.radians(zenit_t[1]))*dist[1]
        
    return dh, x_t, y_t
    