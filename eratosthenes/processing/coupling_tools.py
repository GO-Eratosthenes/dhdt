import math

import numpy as np

from sklearn.neighbors import NearestNeighbors

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
    