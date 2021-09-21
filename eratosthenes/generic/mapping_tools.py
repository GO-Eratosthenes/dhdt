import os

import numpy as np

# geospatial libaries
from osgeo import ogr
from rasterio import Affine

# raster/image libraries
from scipy import ndimage


def castOrientation(I, Az):  # generic
    """
    Emphasises intentisies within a certain direction
    input:   I              array (n x m)     band with intensity values
             Az             array (n x m)     band of azimuth values
    output:  Ican           array (m x m)     Shadow band
    """
#    kernel = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]) # Sobel
    kernel = np.array([[17], [61], [17]])*np.array([-1, 0, 1])/95  # Kroon
    Idx = ndimage.convolve(I, np.flip(kernel, axis=1))  # steerable filters
    Idy = ndimage.convolve(I, np.flip(np.transpose(kernel), axis=0))
    Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
            - np.multiply(np.sin(np.radians(Az)), Idx))
    return Ican

def bilinear_interp_excluding_nodat(DEM, i_DEM, j_DEM, noData=-9999):
    '''
    do simple bi-linear interpolation
    '''
    # do bilinear interpolation
    i_prio = np.floor(i_DEM).astype(int)
    i_post = np.ceil(i_DEM).astype(int)
    j_prio = np.floor(j_DEM).astype(int)
    j_post = np.ceil(j_DEM).astype(int)
      
    wi = np.remainder(i_DEM,1)
    wj = np.remainder(j_DEM,1)
    
    DEM_1 = DEM[i_prio,j_prio]  
    DEM_2 = DEM[i_prio,j_post]  
    DEM_3 = DEM[i_post,j_post]  
    DEM_4 = DEM[i_post,j_prio]  
    
    no_dat = np.any(np.vstack((DEM_1,DEM_2,DEM_3,DEM_4))==noData, axis=0)
    # look at nan
    
    DEM_ij = wj*(wi*DEM_1 + (1-wi)*DEM_4) + (1-wj)*(wi*DEM_2 + (1-wi)*DEM_3)
    DEM_ij[no_dat] = noData
    return DEM_ij

def bboxBoolean(img):  # generic
    """
    get image coordinates of maximum bounding box extent of True values in a
    boolean matrix
    input:   img            array (n x m)     band with boolean values
    output:  rmin           integer           minimum row
             rmax           integer           maximum row
             cmin           integer           minimum collumn
             cmax           integer           maximum collumn
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def ref_trans(Transform, dI, dJ):  # generic
    """
    translate reference transform
    input:   Transform      array (1 x 6)     georeference transform of
                                              an image
             dI             integer           translation in rows
             dJ             integer           translation in collumns
    output:  newransform    array (1 x 6)     georeference transform of
                                              transformed image
    """
    newTransform = (Transform[0]+ dJ*Transform[1] + dI*Transform[2],
                    Transform[1], Transform[2],
                    Transform[3]+ dJ*Transform[4] + dI*Transform[5],
                    Transform[4], Transform[5])
    return newTransform

def RefScale(Transform, scaling):  # generic
    """
    scale reference transform
    input:   Transform      array (1 x 6)     georeference transform of
                                              an image
             scaling        integer           scaling in rows and collumns
    output:  newransform    array (1 x 6)     georeference transform of
                                              transformed image
    """
    # not using center of pixel
    newTransform = (Transform[0], Transform[1]*scaling, Transform[2]*scaling,
                    Transform[3], Transform[4]*scaling, Transform[5]*scaling)
    return newTransform

def rotMat(theta):  # generic
    """
    build rotation matrix, theta is in degrees
    input:   theta        integer             angle
    output:  R            array (2 x 2)     2D rotation matrix
    """
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return R

def pix2map(geoTransform, i, j):  # generic
    """
    Transform image coordinates to map coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    output:  x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    """
    x = (geoTransform[0]
         + geoTransform[1] * j
         + geoTransform[2] * i
         )
    y = (geoTransform[3]
         + geoTransform[4] * j
         + geoTransform[5] * i
         )

    # # offset the center of the pixel
    # x += geoTransform[1] / 2.0
    # y += geoTransform[5] / 2.0
    return x, y

def pix_centers(geoTransform, rows, cols, make_grid=True):
    '''
    Provide the pixel coordinate from the axis, or the whole grid
    '''
    i = np.linspace(1, rows, rows)
    j = np.linspace(1, cols, cols)
    if make_grid:
        J, I = np.meshgrid(j, i)
        X,Y = pix2map(geoTransform, I, J)
        return X, Y
    else:
        x,y_dummy = pix2map(geoTransform, np.repeat(i[0], len(j)), j)
        x_dummy,y = pix2map(geoTransform, i, np.repeat(j[0], len(i)))
        return x, y

def map2pix(geoTransform, x, y):  # generic
    """
    Transform map coordinates to image coordinates
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             x              array (n x 1)     map coordinates
             y              array (n x 1)     map coordinates
    output:  i              array (n x 1)     row coordinates in image space
             j              array (n x 1)     column coordinates in image space
    """
    # # offset the center of the pixel
    # x -= geoTransform[1] / 2.0
    # y -= geoTransform[5] / 2.0
    # ^- this messes-up python with its pointers....
    j = x - geoTransform[0]
    i = y - geoTransform[3]

    if geoTransform[2] == 0:
        j = j / geoTransform[1]
    else:
        j = (j / geoTransform[1]
             + i / geoTransform[2])

    if geoTransform[4] == 0:
        i = i / geoTransform[5]
    else:
        i = (j / geoTransform[4]
             + i / geoTransform[5])

    return i, j

def resize_geotransform(geoTransform, scaling):
    '''
    if an zonal operator is apply to an image, the resulting geoTransform needs
    to change accordingly
    '''
    R = np.asarray(geoTransform).reshape((2,3)).T
    R[0,:] += np.diag(R[1:,:]*scaling/2)
    R[1:,:] = R[1:,:]*scaling/2
    new_geotransform = tuple(map(tuple, np.transpose(R).reshape((1,6))))
    
    return new_geotransform[0]

def get_bbox(geoTransform, rows, cols):
    '''
    given array meta data, calculate the bounding box
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             rows           integer           row size of the image
             cols           integer           collumn size of the image
    output:  bbox           array (1 x 4)     min max X, min max Y   
    '''
    X = geoTransform[0] + \
        np.array([0, cols])*geoTransform[1] + np.array([0, rows])*geoTransform[2] 
        
    Y = geoTransform[3] + \
        np.array([0, cols])*geoTransform[4] + np.array([0, rows])*geoTransform[5]
    spacing_x = np.sqrt(geoTransform[1]**2 + geoTransform[2]**2)/2
    spacing_y = np.sqrt(geoTransform[4]**2 + geoTransform[5]**2)/2
    bbox = np.hstack((np.sort(X), np.sort(Y)))
#    # get extent not pixel centers
#    bbox += np.array([-spacing_x, +spacing_x, -spacing_y, +spacing_y])
    return bbox

def GDAL_transform_to_affine(GDALtransform):
    '''
    the GDAL-style geotransform is like:
        (c, a, b, f, d, e)
    but should be an Affine structre to work wit rasterio:
        affine.Affine(a, b, c,
                      d, e, f)
    '''
    new_transform = Affine(GDALtransform[1], GDALtransform[2], GDALtransform[0], \
                           GDALtransform[4], GDALtransform[5], GDALtransform[3])
    return new_transform
    
def get_map_extent(bbox):
    """
    generate coordinate list in counterclockwise direction from boundingbox
    input:   bbox           array (1 x 4)     min max X, min max Y  
    output:  xB             array (5 x 1)     coordinate list for x  
             yB             array (5 x 1)     coordinate list for y      
    """
    xB = np.array([[ bbox[0], bbox[0], bbox[1], bbox[1], bbox[0] ]]).T
    yB = np.array([[ bbox[3], bbox[2], bbox[2], bbox[3], bbox[3] ]]).T
    return xB, yB    
    
def get_bbox_polygon(geoTransform, rows, cols):
    '''
    given array meta data, calculate the bounding box
    input:   geoTransform   array (1 x 6)     georeference transform of
                                              an image
             rows           integer           row size of the image
             cols           integer           collumn size of the image
    output:  bbox           OGR polygon          
    '''    
    # build tile polygon
    bbox = get_bbox(geoTransform, rows, cols)
    xB,yB = get_map_extent(bbox)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in range(5):
        ring.AddPoint(float(xB[i]),float(yB[i]))
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    # poly_tile.ExportToWkt()
    return poly
    
def find_overlapping_DEM_tiles(dem_path,dem_file, poly_tile):
    '''
    loop through a shapefile of tiles, to find overlap with a given geometry
    '''
    # Get the DEM tile layer
    demShp = ogr.Open(os.path.join(dem_path, dem_file))
    demLayer = demShp.GetLayer()

    url_list = ()
    # loop through the tiles and see if there is an intersection
    for i in range(0, demLayer.GetFeatureCount()):
        # Get the input Feature
        demFeature = demLayer.GetFeature(i)
        geom = demFeature.GetGeometryRef()
        
        intersection = poly_tile.Intersection(geom)
        if(intersection is not None and intersection.Area()>0):
            url_list += (demFeature.GetField('fileurl'),)
    
    return url_list

def make_same_size(Old,geoTransform_old, geoTransform_new, rows_new, cols_new):
    
    # look at upper left coordinate
    dj = np.round((geoTransform_new[0]-geoTransform_old[0])/geoTransform_new[1])
    di = np.round((geoTransform_new[3]-geoTransform_old[3])/geoTransform_new[1])
    
    if np.sign(dj)==-1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(np.expand_dims(Old[:,0], axis = 1), 
                                        abs(dj), axis=1), Old), axis = 1)
    elif np.sign(dj)==1: # reduce array
        Old = Old[:,abs(dj):]
    
    if np.sign(di)==-1: # reduce array
        Old = Old[abs(di):,:]
    elif np.sign(di)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(np.expand_dims(Old[0,:], axis = 1).T, 
                                        abs(di), axis=0), Old), axis = 0)        
    
    # as they are now alligned, look at the lower right corner
    di = rows_new - Old.shape[0]
    dj = cols_new - Old.shape[1]
    
    
    if np.sign(dj)==-1: # reduce array
        Old = Old[:,:dj]
    elif np.sign(dj)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(Old, np.expand_dims(Old[:,-1], axis=1), 
                                        abs(dj), axis=1)), axis = 1)
    
    if np.sign(di)==-1: # reduce array
        Old = Old[di:,:]
    elif np.sign(di)==1: # extend array by simple copy of border values
        Old = np.concatenate((np.repeat(Old, np.expand_dims(Old[-1,:], axis=1).T, 
                                        abs(di), axis=0)), axis = 0) 
    
    New = Old
    return New
    
    
    