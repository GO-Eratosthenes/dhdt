import os

import numpy as np

# geospatial libaries
from osgeo import ogr
from rasterio import Affine

# raster/image libraries
from scipy import ndimage

from .handler_im import get_grad_filters

def cart2pol(x, y):
    if isinstance(x, np.ndarray):
        assert x.shape==y.shape, ('arrays should be of the same size')    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    if isinstance(rho, np.ndarray):
        assert rho.shape==phi.shape, ('arrays should be of the same size')

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def pix2map(geoTransform, i, j):
    """ transform local image coordinates to map coordinates

    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    i : np.array, ndim={1,2,3}, dtype=float
        row coordinate(s) in local image space
    j : np.array, ndim={1,2,3}, dtype=float
        column coordinate(s) in local image space

    Returns
    -------
    x : np.array, size=(m), ndim={1,2,3}, dtype=float
        horizontal map coordinate.
    y : np.array, size=(m), ndim={1,2,3}, dtype=float
        vertical map coordinate.
        
    See Also
    --------
    map2pix        

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
    if isinstance(i, np.ndarray):
        assert i.shape==j.shape, ('arrays should be of the same size')
    assert isinstance(geoTransform, tuple), ('geoTransform should be a tuple')

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

def map2pix(geoTransform, x, y):
    """ transform map coordinates to local image coordinates

    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    x : np.array, size=(m), ndim={1,2,3}, dtype=float
        horizontal map coordinate.
    y : np.array, size=(m), ndim={1,2,3}, dtype=float
        vertical map coordinate.

    Returns
    -------
    i : np.array, ndim={1,2,3}, dtype=float
        row coordinate(s) in local image space
    j : np.array, ndim={1,2,3}, dtype=float
        column coordinate(s) in local image space
        
    See Also
    --------
    pix2map        
    
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
    if isinstance(x, np.ndarray):
        assert x.shape==y.shape, ('arrays should be of the same size')
    assert isinstance(geoTransform, tuple), ('geoTransform should be a tuple')
    
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

def rot_mat(theta):
    """ build a 2x2 rotation matrix
    
    Parameters
    ----------
    theta : float
        angle in degrees

    Returns
    -------
    R : np.array, size=(2,2)
        2D rotation matrix
        
    Notes
    -----
    Matrix is composed through,
    
        .. math:: \mathbf{R}[1,:] = [+\cos(\omega), -\sin(\omega)]
        .. math:: \mathbf{R}[2,:] = [+\sin(\omega), +\cos(\omega)]

    The angle(s) are declared in the following coordinate frame: 
        
        .. code-block:: text
        
                 ^ North & y
                 |  
            - <--|--> +
                 |
                 +----> East & x
                 
    """       
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return R

def cast_orientation(I, Az, indexing='ij'):
    """ emphasises intentisies within a certain direction

    Parameters
    ----------
    I : np.array, size=(m,n)
        array with intensity values
    Az : {float,np.array}, size{(m,n),1}, unit=degrees
        band of azimuth values or single azimuth angle.
    indexing : {‘xy’, ‘ij’}  
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates
         
    Returns
    -------
    Ican : np.array, size=(m,n)
        array with intensity variation in the preferred direction.

    See Also
    --------
    .handler_im.get_grad_filters : collection of general gradient filters
    .handler_im.rotated_sobel : generate rotated gradient filter

    Notes
    ----- 
        The angle(s) are declared in the following coordinate frame: 
        
        .. code-block:: text
        
                 ^ North & y
                 |  
            - <--|--> +
                 |
                 +----> East & x

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

    References
    ----------    
    .. [1] Freeman et al. "The design and use of steerable filters." IEEE 
       transactions on pattern analysis and machine intelligence vol.13(9) 
       pp.891-906, 1991.
    """
    assert type(I)==np.ndarray, ("please provide an array")
    
    fx,fy = get_grad_filters(ftype='sobel', tsize=3, order=1)

    Idx = ndimage.convolve(I, fx)  # steerable filters
    Idy = ndimage.convolve(I, fy)
    
    if indexing=='ij':    
        Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
                - np.multiply(np.sin(np.radians(Az)), Idx))
    else:
        Ican = (np.multiply(np.cos(np.radians(Az)), Idy)
                    + np.multiply(np.sin(np.radians(Az)), Idx))
    return Ican

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

def ref_trans(Transform, dI, dJ):
    """ translate reference transform

    Parameters
    ----------
    Transform : tuple, size=(1,6)
        georeference transform of an image.
    dI : dtype={float,integer}
        translation in rows.
    dJ : dtype={float,integer}
        translation in collumns.

    Returns
    -------
    newTransform : tuple, size=(1,6)
        georeference transform of an image with shifted origin.

    See Also
    --------
    ref_scale 

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
    newTransform = (Transform[0]+ dJ*Transform[1] + dI*Transform[2],
                    Transform[1], Transform[2],
                    Transform[3]+ dJ*Transform[4] + dI*Transform[5],
                    Transform[4], Transform[5])
    return newTransform

def ref_scale(geoTransform, scaling):
    """ scale image reference transform
    
    Parameters
    ----------
    Transform : tuple, size=(1,6)
        georeference transform of an image.
    scaling : float
        isotropic scaling in both rows and collumns.

    Returns
    -------
    newTransform : tuple, size=(1,6)
        georeference transform of an image with different pixel spacing.

    See Also
    --------
    ref_trans 
    """
    # not using center of pixel
    R = np.asarray(geoTransform).reshape((2,3)).T
    R[0,:] += np.diag(R[1:,:]*scaling/2)
    R[1:,:] = R[1:,:]*scaling/2
    newTransform = tuple(map(tuple, np.transpose(R).reshape((1,6))))
    
    return newTransform[0]

def pix_centers(geoTransform, rows, cols, make_grid=True):
    """ provide the pixel coordinate from the axis, or the whole grid
    
    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    rows : integer
        amount of rows in an image.
    cols : integer
        amount of collumns in an image.
    make_grid : bool, optional
        Should a grid be made. The default is True.

    Returns
    -------
    X : np.array, dtype=float    
         * "make_grid" : size=(m,n)
         * otherwise : size=(1,n)
    Y : np.array, dtype=float    
         * "make_grid" : size=(m,n)
         * otherwise : size=(m,1)

    See Also
    --------
    map2pix, pix2map

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

def bilinear_interp_excluding_nodat(I, i_I, j_I, noData=-9999):
    """ do simple bi-linear interpolation, taking care of nodata

    Parameters
    ----------
    I : np.array, size=(m,n), ndim={2,3}
        data array.
    i_I : np.array, size=(k,l), ndim=2
        horizontal locations, within local image frame.
    j_I : np.array, size=(k,l), ndim=2
        vertical locations, within local image frame.
    noData : TYPE, optional
        DESCRIPTION. The default is -9999.

    Returns
    -------
    I_new : np.array, size=(k,l), ndim=2, dtype={float,complex}
        interpolated values.

    See Also
    --------
    .handler_im.bilinear_interpolation

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
    assert type(I)==np.ndarray, ("please provide an array")
    
    i_prio, i_post = np.floor(i_I).astype(int), np.ceil(i_I).astype(int)
    j_prio,j_post = np.floor(j_I).astype(int), np.ceil(j_I).astype(int)
      
    wi,wj = np.remainder(i_I,1), np.remainder(j_I,1)
    
    I_1,I_2 = I[i_prio,j_prio], I[i_prio,j_post]  
    I_3,I_4 = I[i_post,j_post], I[i_post,j_prio]  
    
    no_dat = np.any(np.vstack((I_1,I_2,I_3,I_4))==noData, axis=0)
    
    DEM_new = wj*(wi*I_1 + (1-wi)*I_4) + (1-wj)*(wi*I_2 + (1-wi)*I_3)
    DEM_new[no_dat] = noData
    return DEM_new

def bbox_boolean(img):
    """ get image coordinates of maximum bounding box extent of True values
    in a boolean matrix

    Parameters
    ----------
    img : np.array, size=(m,n), dtype=bool
        data array.

    Returns
    -------
    r_min : integer
        minimum row with a true boolean.
    r_max : integer
        maximum row with a true boolean.
    c_min : integer
        minimum collumn with a true boolean.
    c_max : integer
        maximum collumn with a true boolean.
    """
    assert type(img)==np.ndarray, ("please provide an array")
    
    rows,cols = np.any(img, axis=1), np.any(img, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max, c_min, c_max

def get_bbox(geoTransform, rows, cols):
    """ given array meta data, calculate the bounding box

    Parameters
    ----------
    geoTransform : tuple, size=(1,6)
        georeference transform of an image.
    rows : integer
        amount of rows in an image.
    cols : integer
        amount of collumns in an image.

    Returns
    -------
    bbox : np.array, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y
        
    See Also
    --------
    map2pix, pix2map, pix_centers, get_map_extent

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
    X = geoTransform[0] + \
        np.array([0, cols])*geoTransform[1] + np.array([0, rows])*geoTransform[2] 
        
    Y = geoTransform[3] + \
        np.array([0, cols])*geoTransform[4] + np.array([0, rows])*geoTransform[5]
#    spacing_x = np.sqrt(geoTransform[1]**2 + geoTransform[2]**2)/2
#    spacing_y = np.sqrt(geoTransform[4]**2 + geoTransform[5]**2)/2
    bbox = np.hstack((np.sort(X), np.sort(Y)))
#    # get extent not pixel centers
#    bbox += np.array([-spacing_x, +spacing_x, -spacing_y, +spacing_y])
    return bbox
    
def get_map_extent(bbox):
    """ generate coordinate list in counterclockwise direction from boundingbox
    input:   bbox           array (1 x 4)     min max X, min max Y  
    output:  xB             array (5 x 1)     coordinate list for x  
             yB             array (5 x 1)     coordinate list for y   

    Parameters
    ----------
    bbox : np.array, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y

    Returns
    -------
    xB : TYPE
        coordinate list of horizontal outer points
    yB : TYPE
        coordinate list of vertical outer points

    See Also
    --------
    get_bbox, get_bbox_polygon

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
    xB = np.array([[ bbox[0], bbox[0], bbox[1], bbox[1], bbox[0] ]]).T
    yB = np.array([[ bbox[3], bbox[2], bbox[2], bbox[3], bbox[3] ]]).T
    return xB, yB    
    
def get_bbox_polygon(geoTransform, rows, cols):
    """ given array meta data, create bounding polygon
    
    Parameters
    ----------
    geoTransform : tuple, size=(6,1)
        georeference transform of an image.
    rows : integer
        amount of rows in an image.
    cols : integer
        amount of collumns in an image.

    Returns
    -------
    poly : OGR polygon
        bounding polygon of image.

    See Also
    --------
    get_bbox, get_map_extent
    """
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
    """ clip array to the same size as another array

    Parameters
    ----------
    Old : np.array, size=(m,n), dtype={float,complex}
        data array to be clipped.
    geoTransform_new : tuple, size=(6,1)
        georeference transform of the old image.
    geoTransform_new : tuple, size=(6,1)
        georeference transform of the new image.
    rows_new : integer
        amount of rows of the new image.
    cols_new : integer
        amount of collumns of the new image.

    Returns
    -------
    New : np.array, size=(k,l), dtype={float,complex}
        clipped data array.
    """
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
    di,dj = rows_new - Old.shape[0], cols_new - Old.shape[1]
    
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
