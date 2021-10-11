# general libraries
import warnings
import numpy as np

from ..generic.mapping_tools import map2pix

def get_integer_peak_location(C):
    """ get the location in an array of the highest score
    
    Parameters
    ----------
    C : np.array, size=(m,n)
        similarity score surface

    Returns
    -------
    di : integer
        horizontal location of highest score
    dj : integer
        vertical location of highest score
    snr : float
        signal to noise ratio
    max_corr : float
        value of highest point in the array

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    """
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    
    ij = np.unravel_index(np.argmax(C), C.shape, order='F') # 'C'
    di, dj = ij[::-1]
    di -= C.shape[0] // 2
    dj -= C.shape[1] // 2
    return di, dj, snr, max_corr

# supporting functions
def pad_images_and_filter_coord_list(M1, M2, geoTransform1, geoTransform2,\
                                     X_grd, Y_grd, ds1, ds2, same=True):
    """ pad imagery, depending on the template size, also transform and shift 
    the associated coordinate grids/lists

    Parameters
    ----------
    M1 : np.array, size=(m,n), ndim={2,3}
        first image array
    M2 : np.array, size=(m,n), ndim={2,3}
        second image array
    geoTransform1 : tuple
        affine transformation coefficients of array M1    
    geoTransform2 : tuple
        affine transformation coefficients of array M2
    X_grd : np.array, size=(k,l), type=float
        horizontal map coordinates of matching centers of array M1
    Y_grd : np.array, size=(k,l), type=float
        vertical map coordinates of matching centers of array M2
    ds1 : TYPE
        extra boundary to be added to the first data array
    ds2 : TYPE
        extra boundary to be added to the second data array
    same : dtype=bool, optional
        Either the same coordinates are used, but when a feature is tracked, 
        then the coordinates between time stamps differ. The default is True.

    Returns
    -------
    M1_new : np.array, size=(m+2*ds1,n+2*ds1), ndim={2,3}
        extended data array
    M2_new : np.array, size=(m+2*ds2,n+2*ds2), ndim={2,3}
        extended data array
    i1 : np.array, size=(_,1), dtype=integer
        vertical image coordinates of the template centers of first image
    j1 : np.array, size=(_,1), dtype=integer
        horizontal image coordinates of the template centers of first image
    i2 : np.array, size=(_,1), dtype=integer
        vertical image coordinates of the template centers of second image
    j2 : np.array, size=(_,1), dtype=integer
        horizontal image coordinates of the template centers of second image
    """
    if same: # matching done at the same location
        X1_grd, X2_grd, Y1_grd, Y2_grd = X_grd, X_grd, Y_grd, Y_grd
    
    else: # a moveable entity is tracked
        X1_grd, X2_grd = X_grd[:,0], X_grd[:,1]
        Y1_grd, Y2_grd = Y_grd[:,0], Y_grd[:,1]
    
    # map transformation to pixel domain
    I1_grd,J1_grd = map2pix(geoTransform1, X1_grd, Y1_grd)
    I1_grd = np.round(I1_grd).astype(np.int64) 
    J1_grd = np.round(J1_grd).astype(np.int64)

    i1,j1 = I1_grd.flatten(), J1_grd.flatten()

    I2_grd,J2_grd = map2pix(geoTransform2, X2_grd, Y2_grd)
    I2_grd = np.round(I2_grd).astype(np.int64) 
    J2_grd = np.round(J2_grd).astype(np.int64)

    i2,j2 = I2_grd.flatten(), J2_grd.flatten()

    # remove posts outside image   
    IN = np.logical_and.reduce((i1>=0, i1<=M1.shape[0], 
                                j1>=0, j1<=M1.shape[1],
                                i2>=0, i2<=M2.shape[0], 
                                j2>=0, j2<=M2.shape[1]))
    
    i1, j1, i2, j2 = i1[IN], j1[IN], i2[IN], j2[IN]
    
    # extend image size, so search regions at the border can be used as well
    M1_new = pad_radius(M1, ds1)
    i1 += ds1
    j1 += ds1

    M2_new = pad_radius(M2, ds2)
    i2 += ds2
    j2 += ds2
    
    return M1_new, M2_new, i1, j1, i2, j2

def pad_radius(I, radius):
    """ add extra boundary to array, so templates can be easier extracted    

    Parameters
    ----------
    I : np.array, size=(m,n) or (m,n,b)
        data array
    radius : positive integer
        extra boundary to be added to the data array

    Returns
    -------
    I_xtra : np.array, size=(m+2*radius,n+2*radius)
        extended data array
    """
    if I.ndim==3:
        I_xtra = np.pad(I, \
                        ((radius,radius), (radius,radius), (0,0)), \
                        'constant', constant_values=0)
    else:
        I_xtra = np.pad(I, \
                        ((radius,radius), (radius,radius)), \
                        'constant', constant_values=0)
    return I_xtra

def prepare_grids(im_stack, ds):
    """prepare stack by padding, dependent on the matching template

    the image stack is sampled by a template, that samples without overlap. all
    templates need to be of the same size, thus the image stack needs to be 
    enlarged if the stack is not a multitude of the template size   

    Parameters
    ----------
    im_stack : np.array, size(m,n,k)
        imagery array
    ds : integer
        size of the template, in pixels

    Returns
    -------
    im_stack : np.array, size(m+ds,n+ds,k)
        extended imagery array
    I_grd : np.array, size=(p,q)
        vertical image coordinates of the template centers
    J_grd : np.array, size=(p,q)
        horizontal image coordinates of the template centers

    """
    # padding is needed to let all imagery be of the correct template size
    i_pad = np.int(np.ceil(im_stack.shape[0]/ds)*ds - im_stack.shape[0])
    j_pad = np.int(np.ceil(im_stack.shape[1]/ds)*ds - im_stack.shape[1])
    im_stack = np.pad(im_stack, \
                      ((0, i_pad), (0, j_pad), (0, 0)), \
                      'constant', constant_values=(0, 0))
    
    # ul
    i_samp = np.arange(0,im_stack.shape[0]-ds,ds)
    j_samp = np.arange(0,im_stack.shape[1]-ds,ds)
    J_grd, I_grd = np.meshgrid(j_samp,i_samp)
    return im_stack, I_grd, J_grd

def make_templates_same_size(I1,I2):
    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space
    
    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt
 
    if I1.dim>I2.dim:
        I2 = I2[:,:,np.newaxis]
    elif I1.dim<I2.dim:
        I2 = I2[:,:,np.newaxis]
 
    md, nd = (ms-mt)//2, (ns-nt)//2
    if md==0 | nd==0: # I2[+0:-0, ... does not seem to work
        I2sub = I2
    else:
        if I1.ndim==3:
            if I1.shape[2]==I2.shape[2]:
                I2sub = I2[+md:-md, +nd:-nd, :]
            elif I1.shape[2]>I2.shape[2]:
                I2sub = np.repeat(I2[+md:-md,+nd:-nd], I1.shape[2], axis=2)
            elif I1.shape[2]<I2.shape[2]:
                I2sub = I2[+md:-md, +nd:-nd, :]
                I1 = np.repeat(I1, I2.shape[2], axis=2)
        else:        
            I2sub = I2[+md:-md, +nd:-nd]        
    return I1, I2sub

def test_bounds_reposition(d, temp_size, search_size):
    """
    See Also
    --------
    reposition_templates_from_center
    """    
    space_bound = (search_size-temp_size) // 2
    if abs(d) > space_bound:
        warnings.warn("part of the template will be out of the image" +
                      "with this displacement estimate")
        reposition_dir = np.sign(d)
        d = reposition_dir * np.minimum(abs(d), space_bound)
    return d

def reposition_templates_from_center(I1,I2,di,dj):
    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space
    
    di,dj = int(di),int(dj)
    di,dj = test_bounds_reposition(di,mt,ms), test_bounds_reposition(dj,nt,ns)
    
    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt
    assert I1.ndim==I2.ndim # should be the same dimension
    
    mc,nc = ms//2, ns//2 # center location    

    if I1.ndim==3:
        I2sub = I2[mc-(mt//2)-di : mc+(mt//2)-di, \
                   nc-(nt//2)-dj : nc+(nt//2)-dj, :]
    else: 
        I2sub = I2[mc-(mt//2)-di : mc+(mt//2)-di, \
                   nc-(nt//2)-dj : nc+(nt//2)-dj]       
    return I1, I2sub

def get_coordinates_of_template_centers(grid, temp_size):
    """
    When tiling an array into small templates, this function
    gives the locations of the centers.
    input:   grid           array (n x m)     array with data values
             temp_size       integer           size of the kernel in pixels
    output:  Iidx           array (k x l)     array with row coordinates
             Jidx           array (k x l)     array with collumn coordinates
    """
    radius = np.floor(temp_size / 2).astype('int')
    Iidx = np.arange(radius, grid.shape[0] - radius, temp_size)
    Jidx = np.arange(radius, grid.shape[1] - radius, temp_size)
    # FN ###################################
    # are the following lines equivalent to:
    # Iidx, Jidx = np.meshgrid(Iidx, Jidx)
    # It looks like, but Iidx and Jidx are switched!
    IidxNew = np.repeat(np.transpose([Iidx]), len(Jidx), axis=1)
    Jidx = np.repeat([Jidx], len(Iidx), axis=0)
    Iidx = IidxNew

    return Iidx, Jidx

def get_grid_at_template_centers(grid, temp_size):
    """
    When tiling an array into small templates, this function
    gives the value of the pixel in its center.
    input:   grid           array (n x m)     array with data values
             temp_size       integer           size of the kernel in pixels
    output:  gridnew        array (k x l)     data value of the pixel in the
                                              kernels center
    """
    (Iidx, Jidx) = get_coordinates_of_template_centers(grid, temp_size)

    return grid[Iidx, Jidx]