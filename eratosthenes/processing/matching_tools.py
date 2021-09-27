# general libraries
import warnings
import numpy as np

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
    return (im_stack, I_grd, J_grd)

def make_templates_same_size(I1,I2):
    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space
    
    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt
    assert I1.ndim==I2.ndim # should be the same dimension

    md, nd = (ms-mt)//2, (ns-nt)//2
    if md==0 | nd==0: # I2[+0:-0, ... does not seem to work
        I2sub = I2
    else:
        if I1.ndim==3:
            I2sub = I2[+md:-md, +nd:-nd, :]
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