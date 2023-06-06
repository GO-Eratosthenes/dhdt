# general libraries
import warnings
import numpy as np

from dhdt.generic.mapping_tools import map2pix


def get_integer_peak_location(C, metric=None):
    """ get the location in an array of the highest score

    Parameters
    ----------
    C : np.array, size=(m,n)
        similarity score surface
    metric : {'peak_abs' (default), 'peak_ratio', 'peak_rms', 'peak_ener',
              'peak_nois', 'peak_conf', 'peak_entr'}
        Metric to be used to describe the matching score, the following can be
        chosen from:

        * 'peak_abs' : the absolute score
        * 'peak_ratio' : the primary peak ratio i.r.t. the second peak
        * 'peak_rms' : the peak ratio i.r.t. the root mean square error
        * 'peak_ener' : the peaks' energy
        * 'peak_noise' : the peak reation i.r.t. to the noise
        * 'peak_conf' : the peak confidence
        * 'peak_entr' : the peaks' entropy

    Returns
    -------
    di,dj : integer
        horizontal and vertical location of highest score
    matching_metric : float
        metric as specified by 'method'
    max_corr : float
        value of highest point in the array

    Notes
    -----
    Based upon a image centered coordinate frame:

        .. code-block:: text

         +-----------+-----------+
         |indexing   |           |
         |system 'ij'|           |
         |           |           |
         |           |           |
         |           |         j |
         | ----------o---------> |
         |           |           |
         |           |           |
         |           |           |
         |correlation| i         |
         |image      v           |
         +-----------+-----------+

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools_organization import \
            get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    """
    from .matching_tools_organization import \
        estimate_match_metric

    assert type(C)==np.ndarray, ("please provide an array")
    max_corr = np.argmax(C)

    ij = np.unravel_index(max_corr, C.shape, order='F') # 'C'
    di, dj = ij[::-1]
    di -= C.shape[0] // 2
    dj -= C.shape[1] // 2

    if metric is None:
        score = 0
    else:
        score = estimate_match_metric(C, di=di, dj=dj, metric=metric)
    return di, dj, score, max_corr

def get_peak_indices(C, num_estimates=1):
    """ get the locations in an array where peaks are present

    Parameters
    ----------
    C : np.array, size=(m,n)
        scoring or voting surface
    num_estimates : integer
        number of peaks to be estimated

    Returns
    -------
    idx : np.array, size=(k,2), dtype=integer
        vertical and horizontal location of highest peak(s), based upon an image
        coordinate system, that is [row, collumn] indexing
    val : np.array, size=(k,)
        voting score at peak location

    Notes
    -----
    Based upon a image centered coordinate frame:

        .. code-block:: text

                 j
         +------->---------------+
         |indexing               |
         |system 'ij'            |
         |                       |
         v i                     |
         |                       |
         +-----------------------+
    """
    from skimage.morphology import extrema

    peak_C = extrema.local_maxima(C)
    ids = np.array(np.where(peak_C)) # this seems to go in rows
    scores = C[peak_C]

    # rank scores from max to minimum
    sort_idx = np.flip(np.argsort(scores))
    scores, ids = scores[sort_idx], ids[:,sort_idx]

    idx = np.zeros((num_estimates, C.ndim), dtype=int)
    val = np.zeros(num_estimates)

    maximal_num = np.minimum( len(scores), num_estimates)
    idx[:maximal_num,:] = ids[:,:maximal_num].T # swap axis, because of np.where
    val[:maximal_num] = scores[:maximal_num]
    return idx, val

# supporting functions
def get_template(I, idx_1, idx_2, radius):
    """ get a template, eventhough the index or template might be outside the
    given domain

    Parameters
    ----------
    I : np.array, size=(m,n), dtype={integer,float}
        large numpy array with intensities/data
    idx_1 : integer
        row index of the central pixel of interest
    idx_2 : integer
        collumn index of the central pixel of interest
    radius : integer
        shortest radius of the template

    Returns
    -------
    I_sub : np.array, size=(k,k)
        array with Gaussian peak in the center
    """
    sub_idx = np.mgrid[idx_1 - radius : idx_1 + radius +1,
                       idx_2 - radius : idx_2 + radius +1]

    sub_ids = np.ravel_multi_index(np.vstack((sub_idx[0].flatten(),
                                              sub_idx[1].flatten())),
                                   I.shape, mode='clip')
    I_sub = np.take(I, sub_ids, mode='clip')
    I_sub = np.reshape(I_sub, (2*radius+1, 2*radius+1))
    return I_sub

def pad_images_and_filter_coord_list(M1, M2, geoTransform1, geoTransform2,
                                     X_grd, Y_grd, ds1, ds2, same=True):
    """ pad imagery, depending on the template size, also transform and shift
    the associated coordinate grids/lists

    Parameters
    ----------
    M1 : {numpy.ndarray, numpy.masked.array}, size=(m,n), ndim={2,3}
        first image array
    M2 : {numpy.ndarray, numpy.masked.array}, size=(m,n), ndim={2,3}
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
    IN : np.array, size=(k,l), dtype=boolean
        classification of the grid, which are in and out

    See Also
    --------
    ..generic.mapping_tools.ref_trans : translates geoTransform
    """
    # init
    assert type(M1) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(M2) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert isinstance(geoTransform1, tuple), ('geoTransform should be a tuple')
    assert isinstance(geoTransform2, tuple), ('geoTransform should be a tuple')
    assert(X_grd.shape == Y_grd.shape) # should be of the same size
    assert type(ds1)==int, ("please provide an integer")
    assert type(ds2)==int, ("please provide an integer")

    if same: # matching done at the same location
        X1_grd, X2_grd, Y1_grd, Y2_grd = X_grd, X_grd, Y_grd, Y_grd

    else: # a moveable entity is tracked
        X1_grd, X2_grd = X_grd[:,0], X_grd[:,1]
        Y1_grd, Y2_grd = Y_grd[:,0], Y_grd[:,1]

    # map transformation to pixel domain
    I1_grd,J1_grd = map2pix(geoTransform1, X1_grd, Y1_grd)
#    I1_grd = np.round(I1_grd).astype(np.int64)
#    J1_grd = np.round(J1_grd).astype(np.int64)

    i1,j1 = I1_grd.flatten(), J1_grd.flatten()

    I2_grd,J2_grd = map2pix(geoTransform2, X2_grd, Y2_grd)
#    I2_grd = np.round(I2_grd).astype(np.int64)
#    J2_grd = np.round(J2_grd).astype(np.int64)

    i2,j2 = I2_grd.flatten(), J2_grd.flatten()

    i1,j1,i2,j2,IN = remove_posts_pairs_outside_image(M1,i1,j1, M2,i2,j2)

    # extend image size, so search regions at the border can be used as well
    M1_new = pad_radius(M1, ds1)
    i1 += ds1
    j1 += ds1

    M2_new = pad_radius(M2, ds2)
    i2 += ds2
    j2 += ds2

    return M1_new, M2_new, i1, j1, i2, j2, IN

def pad_radius(I, radius, cval=0):
    """ add extra boundary to array, so templates can be easier extracted

    Parameters
    ----------
    I : {numpy.ndarray, numpy.masked.array}, size=(m,n) or (m,n,b)
        data array
    radius : {positive integer, tuple}
        extra boundary to be added to the data array
    cval = {integer, flaot}
        conastant value to be used in the padding region

    Returns
    -------
    I_xtra : np.array, size=(m+2*radius,n+2*radius)
        extended data array
    """
    assert type(I) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    if not type(radius) is tuple: radius = (radius, radius)

    if I.ndim==3:
        if type(I) in (np.ma.core.MaskedArray, ):
            I_xtra = np.ma.array(np.pad(I,
                            ((radius[0],radius[1]), (radius[0],radius[1]), (0,0)),
                            'constant', constant_values=cval),
                                 mask=np.pad(np.ma.getmaskarray(I),
                                             ((radius[0],radius[1]),
                                              (radius[0],radius[1]),
                                              (0,0)),
                                              'constant', constant_values=True))
            return I_xtra
        I_xtra = np.pad(I,
                        ((radius[0],radius[1]), (radius[0],radius[1]), (0,0)),
                        'constant', constant_values=cval)
        return I_xtra
    if type(I) in (np.ma.core.MaskedArray, ):
        I_xtra = np.ma.array(np.pad(I,
                        ((radius[0],radius[1]), (radius[0],radius[1])),
                        'constant', constant_values=cval),
                             mask=np.pad(np.ma.getmaskarray(I),
                                         ((radius[0],radius[1]),
                                          (radius[0],radius[1])),
                                          'constant', constant_values=True))
        return I_xtra
    I_xtra = np.pad(I,
                    ((radius[0],radius[1]), (radius[0],radius[1])),
                    'constant', constant_values=cval)
    return I_xtra

def prepare_grids(im_stack, ds, cval=0):
    """prepare stack by padding, dependent on the matching template

    the image stack is sampled by a template, that samples without overlap. all
    templates need to be of the same size, thus the image stack needs to be
    enlarged if the stack is not a multitude of the template size

    Parameters
    ----------
    im_stack : {numpy.ndarray, numpy.masked.array}, size(m,n,k)
        imagery array
    ds : integer
        size of the template, in pixels
    cval : {integer, float}
        value to pad the imagery with

    Returns
    -------
    im_stack : numpy.ndarray, size(m+ds,n+ds,k)
        extended imagery array
    I_grd : np.array, size=(p,q)
        vertical image coordinates of the template centers
    J_grd : np.array, size=(p,q)
        horizontal image coordinates of the template centers

    """
    assert type(im_stack) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(ds)==int, ("please provide an integer")
    if im_stack.ndim==2: im_stack = np.atleast_3d(im_stack)

    # padding is needed to let all imagery be of the correct template size
    i_pad = int(np.ceil(im_stack.shape[0]/ds)*ds - im_stack.shape[0])
    j_pad = int(np.ceil(im_stack.shape[1]/ds)*ds - im_stack.shape[1])

    im_stack = np.pad(im_stack,
                      ((0, i_pad), (0, j_pad), (0, 0)),
                      'constant', constant_values=(cval, cval))
    im_stack = np.squeeze(im_stack)

    # ul
    i_samp = np.arange(0,im_stack.shape[0]-ds,ds)
    j_samp = np.arange(0,im_stack.shape[1]-ds,ds)
    J_grd, I_grd = np.meshgrid(j_samp,i_samp)
    return im_stack, I_grd, J_grd

def make_templates_same_size(I1,I2):
    assert type(I1) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")

    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space

    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt

    if I1.ndim>I2.ndim:
        I2 = np.atleast_3d(I2)
    elif I1.ndim<I2.ndim:
        I1 = np.atleast_3d(I1)

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

def remove_posts_outside_image(I, i, j):
    """

    Parameters
    ----------
    I : numpy.ndarray, size=(m,n)
        grid of interest
    i,j : numpy.ndarray, size=(k,)
        post locations given in image coordinates

    Returns
    -------
    i,j : numpy.ndarray, size=(l,)
        filtered locations within the data array "I"

    See Also
    --------
    remove_posts_pairs_outside_image
    """
    assert I.ndim>=2, ('please provide an 2D array')

    IN = np.logical_and.reduce((i>=0, i<(I.shape[0]-1), j>=0, j<(I.shape[1]-1)))
    i, j = i[IN], j[IN]
    return i, j, IN

def remove_posts_pairs_outside_image(I1, i1, j1, I2, i2, j2):
    """

    Parameters
    ----------
    I1 : numpy.ndarray, size=(m,n)
        raster array of interest, where point pairs are located
    i1,j1 : numpy.ndarray, size=(k,)
        post locations of first pair, given in image coordinates
    I2 : numpy.ndarray, size=(m,n)
        raster array of interest, where point pairs are located
    i2,j2 : numpy.ndarray, size=(k,)
        post locations of second pair, given in image coordinates

    Returns
    -------
    i1,j1,i2,j2 : numpy.ndarray, size=(l,)
        filtered locations within the data array "I1" or "I2"

    See Also
    --------
    remove_posts_outside_image
    """
    assert I1.ndim >= 2, ('please provide I1 as an 2D array')
    assert I2.ndim >= 2, ('please provide I2 as an 2D array')

    IN = np.logical_and.reduce((i1>=0, i1<(I1.shape[0]-1),
                                j1>=0, j1<(I1.shape[1]-1),
                                i2>=0, i2<(I2.shape[0]-1),
                                j2>=0, j2<(I2.shape[1]-1)))
    i1, j1, i2, j2 = i1[IN], j1[IN], i2[IN], j2[IN]
    return i1, j1, i2, j2, IN

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

def get_coordinates_of_template_centers(Grid, temp_size):
    """ create grid of template centers

    When tiling an array into smaller templates, this function
    gives the locations of the centers.

    Parameters
    ----------
    Grid : {numpy.ndarray, numpy.masked.array, tuple}, size=(m,n)
        array with data values, or a geoTransform
    temp_size : positive integer
        size of the kernel in pixels

    Returns
    -------
    I_idx : numpy.ndarray, size=(k,l)
        array with row coordinates
    J_idx : numpy.ndarray, size=(k,l)
        array with collumn coordinates

    See Also
    --------
    dhdt.processing.matching_tools.get_value_at_template_centers
    dhdt.processing.matching_tools.prepare_grids
    dhdt.generic.mapping_tools.pix2map

    Notes
    -----
    Two different coordinate system are used here:

        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | i         map         |
          based      v           based       |

    """

    assert type(Grid) in (np.ma.core.MaskedArray, np.ndarray, tuple), \
        ("please provide an array or tuple")
    assert type(temp_size)==int, ("please provide an integer")

    if isinstance(Grid, tuple):
        (m,n) = Grid[-2:]
    else:
        (m,n) = Grid.shape[:2]

    radius = np.floor(temp_size / 2).astype('int')
    I_idx,J_idx = np.mgrid[radius:(m-radius):temp_size,
                  radius:(n-radius):temp_size]
    return I_idx, J_idx

def get_value_at_template_centers(Grid, temp_size):
    """ When tiling an array into small templates, this function gives the
    value of the pixel in its center.

    Parameters
    ----------
    Grid : {numpy.ndarray, numpy.masked.array}, size=(m,n), ndim={2,3}
        array with data values
    temp_size : integer
        size of the kernel in pixels

    Returns
    -------
    grid_centers_values : np.ndarray, size=(k,l), ndim={2,3}
        value of the pixel in the kernels' center

    See Also
    --------
    dhdt.processing.matching_tools.get_coordinates_of_template_centers
    """
    assert type(Grid) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(temp_size)==int, ("please provide an integer")

    Iidx, Jidx = get_coordinates_of_template_centers(Grid, temp_size)
    if (Grid.ndim == 3):
        m,n = Iidx.shape
        b = Grid.shape[2]
        Iidx, Jidx = np.tile(np.atleast_3d(Iidx), (1,1,b)), \
                     np.tile(np.atleast_3d(Jidx), (1,1,b))
        Kidx = np.ones((m,n,b))
        Kidx = np.einsum('k,ijk->ijk',np.linspace(0,b-1,b),Kidx).astype(int)
        grid_centers_values = Grid[Iidx, Jidx, Kidx]
    else:
        grid_centers_values = Grid[Iidx, Jidx]

    return grid_centers_values

def get_data_and_mask(I,M=None):
    """ sometimes data is given in masked array form (numpy.ma), then this is
    separated into regular numpy.arrays

    Parameters
    ----------
    I : {numpy.array, numpy.masked.array}
        data grid
    M : {numpy.array, numpy.masked.array}, default=None
        masking array, where True means data, and False neglecting elements

    Returns
    -------
    I : numpy.array
        data array
    M : numpy.array, dtype=bool
        masking array, where True means data, and False neglecting elements
    """
    # make compatible with masekd array
    if type(I)==np.ma.core.MaskedArray:
        if M is None:
            M = np.invert(np.ma.getmaskarray(I))
        else:
            if type(M)==np.ma.core.MaskedArray: M = np.ma.getdata(M)
            M = np.logical_and(M, np.invert(np.ma.getmaskarray(I)))
        I = np.ma.getdata(I)
    else:
        if M is None: M = np.ones_like(I)
        if M.size==0: M = np.ones_like(I)
        if type(M)==np.ma.core.MaskedArray: M = np.ma.getdata(M)
        M = M.astype(dtype=bool)
    return I, M