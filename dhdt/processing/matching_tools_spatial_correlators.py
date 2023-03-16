# general libraries
import numpy as np

# image processing libraries
from skimage.feature import match_template
from skimage.util import view_as_windows
from scipy import ndimage

from ..preprocessing.image_transforms import mat_to_gray

# spatial pattern matching functions
def normalized_cross_corr(I1, I2):
    """ simple normalized cross correlation

    Parameters
    ----------
    I1 : numpy.ndarray, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.ndarray, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)

    Returns
    -------
    ccs : numpy.array
        similarity surface,

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> _,im2,ti,tj,im1 = create_sample_image_pair(d=2**4, max_range=1)
    >>> C = normalized_cross_corr(im1,im2)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))

    Notes
    -----
    The following nomenclature is used:

    ccs: cross correlation surface
    """
    assert I1.ndim == I2.ndim, 'please provide data with the same dimension'
    assert I1.shape[0:2]<=I2.shape[0:2], \
        ('search space (I2) should be at least as big as the template (I1)')
    ccs = match_template(I2, I1)
    if I1.ndim==3: ccs = np.squeeze(ccs, axis=2)
    return ccs

def cumulative_cross_corr(I1, I2):
    """ doing normalized cross correlation on distance imagery

    Parameters
    ----------
    I1 : numpy.array, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.array, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)

    Returns
    -------
    ccs : numpy.array
        similarity surface

    Notes
    -----
    The following nomenclature is used:

    ccs: cross correlation surface
    """   
    assert I1.shape[0:2]<=I2.shape[0:2], \
        ('search space (I2) should be at least as big as the template (I1)')
    if isinstance(I1, np.floating):
        # get cut-off value
        cu = np.quantile(I1, 0.5)
        I1 = I1<cu
    if isinstance(I1, np.floating):        
        # get cut-off value
        cu = np.quantile(I2, 0.5)
        I2 = I2<cu
        
    I1new = ndimage.distance_transform_edt(I1)
    I2new = ndimage.distance_transform_edt(I2)
    
    ccs = match_template(I2new, I1new)
    return ccs

def sum_sq_diff(I1, I2):
    """ sum of squared difference correlation

    Parameters
    ----------
    I1 : numpy.array, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.array, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)

    Returns
    -------
    ssd : numpy.array, ndim=2
        dissimilarity surface

    Notes
    -----
    The following nomenclature is used:

    ssd: sum of squared differences
    """
    assert I1.ndim == I2.ndim, 'please provide data with the same dimension'
    assert I1.shape[0:2]<=I2.shape[0:2], \
        ('search space (I2) should be at least as big as the template (I1)')

    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                    shape=(I2.shape[0] - t_size[0] + 1,
                           I2.shape[1] - t_size[1] + 1,) +
                          t_size,
                    strides=I2.strides * 2)
    ssd = np.einsum('ijkl,kl->ij', y, I1)
    ssd *= - 2
    ssd += np.einsum('ijkl, ijkl->ij', y, y)
    ssd += np.einsum('ij, ij', I1, I1)
    return ssd

def sum_sad_diff(I1, I2):
    """ sum of absolute difference correlation

    Parameters
    ----------
    I1 : numpy.array, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.array, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)

    Returns
    -------
    sad : numpy.array, ndim=2
        dissimilarity surface

    Notes
    -----
    The following nomenclature is used:

    sad: sum of absolute difference
    """
    assert I1.ndim == I2.ndim, 'please provide data with the same dimension'
    assert I1.shape[0:2]<=I2.shape[0:2], \
        ('search space (I2) should be at least as big as the template (I1)')

    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                    shape=(I2.shape[0] - t_size[0] + 1,
                           I2.shape[1] - t_size[1] + 1,) +
                          t_size,
                    strides=I2.strides * 2)

    sad = np.sum(np.abs(y - I1[None,None, ...]), axis=(0,1))
    return sad

# maximum likelihood
def maximum_likelihood_func(I2,I1):
    # normalization need?
    I1, I2 = I1.flatten(), I2.flatten()

    I12 = I1+I2
    num = np.log(I2, where= I2>0) + \
          np.log(I1, where= I1>0) - \
          2*np.log(I12, where= I12>0) - \
          np.log(I1, where= I1>0)
    ml = np.divide(num, I1.size,
                   out=np.zeros_like(num), where= num>0)
    return np.sum(ml)

def maximum_likelihood(I1,I2):
    """ maximum likelihood texture tracking

    Parameters
    ----------
    I1 : numpy.array, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.array, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)

    Returns
    -------
    C : float
        texture correspondence surface

    References
    ----------
    .. [Er09] Erten et al. "Glacier velocity monitoring by maximum likelihood
              texture tracking" IEEE transactions on geosciences and remote
              sensing, vol.47(2) pp.394--405, 2009.
    """
    I1, I2 = mat_to_gray(I1), mat_to_gray(I2)

    C = ndimage.generic_filter(I2, maximum_likelihood_func,
                               size=I1.shape, extra_keywords = {'I1':I1})
    C[C>1.4] = 0
    return C

def weighted_normalized_cross_correlation(I1,I2,W1=None,W2=None):
    """

    Parameters
    ----------
    I1 : numpy.array, ndim=2, size=(m,n)
        grid with intensities (template)
    I2 : numpy.array, ndim=2, size=(k,l), {k>=m, l>=n}
        grid with intensities (search space)
    W1 : numpy.array, size=(m,n), dtype={boolean,float}
        weighting matrix for the template
    W2 : numpy.array, size=(m,n), dtype={boolean,float}
        weighting matrix for the search range

    Returns
    -------
    wncc : numpy.array
        weighted normalized cross-correlation
    support : numpy.array, range=0...1
        amount of overlap present and used in both imagery

    References
    ----------
    .. [AK21] Altena & Kääb "Quantifying river ice movement through a
              combination of European satellite monitoring services"
              International journal of applied Earth observation and
              geoinformation, vol.98 pp.102315, 2021.
    """
    (m1,n1) = I1.shape
    # create weighting matrix
    if W1 is None: W1 = np.ones_like(I1)
    if W2 is None: W2 = np.ones_like(I2)

    if np.all(W1==1) and np.all(W1==1): return normalized_cross_corr(I1, I2)

    # normalize weighting matrices
    W1[np.isnan(W1)], W2[np.isnan(W2)] = 0, 0
    W1, W2 = np.divide(W1,np.sum(W1)), np.divide(W2,np.sum(W2))

    M1,M2 = W1!=0, W2!=0
    # calculate sample population
    view_M2 = view_as_windows(M2, (m1, n1), 1)
    # amount of elements present in each template
    elem_M2 = np.sum(np.sum(view_M2,axis=-2),axis=-1).astype(np.float64)
    elem_M1 = np.sum(np.sum(M1)).astype(np.float64)

    view_M1 = np.repeat(np.repeat(M1[np.newaxis, np.newaxis, :],
                                  view_M2.shape[0], axis=0),
                        view_M2.shape[1], axis=1)
    view_M12 = np.logical_and(view_M1, view_M2)
    # elements that overlap
    elem_M12 = np.sum(np.sum(view_M12, axis=-2), axis=-1).astype(np.float64)
    del view_M12, view_M1, view_M2

    # calculate weighted means
    view_I2 = view_as_windows(I2, (m1,n1), 1)
    view_W2 = view_as_windows(W2, (m1,n1), 1)
    (m2,n2,_,_) = view_I2.shape

    wght_mu2 = np.sum(np.sum(np.multiply(view_W2, view_I2),axis=-2),axis=-1)
    wght_mu2 = np.squeeze(wght_mu2)
    wght_mu1 = np.sum(W1.flatten()*I1.flatten())

    # calculate weighted co-variances
    wght_var_1 = np.abs(I1 - wght_mu1)
    wght_var_W1 = np.multiply(wght_var_1, W1)
    wght_var_2 = (view_I2 - np.tile(wght_mu2[:,:,np.newaxis,np.newaxis],
                                    (1,1,m1,n1)))
    del view_I2
    sum_view_W2 = np.sum(np.sum(view_W2,
                                axis=-2), axis=-1)
    sum_view_W2_stack = np.tile(sum_view_W2[:,:,np.newaxis,np.newaxis],
                                (1,1,m1,n1))
    wght_var_W2 = np.divide(
        np.multiply(view_W2, wght_var_2),
        sum_view_W2_stack,
        where=sum_view_W2_stack!=0, out=np.zeros_like(sum_view_W2_stack))
    del sum_view_W2_stack, view_W2
    wght_var_W1_stack = np.repeat(np.repeat(wght_var_W1[np.newaxis,np.newaxis,:],
                                            m2, axis=0), n2, axis=1)

    # put terms together
    wght_cov_12 = np.sum(np.sum(np.multiply(wght_var_W1_stack,
                                            wght_var_W2),
                                axis=-2), axis=-1)
    wght_cov_12 = np.divide(wght_cov_12, elem_M12,
                            where=elem_M12!=0,
                            out=np.zeros_like(elem_M12))
    del wght_var_W1_stack

    wght_cov_22 = np.sum(np.sum(wght_var_W2**2,axis=-2),axis=-1)
    wght_cov_22 = np.divide(wght_cov_22, elem_M2,
                            where=elem_M2>=2,
                            out=np.ones_like(elem_M2))

    wght_cov_11 = np.sum(wght_var_W1**2)
    wght_cov_11 = np.divide(wght_cov_11, elem_M1,
                            where=elem_M1!=0,
                            out=np.ones_like(elem_M1))

    denom = np.sqrt(wght_cov_11 * wght_cov_22)
    wncc = np.divide(wght_cov_12, denom,
                     where=denom!=0, out=np.zeros_like(denom))
    support = elem_M12/(m1*n1)
    return wncc, support

def cosine_similarity(I1, I2):
    """ estimate the similarity of orientation imagery, via their dot product.
    Going along via different names, such as Gradient Inner Product (GIP)[Gl08]_
    or Cosine Similarity [De21]_.

    Parameters
    ----------
    I1 : numpy.array, size=(m,n), dtype=complex, {x ∈ ℂ | ║x║ = 1}
        image with intensities (template), given as complex orientation image
    I2 : numpy.array, size=(k,l), dtype=complex, {x ∈ ℂ | ║x║ = 1}
        image with intensities (search space), as complex orientation image

    Returns
    -------
    score : numpy.array, dtype=float, range=0...+1

    See Also
    --------
    dhdt.processing.matching_tools_frequency_correlators.orientation_corr

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> from dhdt.testing.matching_tools import create_sample_image_pair
    >>> from dhdt.generic.handler_im import get_grad_filters
    >>> from dhdt.processing.matching_tools import get_integer_peak_location
    >>> from dhdt.processing.matching_tools_frequency_filters import \
            normalize_power_spectrum

    >>> _,im2,ti,tj,im1 = create_sample_image_pair(d=2**4, max_range=1)

    complex orientation imagery are used here, so these need to be transformed

    >>> H_x = get_grad_filters(ftype='kroon', tsize=3, order=1)[0]
    >>> H_y = np.transpose(H_x)
    >>> G1 = ndimage.convolve(im2, H_x) + \
             1j*ndimage.convolve(im2, H_y)
    >>> G2 = ndimage.convolve(im1_big, H_x) + \
             1j*ndimage.convolve(im1_big, H_y)
    >>> G1, G2 = normalize_power_spectrum(G1), normalize_power_spectrum(G2)

    now processing that actual processing can be done

    >>> C = normalized_cross_corr(G1,G2)
    >>> di,dj,_,_ = get_integer_peak_location(C)

    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))

    References
    ----------
    .. [Gl08] Glocker et al. "Optical flow estimation with uncertainties through
              dynamic MRFs" IEEE conference on computer vision and pattern
              recognition, 2008.
    .. [De21] Dematteis et al., "Comparison of digital image correlation methods
              and the impact of noise in geoscience applications" Remote
              sensing, vol.13(2), pp.327, 2021.
    """

    assert np.iscomplexobj(I1), ('please provide a complex array')
    assert np.iscomplexobj(I2), ('please provide a complex array')
    assert I1.shape[0:2]<=I2.shape[0:2], \
        ('search space (I2) should be at least as big as the template (I1)')

    m1,n1 = I1.shape[0:2]

    view_I2 = view_as_windows(I2, (m1, n1), 1)
    view_I1 = np.repeat(np.repeat(I1[np.newaxis, np.newaxis, :],
                                  view_I2.shape[0], axis=0),
                        view_I2.shape[1], axis=1)

    dot_product = view_I1*np.conj(view_I2)
    score = np.abs(np.mean(np.mean(dot_product, axis=-2),
                           axis=-1)).astype(np.float64)
    return score

# weighted sum of differences
# sum of robust differences, see Li_03
# least squares matching