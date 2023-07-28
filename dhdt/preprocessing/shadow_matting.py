# shadow matting

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from scipy import ndimage

from ..postprocessing.solar_tools import az_to_sun_vector


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides

    A_rolls = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)
    return A_rolls


def compute_laplacian(img, mask=None, eps=10 ** (-7), win_rad=1):
    """ computes matting Laplacian for a given image.

    Parameters
    ----------
    img : np.array, size=(m,n,3), ndim=3, dtype=float
        input image
    mask : np.array, size=(m,n), ndim=3, dtype=bool, optional
        mask of pixels for which Laplacian will be computed. The default is None.
    eps : float, optional
        regularization parameter controlling alpha smoothness
        from Equation 12 of the original paper. The default is 10**(-7).
    win_rad : integer, optional
        radius of window used to build matting Laplacian, that is the radius 
        of :math:`\omega_k` in Equation 12). The default is 1.

    Returns
    -------
    L : np.array
        sparse matrix holding Matting Laplacian.

    Notes
    -----
    Original code from: https://github.com/MarcoForte/closed-form-matting
    
    .. [Le08] Levin et al. "A closed-form solution to natural image matting"
              IEEE Transactions on pattern analysis and machine intelligence.
              vol.30(2) pp.228-242, 2008.
    """
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        dil_mat = np.ones((win_diam, win_diam), dtype=bool)
        mask = ndimage.morphology.binary_dilation(mask, \
                                                  structure=dil_mat)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / \
              win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - \
           (1.0 / win_size) * (
                       1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), \
                                shape=(h * w, h * w))
    return L


def closed_form_matting_with_prior(img,
                                   prior,
                                   prior_confidence,
                                   consts_map=None):
    """ applies closed form matting with prior alpha map to image.   

    Parameters
    ----------
    img : np.array, size=(m,n,3), ndim=3, dtype=float
        input image
    prior : np.array, size=(m,n), ndim=2, dtype=float, range=0...1
        image holding a-priori alpha map.
    prior_confidence : np.array, size=(m,n), ndim=2, dtype=float,
        prior holding confidence of a-priori alpha.
    consts_map : np.array, size=(m,n), ndim=2, dtype=bool, optional
        mask of pixels that aren't expected to change due to high a-priori 
        confidence. The default is None.

    Returns
    -------
    alpha : np.array, size=(m,n), ndim=2, dtype=float, range=0...1
        computed alpha map.

    Notes
    -----
    Original code from: https://github.com/MarcoForte/closed-form-matting
    
    .. [Le08] Levin et al. "A closed-form solution to natural image matting"
              IEEE Transactions on pattern analysis and machine intelligence.
              vol.30(2) pp.228-242, 2008.
    """

    assert img.shape[:-1] == prior.shape, \
        ('prior must be 2D matrix with height and width equal to image.')
    assert img.shape[:-1] == prior_confidence.shape, \
        ('prior_confidence must be 2D matrix with dimensions equal to image.')
    assert (consts_map is None) or img.shape[:-1] == consts_map.shape, \
        ('consts_map must be 2D matrix with dimensions equal to image.')

    laplacian = compute_laplacian(img, \
                                  ~consts_map if consts_map is not None else None)

    confidence = scipy.sparse.diags(prior_confidence.ravel())

    solution = scipy.sparse.linalg.spsolve(
        laplacian + confidence,
        prior.ravel() * prior_confidence.ravel())
    # put estimate within confidence bound: 0...1
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha


def closed_form_matting_with_trimap(img, trimap, trimap_confidence=100.0):
    """Apply Closed-Form matting to given image using trimap."""
    assert img.shape[:-1] == trimap.shape, \
        ('trimap must be 2D matrix with height and width equal to image.')

    consts_map = (trimap < 0.1) | (trimap > 0.9)

    alpha = closed_form_matting_with_prior(img, trimap, \
                                           trimap_confidence * consts_map, \
                                           consts_map)
    return alpha


def closed_form_matting_with_scribbles(img,
                                       scribbles,
                                       scribbles_confidence=100.0):
    """Apply Closed-Form matting to given image using scribbles image."""

    assert img.shape == scribbles.shape, \
        ('scribbles must have exactly same shape as image.')
    prior = np.sign(np.sum(scribbles - img, axis=2)) / 2 + 0.5
    consts_map = prior != 0.5
    alpha = closed_form_matting_with_prior(img, prior, \
                                           scribbles_confidence * consts_map, \
                                           consts_map)
    return alpha


def compute_confidence_through_sun_angle(L, az):
    """ the confidence of the shadow polygon is dependend on the sun angle, as
    well as, the elevation change. Thus this function calculates a distance 
    transform along this direction

    Parameters
    ----------
    L : np.array, size=(m,n), dtype=bool
        grid with shadow image.
    az : float, unit=degrees
        azimuth angle of the sun's direction.

    Returns
    -------
    W : np.array, size=(m,n), dtype=float
        distance transform along sun direction.
    """
    #  coordinate frame:
    #        |
    #   - <--|--> +
    #        |
    #        o_____
    #
    s = np.abs(az_to_sun_vector(az, indexing='ij'))
    W = np.maximum(ndimage.distance_transform_edt(~L, sampling=1 / s),
                   ndimage.distance_transform_edt(L, sampling=1 / s))
    return W
