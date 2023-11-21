import numpy as np
from scipy.optimize import fsolve

from .matching_tools import get_integer_peak_location


def is_estimate_away_from_border(C, i, j, ds=1):
    over_border = (np.abs(i) + ds >= (C.shape[0] + 1) // 2) or \
                  (np.abs(j) + ds >= (C.shape[1] + 1) // 2)
    return not (over_border)


# sub-pixel localization of the correlation peak
def get_top_moment(C, ds=1, top=np.array([])):
    """ find location of highest score through bicubic fitting

    Parameters
    ----------
    C : numpy.ndarray, size=(_,_)
        similarity surface
    ds : integer, default=1
        size of the radius to use neighboring information
    top : numpy.ndarray, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    References
    ----------
    .. [Fe12] Feng et al. "A subpixel registration algorithm for low PSNR
              images" IEEE international conference on advanced computational
              intelligence, pp.626-630, 2012.
    .. [MG15] Messerli & Grinstad, "Image georectification and feature tracking
              toolbox: ImGRAFT" Geoscientific instrumentation, methods and data
              systems, vol.4(1) pp.23-34, 2015.
    """
    assert ds > 0, 'please provide positive integer'
    (subJ, subI) = np.meshgrid(np.linspace(-ds, +ds, 2 * ds + 1),
                               np.linspace(-ds, +ds, 2 * ds + 1))
    subI, subJ = subI.ravel(), subJ.ravel()

    if top.size == 0:
        # find highest score
        di, dj, max_corr, snr = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    if is_estimate_away_from_border(C, di, dj, ds):  # estimate sub-pixel top
        idx_mid = int(np.floor((2. * ds + 1)**2 / 2))

        i_sub = C.shape[0] // 2 + np.arange(-ds, +ds + 1) + di
        j_sub = C.shape[1] // 2 + np.arange(-ds, +ds + 1) + dj

        Csub = C[i_sub[:, None], j_sub[None, :]].ravel()
        Csub = Csub - np.mean(np.hstack((Csub[0:idx_mid], Csub[idx_mid + 1:])))

        IN = Csub > 0

        m = np.array([
            np.divide(np.sum(subI[IN] * Csub[IN]), np.sum(Csub[IN])),
            np.divide(np.sum(subJ[IN] * Csub[IN]), np.sum(Csub[IN]))
        ])
        ddi, ddj = m[0], m[1]
    else:  # top at the border
        ddi, ddj = 0, 0

    return ddi, ddj, i_int, j_int


def get_top_blue(C, ds=1, top=np.array([])):  # todo
    """ find top of correlation peak through best linear unbiased estimation

    References
    ----------
    .. [1]
    """
    # admin
    (subJ, subI) = np.meshgrid(np.linspace(-ds, +ds, 2 * ds + 1),
                               np.linspace(-ds, +ds, 2 * ds + 1))
    subI, subJ = subI.ravel(), subJ.ravel()

    # estimate Jacobian
    H_x = np.array([[-17., 0., 17.], [-61., 0., 61.], [-17., 0., 17.]]) / 95
    # estimate Hessian
    H_xx = 8 / np.array([[105, -46, 105], [50, -23, 50], [105, -46, 105]])
    H_xy = 11 / np.array([[-114, np.inf, +114], [np.inf, np.inf, np.inf],
                          [+114, np.inf, -114]])

    if top.size == 0:
        # find highest score
        di, dj, max_corr, snr = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    if not is_estimate_away_from_border(C, di, dj, ds):
        return 0, 0, i_int, j_int

    # estimate sub-pixel top
    idx_mid = int(np.floor((2. * ds + 1)**2 / 2))

    i_sub = C.shape[0] // 2 + np.arange(-ds, +ds + 1) + di
    j_sub = C.shape[1] // 2 + np.arange(-ds, +ds + 1) + dj
    Csub = C[i_sub[:, None], j_sub[None, :]].ravel()

    H_y, H_x = H_x.T.ravel(), H_x.ravel()
    H_yy, H_yx = H_xx.T.ravel(), H_xy.T.ravel()
    H_xy, H_xx = H_xy.ravel(), H_xx.ravel()

    Jac = np.array([[Csub @ H_x], [Csub @ H_y]])
    Hes = np.array([[Csub @ H_xx, Csub @ H_xy], [Csub @ H_yx, Csub @ H_yy]])
    # estimate sub-pixel top
    m = np.array([[di], [dj]]) - np.linalg.inv(Hes) @ Jac
    ddi, ddj = m[0], m[1]

    return ddi, ddj, i_int, j_int


def get_top_gaussian(C, top=np.array([])):
    """ find location of highest score through 1D gaussian fit

    Parameters
    ----------
    C : numpy.ndarray, size=(_,_)
        similarity surface
    top : numpy.ndarray, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
        ddi : float
        estimated subpixel location on the vertical axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [WG91] Willert & Gharib, "Digital particle image velocimetry",
              Experiments in fluids, vol.10 pp.181-193, 1991.
    .. [AV06] Argyriou & Vlachos, "A Study of sub-pixel motion estimation using
              phase correlation" Proceeding of the British machine vision
              conference, pp.387-396, 2006.
    .. [Ra18] Raffel et al. "Particle Image Velocimetry" Ch.6 pp.184 2018.
    """
    if top.size == 0:  # find highest score
        di, dj, max_corr, snr = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    ddi = ((np.log(C[di + 1, dj]) - np.log(C[di - 1, dj])) / 2 *
           (2 * np.log(C[di, dj]) - np.log(C[di - 1, dj]) -
            np.log(C[di + 1, dj])))
    ddj = ((np.log(C[di, dj + 1]) - np.log(C[di, dj - 1])) / 2 *
           (2 * np.log(C[di, dj]) - np.log(C[di, dj - 1]) -
            np.log(C[di, dj + 1])))
    return ddi, ddj, i_int, j_int


def get_top_centroid(C, top=np.array([])):
    """ find location of highest score through 1D centorid fit

    Parameters
    ----------
    C : numpy.ndarray, size=(_,_)
        similarity surface
    top : numpy.ndarray, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
        ddi : float
        estimated subpixel location on the vertical axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [Ra18] Raffel et al. "Particle Image Velocimetry" Ch.6 pp.184, 2018.
    """
    if top.size == 0:  # find highest score
        di, dj, max_corr, snr = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border(C, i_int, j_int, ds=1):
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    ddi = (((i_int - 1) * C[di - 1, dj] + i_int * C[di, dj] +
            (i_int + 1) * C[di + 1, dj]) /  # noqa: E501
           (C[di - 1, dj] + C[di, dj] + C[di + 1, dj]))
    ddj = (((j_int - 1) * C[di, dj - 1] + j_int * C[di, dj] +
            (j_int + 1) * C[di, dj + 1]) /  # noqa: E501
           (C[di, dj - 1] + C[di, dj] + C[di, dj + 1]))
    ddi -= i_int
    ddj -= j_int

    return ddi, ddj, i_int, j_int


def get_top_mass(C, top=np.array([])):
    """ find location of highest score through 1D center of mass

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [FN96] Fisher & Naidu, "A Comparison of algorithms for subpixel peak
              detection" in Image Technology - Advances in image processing,
              multimedia and machine vision pp.385-404, 1996.
    """

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:  # estimate sub-pixel along each axis
        return 0, 0, i_int, j_int

    ddi = (C[di + 1, dj] - C[di - 1, dj]) / \
          (C[di - 1, dj] + C[di, dj] + C[di + 1, dj])
    ddj = (C[di, dj + 1] - C[di, dj - 1]) / \
          (C[di, dj - 1] + C[di, dj] + C[di, dj + 1])

    return ddi, ddj, i_int, j_int


def get_top_blais(C, top=np.array([])):
    """ find location of highest score through forth order filter

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [BR86] Blais & Rioux, "Real-time numerical peak detector" Signal
              processing vol.11 pp.145-155, 1986.
    """

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    if C[di + 1, dj] > C[di - 1, dj]:
        gx_0 = C[di - 2, dj] + C[di - 1, dj] - C[di + 1, dj] - C[di + 2, dj]
        gx_1 = C[di - 1, dj] + C[di - 0, dj] - C[di + 2, dj] - C[di + 3, dj]
        ddi = (gx_0 / (gx_0 - gx_1))
    else:
        gx_0 = C[di - 2, dj] + C[di - 1, dj] - C[di + 1, dj] - C[di + 2, dj]
        gx_1 = C[di - 3, dj] + C[di - 2, dj] - C[di + 0, dj] - C[di + 1, dj]
        ddi = (gx_1 / (gx_1 - gx_0)) - 1

    if C[di, dj + 1] > C[di, dj - 1]:
        gx_0 = C[di, dj - 2] + C[di, dj - 1] - C[di, dj + 1] - C[di, dj + 2]
        gx_1 = C[di, dj - 1] + C[di, dj - 0] - C[di, dj + 2] - C[di, dj + 3]
        ddj = (gx_0 / (gx_0 - gx_1))
    else:
        gx_0 = C[di, dj - 2] + C[di, dj - 1] - C[di, dj + 1] - C[di, dj + 2]
        gx_1 = C[di, dj - 3] + C[di, dj - 2] - C[di, dj + 0] - C[di, dj + 1]
        ddj = (gx_1 / (gx_1 - gx_0)) - 1

    return ddi, ddj, i_int, j_int


def get_top_parabolic(C, top=np.array([]), ds=1):
    """ find location of highest score through 1D parabolic fit

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [AV06] Argyriou & Vlachos, "A Study of sub-pixel motion estimation using
              phase correlation" Proceeding of the British machine vision
              conference, pp.387-396), 2006.
    .. [Ra18] Raffel et al. "Particle Image Velocimetry" Ch.6 pp.184 2018.
    .. [Br21] Bradley, "Sub-pixel registration for low cost, low dosage, X-ray
              phase contrast imaging", 2021.
    """

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border(C, i_int, j_int, ds=ds):
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    if ds == 1:  # use direct neighbours
        ddi = np.divide(C[di + 1, dj] - C[di - 1, dj],
                        2 * ((2 * C[di, dj]) - C[di - 1, dj] - C[di + 1, dj]))
        ddj = np.divide((C[di, dj + 1] - C[di, dj - 1]),
                        2 * ((2 * C[di, dj]) - C[di, dj - 1] - C[di, dj + 1]))
    else:  # use five points, from [Br21]_
        ddi = np.divide(
            7 * ((2 * C[di + 2, dj]) + C[di + 1, dj] - C[di - 1, dj] -
                 (2 * C[di - 2, dj])),
            5 * (+(2 * C[di - 2, dj]) - C[di - 1, dj] -
                 (2 * C[di, dj]) - C[di + 1, dj] + (2 * C[di + 2, dj])))
        ddj = np.divide(
            7 * ((2 * C[di, dj + 2]) + C[di, dj + 1] - C[di, dj - 1] -
                 (2 * C[di, dj - 2])),
            5 * (+(2 * C[di, dj - 2]) - C[di, dj - 1] -
                 (2 * C[di, dj]) - C[di, dj + 1] + (2 * C[di, dj + 2])))
    return ddi, ddj, i_int, j_int


def get_top_equiangular(C, top=np.array([])):
    """ find location of highest score along each axis by equiangular line

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the crossing

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [SO05] Shimizu & Okutomi. "Sub-pixel estimation error cancellation on
              area-based matching" International journal of computer vision,
              vol.63(3), pp.207–224, 2005.
    """

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:  # estimate sub-pixel along each axis
        return 0, 0, i_int, j_int
    # estimate sub-pixel along each axis
    if C[di + 1, dj] < C[di - 1, dj]:
        ddi = .5 * np.divide(C[di + 1, dj] - C[di - 1, dj],
                             C[di, dj] - C[di - 1, dj])
    else:
        ddi = .5 * np.divide(C[di + 1, dj] - C[di - 1, dj],
                             C[di, dj] - C[di + 1, dj])

    if C[di, dj + 1] < C[di, dj - 1]:
        ddj = .5 * np.divide(C[di, dj + 1] - C[di, dj - 1],
                             C[di, dj] - C[di, dj - 1])
    else:
        ddj = .5 * np.divide(C[di, dj + 1] - C[di, dj - 1],
                             C[di, dj] - C[di, dj + 1])

    return ddi, ddj, i_int, j_int


def get_top_birchfield(C, top=np.array([])):
    """ find location of highest score along each axis

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [BT99] Birchfield & Tomasi. "Depth discontinuities by pixel-to-pixel
              stereo" International journal of computer vision, vol. 35(3)
              pp.269-293, 1999.
    """

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:  # estimate sub-pixel along each axis
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    I_m, I_p = .5 * (C[di - 1, dj] + C[di, dj]), .5 * (C[di + 1, dj] +
                                                       C[di, dj])
    I_min = np.amin([I_m, I_p, C[di, dj]])
    I_max = np.amax([I_m, I_p, C[di, dj]])
    # swapped, since Birchfield uses dissimilarity
    ddi = np.amax([0, I_max - C[di, dj], C[di, dj] - I_min])

    I_m, I_p = .5 * (C[di, dj - 1] + C[di, dj]), .5 * (C[di, dj + 1] +
                                                       C[di, dj])
    I_min = np.amin([I_m, I_p, C[di, dj]])
    I_max = np.amax([I_m, I_p, C[di, dj]])
    ddj = np.amax([0, I_max - C[di, dj], C[di, dj] - I_min])

    return ddi, ddj, i_int, j_int


def get_top_ren(C, top=np.array([])):
    """ find location of highest score

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [Re10] Ren et al. "High-accuracy sub-pixel motion estimation from noisy
              images in Fourier domain." IEEE transactions on image processing,
              vol.19(5) pp.1379-1384, 2010.
    """
    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    D_i = C[di + 1, dj] - C[di - 1, dj]
    ddi = np.divide(np.sign(D_i), 1 + np.divide(C[di, dj], np.abs(D_i)))
    D_j = C[di, dj + 1] - C[di, dj - 1]
    ddj = np.divide(np.sign(D_j), 1 + np.divide(C[di, dj], np.abs(D_j)))
    return ddi, ddj, i_int, j_int


def get_top_triangular(C, top=np.array([])):
    """ find location of highest score through triangular fit

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [OC91] Olsen & Coombs, "Real-time vergence control for binocular robots"
              International journal of computer vision, vol.7(1), pp.67-89,
              1991.
    """
    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel along each axis
    I_m, I_p = C[di - 1, dj], C[di + 1, dj]
    I_min, I_max = np.amin([I_m, I_p]), np.amax([I_m, I_p])
    I_sign = 2 * (I_p > I_m) - 1
    ddi = I_sign * np.divide(1 - (I_max - I_min), C[di, dj] - I_min)

    I_m, I_p = C[di, dj - 1], C[di, dj + 1]
    I_min, I_max = np.amin([I_m, I_p]), np.amax([I_m, I_p])
    I_sign = 2 * (I_p > I_m) - 1
    ddj = I_sign * np.divide(1 - (I_max - I_min), C[di, dj] - I_min)

    return ddi, ddj, i_int, j_int


def get_top_esinc(C, ds=1, top=np.array([])):
    '''find location of highest score using exponential esinc function

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [AV06] Argyriou & Vlachos, "A study of sub-pixel motion estimation using
              phase correlation", proceedings of the British machine vision
              conference, 2006
    '''
    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border(C, i_int, j_int, ds=ds):
        return 0, 0, i_int, j_int

    # estimate sub-pixel per axis
    Cj = C[di, dj - ds:dj + ds + 1].ravel()
    Ci = C[di - ds:di + ds + 1, dj].ravel()

    def func_esinc(x, y):
        a, b, c = x
        # todo
        #        coord = np.linspace(1, y.size, y.size) - (y.size//2+1)
        #        return y - a * np.exp(-(b * (coord - c)) ** 2) * \
        #            (np.sin(np.pi * (coord - c)) / np.pi * (coord - c))**2
        return [(y[0] - a * np.exp(-(b * (-1 - c))**2) *
                 (np.sin(np.pi * (-1 - c)) / np.pi * (-1 - c)))**2,
                (y[1] - a * np.exp(-(b * (+0 - c))**2) *
                 (np.sin(np.pi * (+0 - c)) / np.pi * (+0 - c)))**2,
                (y[2] - a * np.exp(-(b * (+1 - c))**2) *
                 (np.sin(np.pi * (+1 - c)) / np.pi * (+1 - c)))**2]

    _, _, jC = fsolve(func_esinc, (1.0, 1.0, 0.0), args=Cj)
    _, _, iC = fsolve(func_esinc, (1.0, 1.0, 0.0), args=Ci)

    return iC, jC, i_int, j_int


def get_top_2d_gaussian(C, top=np.array([])):
    '''find location of highest score using 2D Gaussian

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [NH05] Nobach & Honkanen, "Two-dimensional Gaussian regression for
              sub-pixel displacement estimation in particle image velocimetry
              or particle position estimation in particle tracking
              velocimetry", Experiments in fluids, vol.38 pp.511-515, 2005
    '''
    (Jsub, Isub) = np.meshgrid(np.linspace(-1, +1, 3), np.linspace(-1, +1, 3))
    Isub = Isub.ravel()
    Jsub = Jsub.ravel()

    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel along both axis
    i_sub = np.arange(-1, +2) + di
    j_sub = np.arange(-1, +2) + dj
    Csub = C[i_sub[:, None], j_sub[None, :]].ravel()
    Clog = np.log(Csub)

    # estimate sub-pixel per axis
    c_10 = (1 / 6) * np.sum(Isub * Clog)
    c_01 = (1 / 6) * np.sum(Jsub * Clog)
    c_11 = (1 / 4) * np.sum(Isub * Jsub * Clog)
    c_20 = (1 / 6) * np.sum(((3 * Isub**2) - 2) * Clog)
    c_02 = (1 / 6) * np.sum(((3 * Jsub**2) - 2) * Clog)

    ddj = np.divide((c_11 * c_10) - (2 * c_01 * c_20),
                    (4 * c_20 * c_02) - (c_11**2))
    ddi = np.divide((c_11 * c_01) - (2 * c_10 * c_02),
                    (4 * c_20 * c_02) - (c_11**2))

    return ddi, ddj, i_int, j_int


def get_top_paraboloid(C, top=np.array([])):
    '''find location of highest score using paraboloid

    Parameters
    ----------
    C : numpy.array, size=(_,_)
        similarity surface
    top : numpy.array, size=(1,2)
        location of the maximum score

    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    i_int : integer
        location of highest score on the vertical axis
    j_int : integer
        location of highest score on the horizontal axis

    Notes
    -----
    .. [Pa20] Pallotta et al. "Subpixel SAR image registration through
              parabolic interpolation of the 2-D cross correlation", IEEE
              transactions on geoscience and remote sensing, vol.58(6)
              pp.4132--4144, 2020.
    '''
    if top.size == 0:  # find highest score
        di, dj, _, _ = get_integer_peak_location(C)
    else:
        di, dj = top[0], top[1]
    i_int, j_int = np.copy(di), np.copy(dj)

    di += C.shape[0] // 2  # using a central coordinate system
    dj += C.shape[1] // 2

    if not is_estimate_away_from_border:
        return 0, 0, i_int, j_int

    # estimate sub-pixel per axis
    a_1 = C[di + 0, dj + 0]
    a_2, a_3 = C[di + 1, dj + 0], C[di - 1, dj + 0]
    a_4, a_5 = C[di + 1, dj + 0], C[di - 1, dj + 0]

    a_6 = np.max(
        np.array([
            C[di + 1, dj + 1], C[di - 1, dj + 1], C[di + 1, dj - 1], C[di - 1,
                                                                       dj - 1]
        ]))

    a = a_6 + a_1 - a_2 - a_4
    b = a_4 + a_5 - a_1 - a_1
    c = a_2 + a_3 - a_1 - a_1

    ddj = np.divide(-a * (a_4 - a_5) + b * (a_2 - a_3),
                    (2 * a * a) - (2 * b * c))
    ddi = np.divide(-a * (a_2 - a_3) + b * (a_4 - a_5),
                    (2 * a * a) - (2 * b * c))
    return ddi, ddj, i_int, j_int
