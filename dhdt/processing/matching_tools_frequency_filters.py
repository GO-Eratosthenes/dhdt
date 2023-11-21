import warnings

import numpy as np
from scipy import fft, interpolate, ndimage, signal

from ..generic.unit_conversion import deg2compass
# frequency preparation
from ..preprocessing.image_transforms import mat_to_gray


def perdecomp(img):
    """calculate the periodic and smooth components of an image, based upon
    [Mo11]_.

    Parameters
    ----------
    img : numpy.ndarray, size=(m,n)
        array with intensities

    Returns
    -------
    per : numpy.ndarray, size=(m,n)
        periodic component
    cor : numpy.ndarray, size=(m,n)
        smooth component

    References
    ----------
    .. [Mo11] Moisan, L. "Periodic plus smooth image decomposition", Journal of
              mathematical imaging and vision vol. 39.2 pp. 161-179, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,_,_,_,_ = create_sample_image_pair(d=2**7, max_range=1)
    >>> per,cor = perdecomp(im1)

    >>> spec1 = np.fft.fft2(per)
    """
    assert type(img) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")

    # don't need to process empty arrays
    if (0 in img.shape):
        return img, np.zeros_like(img)

    img = img.astype(float)
    if img.ndim == 2:
        (m, n) = img.shape
        per = np.zeros((m, n), dtype=float)

        per[+0, :] = +img[0, :] - img[-1, :]
        per[-1, :] = -per[0, :]

        per[:, +0] = per[:, +0] + img[:, +0] - img[:, -1]
        per[:, -1] = per[:, -1] - img[:, +0] + img[:, -1]

    elif img.ndim == 3:
        (m, n, b) = img.shape
        per = np.zeros((m, n, b), dtype=float)

        per[+0, :, :] = +img[0, :, :] - img[-1, :, :]
        per[-1, :, :] = -per[0, :, :]

        per[:, +0, :] = per[:, +0, :] + img[:, +0, :] - img[:, -1, :]
        per[:, -1, :] = per[:, -1, :] - img[:, +0, :] + img[:, -1, :]

    fy = np.cos(2 * np.pi * (np.arange(0, m)) / m)
    fx = np.cos(2 * np.pi * (np.arange(0, n)) / n)

    Fx = np.repeat(fx[np.newaxis, :], m, axis=0)
    Fy = np.repeat(fy[:, np.newaxis], n, axis=1)
    Fx[0, 0] = 0
    if img.ndim == 3:
        Fx = np.repeat(Fx[:, :, np.newaxis], b, axis=2)
        Fy = np.repeat(Fy[:, :, np.newaxis], b, axis=2)
        cor = np.real(np.fft.ifftn(np.fft.fft2(per) * .5 / (2 - Fx - Fy)))
    else:
        cor = np.real(np.fft.ifft2(np.fft.fft2(per) * .5 / (2 - Fx - Fy)))
    per = img - cor
    return per, cor


def make_template_float(Z):
    """ templates for frequency matching should be float and ideally have
    limited border issues.

    Parameters
    ----------
    Z : numpy.ndarray, size={(m,n), (m,n,b)}
        grid with intensity values

    Returns
    -------
    Z : numpy.ndarray, size={(m,n), (m,n,b)}, dtype=float, range=0...1
        grid with intensity values
    """
    if Z.dtype.type in (np.uint8, np.uint16):
        Z = mat_to_gray(Z)  # transform to float in range 0...1
        Z = perdecomp(Z)[0]  # spectral pre-processing
        Z = np.atleast_2d(np.squeeze(Z))
    return Z


def normalize_power_spectrum(Q):
    """transform spectrum to complex vectors with unit length

    Parameters
    ----------
    Q : numpy.ndarray, size=(m,n), dtype=complex
        cross-spectrum

    Returns
    -------
    Qn : numpy.ndarray, size=(m,n), dtype=complex
        normalized cross-spectrum, that is elements with unit length

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    >>> Q = spec1 * np.conjugate(spec2) # fourier based image matching
    >>> Qn = normalize_spectrum(Q)

    """
    assert isinstance(Q, np.ndarray), "please provide an array"
    Qn = np.divide(Q, abs(Q), out=np.zeros_like(Q), where=Q != 0)
    return Qn


def local_coherence(Q, ds=1):
    """ estimate the local coherence of a spectrum

    Parameters
    ----------
    Q : numpy.ndarray, size=(m,n), dtype=complex
        array with cross-spectrum, with centered coordinate frame
    ds : integer, default=1
        kernel radius to describe the neighborhood

    Returns
    -------
    M : numpy.ndarray, size=(m,n), dtype=float
        vector coherence from no to ideal, i.e.: 0...1

    See Also
    --------
    thresh_masking

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> # create cross-spectrum with random displacement
    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    >>> Q = spec1 * np.conjugate(spec2)
    >>> Q = normalize_spectrum(Q)
    >>> Q = np.fft.fftshift(Q) # transform to centered grid

    >>> C = local_coherence(Q)

    >>> plt.figure(), plt.imshow(C, cmap='OrRd'), plt.colorbar(), plt.show()
    >>> plt.figure(), plt.imshow(np.angle(Q), cmap='twilight'),
    >>> plt.colorbar(), plt.show()
    """
    assert isinstance(Q, np.ndarray), "please provide an array"

    diam = 2 * ds + 1
    C = np.zeros_like(Q)
    isteps, jsteps = np.meshgrid(np.linspace(-ds, +ds, 2 * ds + 1, dtype=int),
                                 np.linspace(-ds, +ds, 2 * ds + 1, dtype=int))
    IN = np.ones(diam**2, dtype=bool)
    IN[diam**2 // 2] = False
    isteps, jsteps = isteps.flatten()[IN], jsteps.flatten()[IN]

    for idx, istep in enumerate(isteps):
        jstep = jsteps[idx]
        Q_step = np.roll(Q, (istep, jstep))
        # if the spectrum is normalized, then no division is needed
        C += Q * np.conj(Q_step)
    C = np.abs(C) / np.sum(IN)
    return C


def make_fourier_grid(Q,
                      indexing='ij',
                      system='radians',
                      shift=True,
                      axis='center'):
    """
    The four quadrants of the coordinate system of the discrete Fourier
    transform are flipped. This function gives its coordinate system as it
    would be in a map (xy) or pixel based (ij) system.

    Parameters
    ----------
    Q : numpy.ndarray, size=(m,n), dtype=complex
        Fourier based (cross-)spectrum.
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates
    system : {‘radians’, ‘unit’, 'normalized'}
       the extent of the cross-spectrum can span from
         * "radians" : -pi..+pi (default)
         * "unit" : -1...+1
         * "normalized" : -0.5...+0.5
         * "pixel" : -m/2...+m/2
    shift : boolean, default=True
        the coordinate frame of a Fourier transform is flipped

    Returns
    -------
    F_1 : numpy.ndarray, size=(m,n), dtype=integer
        first coordinate index of the Fourier spectrum in a map system.
    F_2 : numpy.ndarray, size=(m,n), dtype=integer
        second coordinate index  of the Fourier spectrum in a map system.

    Notes
    -----
        .. code-block:: text

          metric system:         Fourier-based flip
                 y               ┼------><------┼
                 ^               |              |
                 |               |              |
                 |               v              v
          <------┼-------> x
                 |               ^              ^
                 |               |              |
                 v               ┼------><------┼

    It is important to know what type of coordinate systems exist, hence:

        .. code-block:: text

          coordinate |           coordinate  ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       j               |       x
             --------┼-------->      --------┼-------->
                     |                       |
                     |                       |
                     | i                     |
                     v                       |
   """
    assert type(Q) in (np.ma.core.MaskedArray, np.ndarray), \
        "please provide an array"
    m, n = Q.shape
    if indexing == 'ij':
        if axis in ('center', ):
            (I_grd, J_grd) = np.meshgrid(np.arange(0, m),
                                         np.arange(0, n),
                                         indexing='ij')
        else:
            (I_grd, J_grd) = np.meshgrid(np.arange(0, m) - (m // 2),
                                         np.arange(0, n) - (n // 2),
                                         indexing='ij')
        F_1, F_2 = I_grd / m, J_grd / n
    else:
        fy, fx = np.linspace(0, m, m), np.linspace(0, n, n)
        if axis in ('center', ):
            fy, fx = fy - (m // 2), fx - (n // 2)
        fy, fx = np.flip(fy) / m, fx / n

        F_1 = np.repeat(fx[np.newaxis, :], m, axis=0)
        F_2 = np.repeat(fy[:, np.newaxis], n, axis=1)

    if system == 'radians':  # what is the range of the axis
        F_1 *= 2 * np.pi
        F_2 *= 2 * np.pi
    elif system == 'pixel':
        F_1 *= n
        F_1 *= m
    elif system == 'unit':
        F_1 *= 2
        F_2 *= 2
    if shift:
        F_1, F_2 = np.fft.fftshift(F_1), np.fft.fftshift(F_2)
    return F_1, F_2


def construct_phase_plane(Z, di, dj, indexing='ij'):
    """given a displacement, create what its phase plane in Fourier space

    Parameters
    ----------
    Z : np.array, size=(m,n)
        image domain
    di : float
        displacement along the vertical axis
    dj : float
        displacement along the horizantal axis
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates

    Returns
    -------
    Q : np.array, size=(m,n), complex
        array with phase angles

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
    (m, n) = Z.shape

    I_grd, J_grd = np.meshgrid(np.arange(0, n) - (n // 2),
                               np.arange(0, m) - (m // 2),
                               indexing=indexing)
    I_grd, J_grd = I_grd / m, J_grd / n

    Q_unwrap = ((I_grd * di) + (J_grd * dj)) * (2 * np.pi)  # in radians
    Q = np.cos(-Q_unwrap) + 1j * np.sin(-Q_unwrap)

    Q = np.fft.fftshift(Q)
    return Q


def cross_spectrum_to_coordinate_list(data, W=np.array([])):
    """ if data is given in array for, then transform it to a coordinate list

    Parameters
    ----------
    data : np.array, size=(m,n), dtype=complex
        cross-spectrum
    W : np.array, size=(m,n), dtype=boolean
        weigthing matrix of the cross-spectrum

    Returns
    -------
    data_list : np.array, size=(m*n,3), dtype=float
        coordinate list with angles, in normalized ranges, i.e: -1 ... +1
    """
    assert isinstance(data, np.ndarray), "please provide an array"
    assert isinstance(W, np.ndarray), "please provide an array"

    if data.shape[0] == data.shape[1]:
        (m, n) = data.shape
        F1, F2 = make_fourier_grid(np.zeros((m, n)),
                                   indexing='ij',
                                   system='unit')

        # transform from complex to -1...+1
        Q = np.fft.fftshift(np.angle(data) / np.pi)  # (2*np.pi))

        data_list = np.vstack((F1.flatten(), F2.flatten(), Q.flatten())).T
        if W.size > 0:  # remove masked data
            data_list = data_list[W.flatten() == 1, :]
    elif W.size != 0:
        data_list = data[W.flatten() == 1, :]
    else:
        data_list = data
    return data_list


def construct_phase_values(IJ,
                           di,
                           dj,
                           indexing='ij',
                           system='radians'):  # todo implement indexing
    """given a displacement, create what its phase plane in Fourier space

    Parameters
    ----------
    IJ : np.array, size=(_,2), dtype=float
        locations of phase values
    di : float
        displacment along the vertical axis
    dj : float
        displacment along the horizantal axis
    indexing : {‘xy’, ‘ij’}
         * "xy" : using map coordinates
         * "ij" : using local image  coordinates
    indexing : {‘radians’ (default), ‘unit’, 'normalized'}
        the extent of the cross-spectrum can span different ranges

         * "radians" : -pi..+pi
         * "unit" : -1...+1
         * "normalized" : -0.5...+0.5

    Returns
    -------
    Q : np.array, size=(_,1), complex
        array with phase angles
    """

    if system == 'radians':  # -pi ... +pi
        scaling = 1
    elif system == 'unit':  # -1 ... +1
        scaling = np.pi
    else:  # normalized -0.5 ... +0.5
        scaling = 2 * np.pi

    Q_unwrap = ((IJ[:, 0] * di) + (IJ[:, 1] * dj)) * scaling
    Q = np.cos(-Q_unwrap) + 1j * np.sin(-Q_unwrap)
    return Q


def gradient_fourier(Z):
    """ spectral derivative estimation

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype=float
        intensity array

    Returns
    -------
    dZdx_1, dZdx_2 : numpy.ndarray, size=(m,n), dtype=float
        first order derivative along both axis
    """
    m, n = Z.shape[0:2]

    x_1 = np.linspace(0, 2 * np.pi, m, endpoint=False)
    dx = x_1[1] - x_1[0]

    k_1, k_2 = 2 * np.pi * np.fft.fftfreq(m, dx), 2 * np.pi * np.fft.fftfreq(
        n, dx)
    K_1, K_2 = np.meshgrid(k_1, k_2, indexing='ij')

    # make sure Gibbs phenomena are reduced
    I_g = perdecomp(Z)[0]
    W_1, W_2 = hanning_window(K_1), hanning_window(K_2)

    dZdx_1 = np.fft.ifftn(W_1 * K_1 * 1j * np.fft.fftn(Z)).real
    dZdx_2 = np.fft.ifftn(W_2 * K_2 * 1j * np.fft.fftn(Z)).real
    return dZdx_1, dZdx_2


# frequency matching filters
def raised_cosine(Z, beta=0.35):
    """ raised cosine filter, based on [St01]_ and used by [Le07]_.
    
    Parameters
    ----------    
    Z : numpy.array, size=(m,n)
        array with intensities
    beta : float, default=0.35
        roll-off factor
    
    Returns
    -------
    W : numpy.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    tpss   

    References
    ----------    
    .. [St01] Stone et al. "A fast direct Fourier-based algorithm for subpixel
              registration of images." IEEE Transactions on geoscience and
              remote sensing. vol. 39(10) pp. 2235-2243, 2001.
    .. [Le07] Leprince, et.al. "Automatic and precise orthorectification,
              coregistration, and subpixel correlation of satellite images,
              application to ground deformation measurements", IEEE 
              Transactions on geoscience and remote sensing vol. 45.6 pp. 
              1529-1558, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)

    >>> rc1 = raised_cosine(spec1, beta=0.35)
    >>> rc2 = raised_cosine(spec2, beta=0.50)

    >>> Q = (rc1*spec1) * np.conjugate((rc2*spec2)) # Fourier based image matching
    >>> Qn = normalize_spectrum(Q)    
    """  # noqa: E501
    assert isinstance(Z, np.ndarray), "please provide an array"
    (m, n) = Z.shape

    Fx, Fy = make_fourier_grid(Z, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy)  # radius
    # filter formulation
    Hamm = np.cos((np.pi / (2 * beta)) * (R - (.5 - beta)))**2
    selec = np.logical_and((.5 - beta) <= R, R <= .5)

    # compose filter
    W = np.zeros((m, n))
    W[(.5 - beta) > R] = 1
    W[selec] = Hamm[selec]
    return W


def hamming_window(Z):
    """ create two-dimensional Hamming filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, blackman_window,
    hamming_window
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    m, n = Z.shape
    W = np.sqrt(np.outer(np.hamming(m), np.hamming(n)))
    W = np.fft.fftshift(W)
    return W


def hanning_window(Z):
    """ create two-dimensional Hanning filter, also known as Cosine Bell

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, blackman_window,
    hamming_window
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    m, n = Z.shape
    W = np.sqrt(np.outer(np.hanning(m), np.hanning(n)))
    W = np.fft.fftshift(W)
    return W


def blackman_window(Z):
    """ create two-dimensional Blackman filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, hamming_window,
    hanning_window
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    m, n = Z.shape
    W = np.sqrt(np.outer(np.blackman(m), np.blackman(n)))
    W = np.fft.fftshift(W)
    return W


def kaiser_window(Z, beta=14.):
    """ create two dimensional Kaiser filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    beta: float
        0.0 - rectangular window
        5.0 - similar to Hamming window
        6.0 - similar to Hanning window
        8.6 - similar to Blackman window

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, hamming_window,
    hanning_window
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    m, n = Z.shape
    W = np.sqrt(np.outer(np.kaiser(m, beta), np.kaiser(n, beta)))
    W = np.fft.fftshift(W)
    return W


def low_pass_rectancle(Z, r=0.50):
    """ create hard two dimensional low-pass filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the rectangle, r=.5 is same as its width

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    low_pass_circle, low_pass_pyramid, low_pass_bell

    References
    ----------
    .. [Ta03] Takita et al. "High-accuracy subpixel image registration based on
              phase-only correlation" IEICE transactions on fundamentals of
              electronics, communications and computer sciences, vol.86(8)
              pp.1925-1934, 2003.
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    Fx, Fy = make_fourier_grid(Z, indexing='xy', system='normalized')

    # filter formulation
    W = np.logical_and(np.abs(Fx) <= r, np.abs(Fy) <= r)
    return W


def low_pass_pyramid(Z, r=0.50):
    """ create low-pass two-dimensional filter with pyramid shape, see also
    [Ta03]_.

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_bell

    References
    ----------
    .. [Ta03] Takita et al. "High-accuracy subpixel image registration based on
              phase-only correlation" IEICE transactions on fundamentals of
              electronics, communications and computer sciences, vol.86(8)
              pp.1925-1934, 2003.
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    R = low_pass_rectancle(Z, r)
    W = signal.convolve2d(R.astype(float),
                          R.astype(float),
                          mode='same',
                          boundary='wrap')
    W = np.fft.fftshift(W / np.max(W))
    return W


def low_pass_bell(Z, r=0.50):
    """ create low-pass two-dimensional filter with a bell shape, see also
    [Ta03]_.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width

    Returns
    -------
    W : numpy.ndarray, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_pyramid

    References
    ----------
    .. [Ta03] Takita et al. "High-accuracy subpixel image registration based on
              phase-only correlation" IEICE transactions on fundamentals of
              electronics, communications and computer sciences, vol.86(8)
              pp.1925-1934, 2003.
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    R1 = low_pass_rectancle(Z, r)
    R2 = low_pass_pyramid(Z, r)
    W = signal.convolve2d(R1.astype(float),
                          R2.astype(float),
                          mode='same',
                          boundary='wrap')
    W = np.fft.fftshift(W / np.max(W))
    return W


def low_pass_circle(Z, r=0.50):
    """ create hard two-dimensional low-pass filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle
    """
    assert type(Z) in (np.ma.core.MaskedArray, np.ndarray), \
        "please provide an array"
    if type(Z) in (np.ma.core.MaskedArray, ):
        Z = np.ma.getdata(Z)

    Fx, Fy = make_fourier_grid(Z,
                               indexing='xy',
                               system='normalized',
                               shift=False)
    R = np.hypot(Fx, Fy)  # radius
    # filter formulation
    W = R <= r
    return W


def low_pass_ellipse(Z, r1=0.50, r2=0.50):
    """ create hard low-pass filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    r1 : float, default=0.5
        radius of the ellipse along the first axis, r=.5 is same as its width
    r2 : float, default=0.5
        radius of the ellipse along the second axis

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, low_pass_circle
    """
    sc = r1 / r2
    assert isinstance(Z, np.ndarray), "please provide an array"
    Fx, Fy = make_fourier_grid(Z, indexing='xy', system='normalized')
    R = np.hypot(sc * Fx, Fy)  # radius
    # filter formulation
    W = R <= r1
    return W


def high_pass_circle(Z, r=0.50):
    """ create hard high-pass filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask

    See Also
    --------
    raised_cosine, cosine_bell, low_pass_circle
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    Fx, Fy = make_fourier_grid(Z, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy)  # radius
    # filter formulation
    W = R >= r
    return W


def cosine_bell(Z):
    """ cosine bell filter

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with intensities

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=float
        weighting mask

    See Also
    --------
    raised_cosine
    """
    assert isinstance(Z, np.ndarray), "please provide an array"
    Fx, Fy = make_fourier_grid(Z, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy)  # radius

    # filter formulation
    W = .5 * np.cos(2 * R * np.pi) + .5
    W[R > .5] = 0
    return W


def cross_shading_filter(Q, az_1, az_2):  # todo
    """

    Parameters
    ----------
    Q : numpy.array, size=(m,n)
        cross-power spectrum
    az_1 : float, unit=degrees, range=-180...+180
        illumination direction of template
    az_2 : float, unit=degrees, range=-180...+180
        illumination direction of search space

    Returns
    -------
    W : numpy.array, size=(m,n), dtype=float
        weigthing grid

    References
    ----------
    .. [Wa15] Wan, "Phase correlation-based illumination-insensitive image
              matching for terrain-related applications" PhD dissertation at
              Imperical College London, 2015.
    """
    F_1, F_2 = make_fourier_grid(Q, indexing='xy', system='radians')
    θ = np.angle(F_2 + 1j * F_1)
    az_1, az_2 = np.deg2rad(az_1), np.deg2rad(az_2)

    def _get_angular_sel(θ, az_1, az_2):
        c_θ, s_θ = np.cos(θ), np.sin(θ)
        c_down = np.minimum(np.cos(az_1), np.cos(az_2))
        s_down = np.minimum(np.sin(az_1), np.sin(az_2))
        c_up = np.maximum(np.cos(az_1), np.cos(az_2))
        s_up = np.maximum(np.sin(az_1), np.sin(az_2))
        OUT = np.logical_and(
            np.logical_and(np.less(c_down, c_θ), np.less(c_θ, c_up)),
            np.logical_and(np.less(s_down, s_θ),
                           np.less(s_θ, s_up)))  # (4.42) in [1], pp.51
        return OUT

    W = np.ones_like(Q, dtype=float)
    down, up = az_2 - (3 * np.pi / 2), az_1 - (np.pi / 2)
    OUT1 = _get_angular_sel(θ, down, up)
    np.putmask(W, OUT1, 0)

    down, up = az_2 - (np.pi / 2), az_1 + (np.pi / 2)
    OUT2 = _get_angular_sel(θ, down, up)
    np.putmask(W, OUT2, 0)
    return W


# cross-spectral and frequency signal metrics for filtering
def thresh_masking(S, m=1e-4, s=10):
    """
    Mask significant intensities in spectrum, following [St01]_ and [Le07]_.

    Parameters
    ----------
    S : numpy.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
    m : float, default=1e-3
        cut-off intensity in respect to maximum
    s : integer, default=10
        kernel size of the median filter

    Returns
    -------
    M : numpy.array, size=(m,n), dtype=bool
        frequency mask

    See Also
    --------
    tpss

    References
    ----------
    .. [St01] Stone et al. "A fast direct Fourier-based algorithm for subpixel
              registration of images." IEEE Transactions on geoscience and
              remote sensing vol. 39(10) pp. 2235-2243, 2001.
    .. [Le07] Leprince, et.al. "Automatic and precise orthorectification,
              coregistration, and subpixel correlation of satellite images,
              application to ground deformation measurements", IEEE
              Transactions on geoscience and remote sensing vol. 45.6 pp.
              1529-1558, 2007.
    """
    assert isinstance(S, np.ndarray), "please provide an array"
    S_bar = np.abs(S)
    th = np.max(S_bar) * m

    # compose filter
    M = S_bar > th
    if s > 0:
        M = ndimage.median_filter(M, size=(s, s))
    return M


def perc_masking(S, m=.95, s=10):
    assert isinstance(S, np.ndarray), "please provide an array"
    S_bar = np.abs(S)
    th = np.percentile(S_bar, m * 100)

    # compose filter
    M = S_bar > th
    if s > 0:
        M = ndimage.median_filter(M, size=(s, s))
    return M


def coherence_masking(S, m=.7, s=0):
    M = local_coherence(normalize_power_spectrum(S)) > m
    if s > 0:
        M = ndimage.median_filter(M, size=(s, s))
    return M


def adaptive_masking(S, m=.9):
    """ mark significant intensities in spectrum, following [Le07]_.

    Parameters
    ----------
    S : numpy.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
    m : float, default=.9
        cut-off intensity in respect to maximum

    Returns
    -------
    M : numpy.array, size=(m,n), dtype=bool
        frequency mask

    See Also
    --------
    tpss

    References
    ----------
    .. [Le07] Leprince, et.al. "Automatic and precise orthorectification,
              coregistration, and subpixel correlation of satellite images,
              application to ground deformation measurements", IEEE
              Transactions on geoscience and remote sensing vol. 45.6 pp.
              1529-1558, 2007.
    """
    assert isinstance(S, np.ndarray), "please provide an array"
    np.seterr(divide='ignore')
    LS = np.log10(np.abs(S))
    LS[np.isinf(LS)] = np.nan
    np.seterr(divide='warn')

    NLS = LS - np.nanmax(LS.flatten())
    mean_NLS = m * np.nanmean(NLS.flatten())

    M = NLS > mean_NLS
    return M


def gaussian_mask(S):
    """ mask significant intensities in spectrum, following [Ec08]_.

    Parameters
    ----------
    S : numpy.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)

    Returns
    -------
    M : numpy.array, size=(m,n), dtype=bool
        frequency mask

    See Also
    --------
    tpss

    References
    ----------
    .. [Ec08] Eckstein et al. "Phase correlation processing for DPIV
              measurements", Experiments in fluids, vol.45 pp.485-500, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    >>> Q = spec1 * np.conjugate(spec2) # Fourier based image matching
    >>> Qn = normalize_spectrum(Q)

    >>> W = gaussian_mask(Q)
    >>> C = np.fft.ifft2(W*Q)
    """
    assert isinstance(S, np.ndarray), "please provide an array"
    (m, n) = S.shape
    Fx, Fy = make_fourier_grid(S, indexing='xy', system='normalized')

    M = np.exp(-.5 * ((Fy * np.pi) / m)**2) * np.exp(-.5 *
                                                     ((Fx * np.pi) / n)**2)
    return M
