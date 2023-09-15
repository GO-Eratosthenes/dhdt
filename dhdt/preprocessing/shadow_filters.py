# generic libraries
import numpy as np
from numpy.lib.stride_tricks import as_strided

# image processing libraries
from scipy import ndimage, signal
from skimage.filters import threshold_otsu
from sklearn.cluster import MeanShift, estimate_bandwidth

from ..generic.filtering_statistical import make_2D_Gaussian
from ..generic.handler_im import rotated_sobel, diff_compass

from ..processing.matching_tools_frequency_filters import perdecomp


def enhance_shadows(Shw, method, **kwargs):
    """ Given a specific method, employ shadow transform

    Parameters
    ----------
    Shw : np.array, size=(m,n), dtype={float,integer}
        array with intensities of shading and shadowing
    method : {‘mean’,’kuwahara’,’median’,’otsu’,'anistropic'}
        method name to be implemented,can be one of the following:

            * 'mean' : mean shift filter
            * 'kuwahara' : kuwahara filter
            * 'median' : iterative median filter
            * 'anistropic' : anistropic diffusion filter

    Returns
    -------
    M : np.array, size=(m,n), dtype={float,integer}
        shadow enhanced image, done through the given method

    See Also
    --------
    mean_shift_filter, kuwahara_filter, iterative_median_filter,
    anistropic_diffusion_scalar
    """
    if method in ('mean', 'mean-shift'):
        quantile = 0.1 if kwargs.get('quantile') is None else kwargs.get(
            'quantile')
        M = mean_shift_filter(Shw, quantile=quantile)
    elif method in ('kuwahara'):
        tsize = 5 if kwargs.get('tsize') is None else kwargs.get('tsize')
        M = kuwahara_filter(Shw, tsize=tsize)
    elif method in ('median'):
        tsize = 5 if kwargs.get('tsize') is None else kwargs.get('tsize')
        iter = 50 if kwargs.get('loop') is None else kwargs.get('loop')
        M = iterative_median_filter(Shw, tsize=tsize, loop=iter)
    elif method in ('anistropic', 'anistropic-diffusion'):
        iter = 10 if kwargs.get('iter') is None else kwargs.get('iter')
        K = .15 if kwargs.get('K') is None else kwargs.get('K')
        s = .25 if kwargs.get('s') is None else kwargs.get('s')
        n = 4 if kwargs.get('n') is None else kwargs.get('n')
        M = anistropic_diffusion_scalar(Shw, iter=iter, K=K, s=s, n=n)
    elif method in ('L0', 'L0smoothing'):
        lamb = 2E-2 if kwargs.get('lamb') is None else kwargs.get('lamb')
        kappa = 2. if kwargs.get('kappa') is None else kwargs.get('kappa')
        M = L0_smoothing(Shw, lamb=lamb, kappa=kappa)
    else:
        assert 1 == 2, 'please provide a correct method'
    return M


# functions to spatially enhance the shadow image
def mean_shift_filter(Z, quantile=0.1):
    """ Transform intensity to more clustered intensity, through mean-shift

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype=float
        array with intensity values
    quantile : float, range=0...1

    Returns
    -------
    labels : np.array, size=(m,n), dtype=integer
        array with numbered labels
    """
    bw = estimate_bandwidth(Z, quantile=quantile, n_samples=Z.shape[1])
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(Z.reshape(-1, 1))

    labels = np.reshape(ms.labels_, Z.shape)
    return labels


def kuwahara(buffer):
    d = int(np.sqrt(len(buffer)))  # assuming square kernel
    d_sub = int(((d - 1) // 2) + 1)

    buffer = np.reshape(buffer, (d, d))
    r_a = buffer[:-d_sub + 1, :-d_sub + 1]
    r_b = buffer[d_sub - 1:, :-d_sub + 1]
    r_c = buffer[:-d_sub + 1, d_sub - 1:]
    r_d = buffer[d_sub - 1:, d_sub - 1:]

    var_abcd = np.array([np.var(r_a), np.var(r_b), np.var(r_c), np.var(r_d)])
    bar_abcd = np.array(
        [np.mean(r_a), np.mean(r_b),
         np.mean(r_c), np.mean(r_d)])
    idx_abcd = np.argmin(var_abcd)
    return bar_abcd[idx_abcd]


def kuwahara_strided(buffer):  # todo: gives error, but it might run faster...?
    d = int(np.sqrt(len(buffer)))  # assuming square kernel
    if d == 5:
        strides = (10, 2, 5, 1)
        shape = (2, 2, 3, 3)
    elif d == 3:
        strides = (3, 1, 3, 1)
        shape = (2, 2, 2, 2)
    abcd = as_strided(np.reshape(buffer, (d, d)), shape=shape, strides=strides)
    var_abcd = np.var(abcd, (2, 3))
    bar_abcd = np.mean(abcd, (2, 3))
    idx_abcd = np.argmin(var_abcd)
    return bar_abcd[idx_abcd]


def kuwahara_filter(Z, tsize=5):
    """ Transform intensity to more clustered intensity

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype=float
        array with intensity values
    tsize : integer, {x ∈ ℕ | x ≥ 1}
        dimension of the kernel

    Returns
    -------
    Z_new : np.array, size=(m,n), dtype=float

    Notes
    -----
    The template is subdivided into four blocks, where the sampled mean and
    standard deviation are calculated. Then mean of the block with the lowest
    variance is assigned to the central pixel. The configuration of the blocks
    look like:

        .. code-block:: text

          ┌-----┬-┬-----┐ ^
          |  A  | |  B  | |
          ├-----┼-┼-----┤ | tsize
          ├-----┼-┼-----┤ |
          |  C  | |  D  | |
          └-----┴-┴-----┘ v

    See Also
    --------
    iterative_median_filter
    """
    assert np.remainder(tsize, 1) == 0, ('please provide square kernel')
    assert np.remainder(tsize - 1, 2) == 0, ('kernel dimension should be odd')
    assert tsize >= 3, ('kernel should be big enough')

    # if (tsize==3) or (tsize==5): # todo
    #    Z_new = ndimage.generic_filter(
    #        I, kuwahara_strided, size=(tsize,tsize)
    #    )
    # else:
    Z_new = ndimage.generic_filter(Z, kuwahara, size=(tsize, tsize))
    return Z_new


def iterative_median_filter(Z, tsize=5, loop=50):
    """ Transform intensity to more clustered intensity, through iterative
    filtering with a median operation

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype=float
        array with intensity values
    tsize : integer, {x ∈ ℕ | x ≥ 1}
        dimension of the kernel
    loop : integer, {x ∈ ℕ | x ≥ 0}
        amount of iterations

    Returns
    -------
    I_new : np.array, size=(m,n), dtype=float
        array with stark edges

    See Also
    --------
    kuwahara_filter
    """
    for i in range(loop):
        Z = ndimage.median_filter(Z, size=tsize)
    return Z


def selective_blur_func(C, t_size):
    # decompose arrays, these seem to alternate....
    C_col = C.reshape(t_size[0], t_size[1], 2)
    if np.any(np.sign(C[::2]) == -1):
        M, Z = -1 * C_col[:, :, 0].flatten(), C_col[:, :, 1].flatten()
    else:  # redundant calculation, just ignore and exit
        return 0

    m_central = M[(t_size[0] * t_size[1]) // 2]
    i_central = Z[(t_size[0] * t_size[1]) // 2]
    if m_central != 1:
        return i_central

    W = make_2D_Gaussian(t_size, fwhm=3).flatten()
    W /= np.sum(W)
    new_intensity = np.sum(W * Z)
    return new_intensity


def fade_shadow_cast(Shw, az, t_size=9):
    """

    Parameters
    ----------
    Shw : np.array, size=(m,n)
        image with mask of shadows
    az : float, unit=degrees
        illumination orientation
    t_size : integer, {x ∈ ℕ | x ≥ 1}
        buffer size of the Gaussian blur

    Returns
    -------
    Shw : np.array, size=(m,n), dtype=float, range=0...1
        image with faded shadows on the occluding end
    """
    kernel_az = rotated_sobel(az, size=5, indexing='xy')

    new_dims = []
    for original_length, new_length in zip(kernel_az.shape, (t_size, t_size)):
        new_dims.append(np.linspace(0, original_length - 1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    kernel_az = ndimage.map_coordinates(kernel_az, coords)

    W_f = ndimage.convolve(Shw.astype(np.float64), kernel_az)
    M_f = np.logical_and(W_f > 0.001, ~(W_f < 0.001))  # cast ridges

    Combo = np.dstack((Shw, -1. * (M_f.astype(np.float64))))

    S_b = ndimage.generic_filter(Combo,
                                 selective_blur_func,
                                 footprint=np.ones((t_size, t_size, 2)),
                                 mode='mirror',
                                 cval=np.nan,
                                 extra_keywords={'t_size': (t_size, t_size)})
    S_b = S_b[..., 0]
    Shf = np.copy(Shw).astype(np.float64)
    Shf[M_f] = S_b[M_f]
    return Shf


def diffusion_strength_1(Z, K):
    """ first diffusion function, proposed by [PM87]_, when a complex array is
    provided the absolute magnitude is used, following [Ge92]_

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype={float,complex}, ndim={2,3}
        array with intensities
    K : float
        parameter based upon the noise level

    Returns
    -------
    g : np.array, size=(m,n), dtype=float
        array with diffusion parameters

    References
    ----------
    .. [PM87] Perona & Malik "Scale space and edge detection using anisotropic
              diffusion" Proceedings of the IEEE workshop on computer vision,
              pp.16-22, 1987
    .. [Ge92] Gerig et al. "Nonlinear anisotropic filtering of MRI data" IEEE
              transactions on medical imaging, vol.11(2), pp.221-232, 1992
    """
    # admin
    if np.iscomplexobj(Z):  # support complex input
        Z_abs = np.abs(Z)
        g_1 = np.exp(-1 * np.divide(np.abs(Z), K)**2)
    elif Z.ndim == 3:  # support multispectral input
        I_sum = np.sum(Z**2, axis=2)
        Z_abs = np.sqrt(I_sum, out=np.zeros_like(I_sum), where=I_sum != 0)
    else:
        Z_abs = Z

    # calculation
    g_1 = np.exp(-1 * np.divide(Z_abs, K)**2)
    return g_1


def diffusion_strength_2(Z, K):
    """ second diffusion function, proposed by [1], when a complex array is
    provided the absolute magnitude is used, following [2]

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype={float,complex}
        array with intensities
    K : float
        parameter based upon the noise level

    Returns
    -------
    g : np.array, size=(m,n), dtype=float
        array with diffusion parameters

    References
    ----------
    .. [PM87] Perona & Malik "Scale space and edge detection using anisotropic
              diffusion" Proceedings of the IEEE workshop on computer vision,
              pp.16-22, 1987
    .. [Ge92] Gerig et al. "Nonlinear anisotropic filtering of MRI data" IEEE
              transactions on medical imaging, vol.11(2), pp.221-232, 1992
    """
    if np.iscomplexobj(Z):
        Z_abs = np.abs(Z)
        denom = (1 + np.divide(np.abs(Z), K)**2)
    elif Z.ndim == 3:  # support multispectral input
        I_sum = np.sum(Z**2, axis=2)
        Z_abs = np.sqrt(I_sum, out=np.zeros_like(I_sum), where=I_sum != 0)
    else:
        Z_abs = Z
    # calculation
    denom = (1 + np.divide(Z_abs, K)**2)
    g_2 = np.divide(1, denom, where=denom != 0)
    return g_2


def anistropic_diffusion_scalar(Z, iter=10, K=.15, s=.25, n=4):
    """ non-linear anistropic diffusion filter of a scalar field

    Parameters
    ----------
    Z : np.array, size=(m,n), dtype={float,complex}, ndim={2,3}
        intensity array
    iter : integer, {x ∈ ℕ | x ≥ 0}
        amount of iterations
    K : float, default=.15
        parameter based upon the noise level
    s : float, default=.25
        parameter for the integration constant
    n : {4,8}
        amount of neighbors used

    Returns
    -------
    I : np.array, size=(m,n)

    References
    ----------
    .. [PM87] Perona & Malik "Scale space and edge detection using anisotropic
              diffusion" Proceedings of the IEEE workshop on computer vision,
              pp.16-22, 1987
    .. [Ge92] Gerig et al. "Nonlinear anisotropic filtering of MRI data" IEEE
              transactions on medical imaging, vol.11(2), pp.221-232, 1992
    """
    # admin
    Z_new = np.copy(Z)
    multispec = True if Z.ndim == 3 else False
    if multispec:
        b = Z.shape[2]
    if n != 4:
        delta_d = np.sqrt(2)
    s = np.minimum(s, 1 / n)  # see Appendix A in [2]

    # processing
    for i in range(iter):
        Z_u = diff_compass(Z_new, direction='n')
        Z_d = diff_compass(Z_new, direction='s')
        Z_r = diff_compass(Z_new, direction='e')
        Z_l = diff_compass(Z_new, direction='w')
        if not multispec:
            Z_update = (diffusion_strength_1(Z_u, K) * Z_u +
                        diffusion_strength_1(Z_d, K) * Z_d +
                        diffusion_strength_1(Z_r, K) * Z_r +
                        diffusion_strength_1(Z_l, K) * Z_l)
        else:
            Z_update = (np.tile(np.atleast_3d(diffusion_strength_1(Z_u, K)),
                                (1, 1, b)) * Z_u +
                        np.tile(np.atleast_3d(diffusion_strength_1(Z_r, K)),
                                (1, 1, b)) * Z_r +
                        np.tile(np.atleast_3d(diffusion_strength_1(Z_d, K)),
                                (1, 1, b)) * Z_d +
                        np.tile(np.atleast_3d(diffusion_strength_1(Z_l, K)),
                                (1, 1, b)) * Z_l)
        del Z_u, Z_d, Z_r, Z_l

        if n > 4:
            # alternate with the cross-domain, following the approach of [2]
            Z_ur = diff_compass(Z_new, direction='n') / delta_d
            Z_lr = diff_compass(Z_new, direction='s') / delta_d
            Z_ll = diff_compass(Z_new, direction='e') / delta_d
            Z_ul = diff_compass(Z_new, direction='w') / delta_d

            if not multispec:
                Z_xtra = (diffusion_strength_1(Z_ur, K) * Z_ur +
                          diffusion_strength_1(Z_lr, K) * Z_lr +
                          diffusion_strength_1(Z_ll, K) * Z_ll +
                          diffusion_strength_1(Z_ul, K) * Z_ul)
            else:  # extent the diffusion parameter to all bands
                Z_xtra = (np.tile(np.atleast_3d(diffusion_strength_1(Z_ur, K)),
                                  (1, 1, b)) * Z_ur +
                          np.tile(np.atleast_3d(diffusion_strength_1(Z_lr, K)),
                                  (1, 1, b)) * Z_lr +
                          np.tile(np.atleast_3d(diffusion_strength_1(Z_ll, K)),
                                  (1, 1, b)) * Z_ll +
                          np.tile(np.atleast_3d(diffusion_strength_1(Z_ul, K)),
                                  (1, 1, b)) * Z_ul)
            Z_update += Z_xtra
        Z_new[1:-1, 1:-1] += s * Z_update
    return Z_new


def psf2otf(psf, dim):
    m, n = psf.shape[0], psf.shape[1]

    new_psf = np.zeros(dim)
    new_psf[:m, :n] = psf[:, :]

    new_psf = np.roll(new_psf, -dim[0] // 2, axis=0)
    new_psf = np.roll(new_psf, -dim[1] // 2, axis=1)
    otf = np.fft.fft2(new_psf)
    return otf


def L0_smoothing(Z, lamb=2E-2, kappa=2., beta_max=1E5):
    """

    Parameters
    ----------
    Z : numpy.ndarray, size={(m,n), (m,n,b)}, dtype=float
        grid with intensity values
    lamb : float, default=.002
    kappa : float, default=2.
    beta_max : float, default=10000

    Returns
    -------
    Z : numpy.ndarray, size={(m,n), (m,n,b)}, dtype=float
        modified intensity image

    References
    ----------
    .. [Xu11] Xu et al. "Image smoothing via L0 gradient minimization" ACM
              transactions on graphics, 2011.
    """
    m, n = Z.shape[:2]
    b = 1 if Z.ndim == 2 else Z.shape[2]

    dx, dy = np.array([[+1, -1]]), np.array([[+1], [-1]])
    dx_F, dy_F = psf2otf(dx, (m, n)), psf2otf(dy, (m, n))

    Z = perdecomp(Z)[0]
    N_1 = np.fft.fft2(Z)
    D_2 = np.abs(dx_F)**2 + np.abs(dy_F)**2

    if b > 1:
        dx, dy = np.tile(np.atleast_3d(dx), (1, 1, b)), \
                 np.tile(np.atleast_3d(dy), (1, 1, b))
        D_2 = np.tile(np.atleast_3d(D_2), (1, 1, b))

    beta = 2 * lamb
    while beta < beta_max:
        D_1 = 1 + beta * D_2

        h = np.roll(signal.fftconvolve(Z, dx, mode='same'), -1, axis=1)
        v = np.roll(signal.fftconvolve(Z, dy, mode='same'), -1, axis=0)

        t = (h**2 + v**2) < (lamb / beta)
        np.putmask(h, t, 0)
        np.putmask(v, t, 0)

        if b == 1:
            N_2 = np.concatenate(
                (np.array(h[:, -1] - h[:, 0], ndmin=2).T, -np.diff(h, 1, 1)),
                axis=1)
            N_2 += np.concatenate(
                (np.array(v[-1, ...] - v[0, ...], ndmin=2), -np.diff(v, 1, 0)),
                axis=0)
        else:
            N_2 = np.concatenate(
                (np.array(h[:, -1, :] - h[:, 0, :], ndmin=3).transpose(
                    (1, 0, 2)), -np.diff(h, 1, 1)),
                axis=1)
            N_2 += np.concatenate(
                (np.array(v[-1, ...] - v[0, ...], ndmin=3), -np.diff(v, 1, 0)),
                axis=0)
            N_2 = np.tile(np.atleast_3d(np.sum(N_2, axis=2)), (1, 1, b))
        I_F = np.divide(N_1 + beta * np.fft.fft2(N_2), D_1)
        Z = np.real(np.fft.ifft2(I_F))
        beta *= kappa
    return Z
