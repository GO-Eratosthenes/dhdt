# general libraries
import warnings
import numpy as np

# image processing libraries
from scipy import ndimage, interpolate

from ..generic.unit_check import are_two_arrays_equal
from ..preprocessing.image_transforms import mat_to_gray, histogram_equalization
from ..processing.matching_tools import get_peak_indices
from ..generic.filtering_statistical import make_2D_Gaussian
from ..generic.handler_im import get_grad_filters, \
    nan_resistant_conv2, nan_resistant_diff2

# spatial sub-pixel allignment functions
def create_differential_data(I1,I2):
    """ estimate the spatial and temporal first derivatives of two arrays

    Parameters
    ----------
    I1 : numpy.ndarray, size=(m,n), type=float
        array with image intensities
    I2 : numpy.ndarray, size=(m,n), type=float
        array with image intensities

    Returns
    -------
    I_di : numpy.ndarray, size=(m,n), type=float
        vertical gradient of first image
    I_dj : numpy.ndarray, size=(m,n), type=float
        horizontal gradient of first image
    I_dt : numpy.ndarray, size=(m,n), type=float
        temporal gradient between first and second image

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
    assert type(I1) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    are_two_arrays_equal(I1, I2)

    # admin
    kernel_t = np.array(
        [[1., 1., 1.],
         [1., 2., 1.],
         [1., 1., 1.]]
    ) / 10
    # smooth to not have very sharp derivatives
    kernel_g = make_2D_Gaussian((3, 3), fwhm=3)
    kernel_x, kernel_y = get_grad_filters('kroon')
    # since this function works in pixel space, change orientation of axis
    di,dj = np.flipud(kernel_y), kernel_x

    # estimation
    if np.any(np.isnan(I1)):
        I1 = nan_resistant_conv2(I1, kernel_g, cval=0)
        I_di = nan_resistant_diff2(I1, di, cval=0) /2
        I_dj = nan_resistant_diff2(I1, dj, cval=0) /2
        I1_dt = nan_resistant_conv2(I1, -kernel_t, cval=0)
    else:
        I1 = ndimage.convolve(I1, kernel_g)
        # spatial derivative
        I_di = ndimage.convolve(I1, di) / 4
        I_dj = ndimage.convolve(I1, dj) / 4

        #I_di = ndimage.convolve(I1, di) /2 # take grid spacing into account
        #I_dj = ndimage.convolve(I1, dj) /2
        # temporal derivative
        I1_dt = ndimage.convolve(I1, -kernel_t)

    if np.any(np.isnan(I2)):
        I2 = nan_resistant_conv2(I2, kernel_g, cval=0)
        I2_dt = nan_resistant_conv2(I2, kernel_t, cval=0)
    else:
        I2 = ndimage.convolve(I2, kernel_g)
        I_di += ndimage.convolve(I2, di) / 4
        I_dj += ndimage.convolve(I2, dj) / 4

        I2_dt = ndimage.convolve(I2, kernel_t)

    I_dt = I2_dt + I1_dt
    return I_di, I_dj, I_dt

def simple_optical_flow(I1, I2, window_size, sampleI, sampleJ,
                        tau=1e-2, sigma=0.):
    r""" displacement estimation through optical flow, based on [LK81]_.

    Parameters
    ----------
    I1 : numpy.ndarray, size=(m,n)
        array with intensities
    I2 : numpy.ndarray, size=(m,n)
        array with intensities
    window_size: integer, {x ∈ ℕ | x ≥ 1}
        kernel size of the neighborhood
    sampleI: numpy.ndarray, size=(k,l)
        grid with image coordinates, its vertical coordinate in a pixel system
    sampleJ: numpy.ndarray, size=(k,l)
        grid with image coordinates, its horizontical coordinate
    sigma: float
        smoothness for gaussian image blur

    Returns
    -------
    Ugrd : numpy.ndarray, size=(k,l)
        vertical displacement estimate, in "ij"-coordinate system
    Vgrd : numpy.ndarray, size=(k,l)
        vertical displacement estimate, in "ij"-coordinate system
    Ueig : numpy.ndarray, size=(k,l)
        eigenvalue of system of equation for vertical estimate
    Veig : numpy.ndarray, size=(k,l)
        eigenvalue of system of equation for horizontal estimate

    See Also
    --------
    affine_optical_flow

    Notes
    -----
    This is a fast implementation using the direct surrounding to solve the
    following problem:

        .. math:: \frac{\partial I}{\partial t} =
        \frac{\partial I}{\partial x} \cdot \frac{\partial x}{\partial t} +
        \frac{\partial I}{\partial y} \cdot \frac{\partial y}{\partial t}

    Where :math:`\frac{\partial I}{\partial x}` are spatial derivatives, and
    :math:`\frac{\partial I}{\partial t}` the temporal change, while
    :math:`\frac{\partial x}{\partial t}` and
    :math:`\frac{\partial y}{\partial t}` are the parameters of interest.

    Two different coordinate system are used here:

        .. code-block:: text

            o____j                          y  map(x, y)
            |                               |
            |                               |
            i image(i, j)                   o_____x
          pixel coordinate frame        metric coordinate frame

    References
    ----------
    .. [LK81] Lucas & Kanade, "An iterative image registration technique with an
              application to stereo vision", Proceedings of 7th international
              joint conference on artificial intelligence, 1981.
    """
    are_two_arrays_equal(I1, I2)

    # check and initialize
    if isinstance(sampleI, int):
        sampleI = np.array([sampleI])
    if isinstance(sampleJ, int):
        sampleJ = np.array([sampleJ])

    # if data range is bigger than 1, transform
    if np.ptp(I1.flatten())>1:
        I1 = mat_to_gray(I1)
    if np.ptp(I1.flatten())>1:
        I2 = mat_to_gray(I2)

    # smooth the image, so derivatives are not so steep
    if sigma!=0:
        I1 = ndimage.gaussian_filter(I1, sigma=sigma)
        I2 = ndimage.gaussian_filter(I2, sigma=sigma)

    kernel_x = np.array(
        [[-1., 1.],
         [-1., 1.]]
    ) * .25
    kernel_y = np.array(
         [[1., 1.],
          [-1., -1.]]
     ) * .25
    kernel_t = np.array(
        [[1., 1.],
         [1., 1.]]
    ) * .25

    fx = ndimage.convolve(I1, kernel_x)
    fy = ndimage.convolve(I1, kernel_y)
    ft = ndimage.convolve(I2, kernel_t) + ndimage.convolve(I1, -kernel_t)

    # grid or single estimation
    if sampleI.ndim>1:
        assert sampleI.shape == sampleJ.shape
        Ugrd = np.zeros_like(sampleI, dtype="float")
        Vgrd = np.zeros_like(sampleI, dtype="float")
        Ueig,Veig = np.zeros_like(Ugrd), np.zeros_like(Ugrd)
    else:
        Ugrd,Vgrd,Ueig,Veig = 0,0,0,0

    # window_size should be odd
    radius = np.floor(window_size / 2).astype('int')
    for iIdx in np.arange(sampleI.size):
        iIm = sampleI.flat[iIdx]
        jIm = sampleJ.flat[iIdx]

        if sampleI.ndim>1:
            (iGrd, jGrd) = np.unravel_index(iIdx, sampleI.shape)

        # get templates
        Ix = fx[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()
        Iy = fy[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()
        It = ft[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()

        # look if variation is present
        if np.std(It) != 0:
            b = It  # get b here
            A = np.stack((Ix, Iy), axis=1)  # get A here

            # caluclate eigenvalues to see directional contrast distribution
            ɛ = np.linalg.eigvals(np.matmul(A.T, A))
            nu = np.matmul(np.linalg.pinv(A), -b)  # get velocity here

            if sampleI.ndim>1:
                Ugrd[iGrd,jGrd], Vgrd[iGrd,jGrd] = nu[0][0],nu[1][0]
                Ueig[iGrd,jGrd], Veig[iGrd,jGrd] = ɛ[0],ɛ[1]
            else:
                Ugrd, Vgrd = nu[1][0], nu[0][0]
                Ueig, Veig = ɛ[1], ɛ[0]

    return Ugrd, Vgrd, Ueig, Veig

def affine_optical_flow(I1, I2, model='affine', iteration=10,
                        preprocessing=None, episolar=np.array([])):
    """ displacement estimation through optical flow with an affine model,
    based on [LK81]_

    Parameters
    ----------
    I1 : numpy.ndarray, size=(m,n)
        array with intensities
    I2 : numpy.ndarray, size=(m,n)
        array with intensities
    preprocessing : {None, 'hist_equal'}
        preprocessing steps to apply to the input imagery
            * None : use the raw imagery
            * 'hist_eq' : apply histogram equalization, this is a common step
            to comply with the brightness consistency assumption in remote
            sensing imagery [Br16]_
    model : string
        several models can be used:
            * 'simple' : translation only
            * 'affine' : affine transformation and translation
            * 'similarity' : scaling, rotation and translation
    episolar : numpy.ndarray, size=(1,2)
        vector, for additional constrains [M015]_.
    iteration : integer, {x ∈ ℕ | x ≥ 0}
        number of iterations used

    Returns
    -------
    u,v : float
        displacement estimate
    A : numpy.array, size=(2,2)
        estimated mapping matrix
    snr : float
        signal to noise ratio

    Notes
    -----
    The following coordinate system is used here:

    .. code-block:: text

      indexing   |
      system 'ij'|
                 |
                 |       j
         --------+-------->
                 |
                 |
      image      | i
      based      v

    References
    ----------
    .. [LK81] Lucas & Kanade, "An iterative image registration technique with an
              application to stereo vision", Proceedings of 7th international
              joint conference on artificial intelligence, 1981.
    .. [Br16] Brigot et al. "Adaptation and evaluation of an optical flow method
              applied to coregisgtration of forest remote sensing images", IEEE
              journal of selected topics in applied remote sensing, vol.9(7)
              pp.2923-2939, 2016
    .. [M015] Mohamed et al. "Differential optical flow estimation under
              monocular epipolar line constraint", Proceedings of the
              international conference on computer vision systems, 2015.
    """
    assert isinstance(model, str), ('please provide a model; ',
                                    '{''simple'',''affine'',''similarity''}')
    assert ~np.any(np.isnan(I2)), ("arrays with missing data are not yet supported")
    are_two_arrays_equal(I1, I2)

    model = model.lower()

    kernel_i,kernel_j = get_grad_filters('kroon', tsize=3, order=1,
                                         indexing='ij')

    if model in ('simple'):
        kernel_j2,_ = get_grad_filters('kroon', tsize=3, order=2,
                                         indexing='ij')
        kernel_i2 = kernel_j2.T

    (mI,nI) = I1.shape
    mnI = mI*nI

    if preprocessing in ['hist_equal']:
        I1 = histogram_equalization(I1, I2)
    # smooth to not have very sharp derivatives
    I1 = ndimage.convolve(I1, make_2D_Gaussian((3,3), fwhm=3))

    # calculate spatial and temporal derivatives
    I_di, I_dj = ndimage.convolve(I1, kernel_i), ndimage.convolve(I1, kernel_j)
    if model in ('simple'):
        I_di2 = ndimage.convolve(I1, kernel_i2)
        I_dj2 = ndimage.convolve(I1, kernel_j2)

    W = make_2D_Gaussian((mI,nI), fwhm=np.maximum(mI,nI))

    # create local coordinate grid
    (grd_i,grd_j) = np.meshgrid(np.linspace(-(mI-1)/2, +(mI-1)/2, mI),
                                np.linspace(-(nI-1)/2, +(nI-1)/2, nI),
                                indexing='ij')
    stk_ij = np.column_stack([grd_i.flatten(), grd_j.flatten()])

    # initialize iteration
    p = np.zeros((1,6), dtype=float)
    p_stack = np.zeros((iteration,6), dtype=float)
    for i in np.arange(iteration):
        # affine transform
        Aff = np.array([[1, 0, 0], [0, 1, 0]]) + p.reshape(3,2).T
        grd_new = np.matmul(Aff,
                            np.vstack((stk_ij.T,
                                       np.ones(mnI))))
        new_i = np.reshape(grd_new[0,:], (mI, nI))
        new_j = np.reshape(grd_new[1,:], (mI, nI))

        # construct new templates
        try:
            I2_new = interpolate.griddata(stk_ij, I2.flatten().T,
                                          (new_i,new_j), method='cubic')
            I2_new = ndimage.convolve(I2_new, make_2D_Gaussian((3,3), fwhm=3))
        except:
            print('different number of values and points')

        I_dt_new = I2_new - I1

        # compose Jacobian and Hessian
        grd_i, grd_j = grd_i.flatten(), grd_j.flatten()
        if model in ('affine', 'similarity'):
            I_dt_new, W = I_dt_new.flatten(), W.flatten()
            IN = ~np.isnan(I_dt_new)
            I_di, I_dj = I_di.flatten(), I_dj.flatten()

            if model in ('affine'):
                dWdp = np.array([
                          I_di * grd_i,
                          I_dj * grd_i,
                          I_di * grd_j,
                          I_dj * grd_j,
                          I_di, I_dj])

                A = (np.tile(W[IN],(6,1)) * dWdp[:,IN]) @ dWdp[:,IN].T
                y = (np.tile(W[IN],(6,1)) * dWdp[:,IN]) @ (I_dt_new[IN] * W[IN])
            elif model in ():
                dWdp = np.array([
                    (I_di * grd_i) - (I_di * grd_j),
                    (I_dj * grd_i) + (I_dj * grd_j),
                    I_di, I_dj])

                A = (np.tile(W[IN], (4, 1)) * dWdp[:, IN]) @ dWdp[:, IN].T
                y = (np.tile(W[IN], (4, 1)) * dWdp[:, IN]) @ (I_dt_new[IN] * W[IN])
                A = np.pad(A, ((2, 0), (2, 0)), 'constant')
                y = np.pad(y, ((2, 0), (0, 0)), 'constant')
        elif model in ('simple'):
            IN = ~np.isnan(I_dt_new)
            dWdp = np.array([[np.sum(W[IN] * I_di2[IN]),
                              np.sum(W[IN] * I_di[IN] * I_dj[IN])],
                             [np.sum(W[IN] * I_di[IN] * I_dj[IN]),
                              np.sum(W[IN] * I_dj2[IN])]])

            A = dWdp
            y = - np.array([[np.sum(W[IN] * I_di[IN] * I_dt_new[IN])],
                            [np.sum(W[IN] * I_dj[IN] * I_dt_new[IN])]])
            A = np.pad(A, ((4,0),(4,0)), 'constant') # add extra zero-components
            y = np.pad(y, ((4,0),(0,0)), 'constant')

        #todo
        #if episolar.shape[0] != 0:
        #    y =
        #    A = np.vstack((A, episolar.T))


        if y.size>=6: # structure should not become ill-posed
            try:
                (dp,_,_,s) = np.linalg.lstsq(A, y, rcond=None)#[0]
            except ValueError:
                pass
        else:
            break
        p += dp.T
        p_stack[i,:] = p

    Aff = np.array([[1, 0, 0], [0, 1, 0]]) + \
        p_stack[-1,:].reshape(3,2).T
    # np.argmin(res)
    u, v = Aff[0,-1], Aff[1,-1]
    A = np.linalg.inv(Aff[:,0:2]).T
    if model in ('similarity'):
        # s*cos(theta), s*sin(theta)
        breakpoint()
    snr = s[:2]
    return -2*u, -2*v, A, snr

def hough_optical_flow(I1, I2, param_resol=100, sample_fraction=0,
                       num_estimates=1, max_amp=1,
                       preprocessing=None):
    """ estimating optical flow through the Hough transform, based on [FT79]_.

    Parameters
    ----------
    I1 : {numpy.array, numpy.ma}, size=(m,n,b), dtype=float, ndim=2
        first image array, np.nan entries indicate no-data.
    I2 : {numpy.array, numpy.ma}, size=(m,n,b), dtype=float, ndim=2
        second image array, np.nan entries indicate no-data.
    param_resol : integer
        resolution of the Hough transform, that is, the amount of elements along
        one axis
    sample_fraction : {float,integer}
        * 0< & <1, a fraction is used in the sampling
        * >1, the number of elements given are used for sampling
    max_amp : float, default=1.
        maximum displacement to be taken into account
    preprocessing : {None, 'hist_equal'}
        preprocessing steps to apply to the input imagery
            * None : use the raw imagery
            * 'hist_eq' : apply histogram equalization, this is a common step
            to comply with the brightness consistency assumption in remote
            sensing imagery [GL12]_
    num_estimates : integer
        amount of displacement estimates

    Returns
    -------
    di,dj : float
        sub-pixel displacement
    score : float, range=0...1
        probability or amount of support for the estimate

    References
    ----------
    .. [FT79] Fennema & Thompson, "Velocity determination in scenes containing
              several moving objects" Computer graphics and image processing,
              vol.9 pp.301-317, 1979.
    .. [GL12] Guo & Lü, "Phase-shifting algorithm by use of Hough transform"
              Optics express vol.20(23) pp.26037-26049, 2012.
    """
    assert type(I1) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert type(I2) in (np.ma.core.MaskedArray, np.ndarray), \
        ("please provide an array")
    assert I1.ndim == 2, ("only grayscale imagery are implemented")
    assert I2.ndim == 2, ("only grayscale imagery are implemented")
    are_two_arrays_equal(I1, I2)

    # resolve no-data or masked array
    Msk_1, Msk_2 = np.isnan(I1), np.isnan(I2)
    if type(I1) == np.ma.core.MaskedArray:
        Msk_1 = np.logical_or(Msk_1, np.ma.getmaskarray(I1))
    if type(I1) == np.ma.core.MaskedArray:
        Msk_1 = np.logical_or(Msk_2, np.ma.getmaskarray(I2))
    I1[Msk_1], I2[Msk_2] = np.nan, np.nan

    if preprocessing in ['hist_equal']:
        I1 = histogram_equalization(I1, I2)

    I_di, I_dj, I_dt = create_differential_data(I1, I2)

    # create data
    abs_G = np.hypot(I_di, I_dj)
    θ_G = np.arctan2(I_dj,I_di)

    ρ = np.divide(I_dt, abs_G, out=np.zeros_like(abs_G), where=abs_G!=0)

    # remove flat contrast data or data with NaN's
    IN = np.logical_and(~np.logical_and(I_di == 0, I_dj == 0),
                        np.logical_or(Msk_1, Msk_2))

    di, dj, score = hough_sinus(θ_G[IN], ρ[IN],
                                param_resol=param_resol,
                                max_amp=max_amp,
                                sample_fraction=sample_fraction,
                                num_estimates=num_estimates,
                                indexing='cartesian')
    # import matplotlib.pyplot as plt
    # plt.hexbin(θ_G[IN], ρ[IN], extent=(-3.14, +3.14, -1, +1)), plt.show()
    return di,dj, score

def _point_sample(φ, ρ, idx, param_resol, max_amp, u, v):
    democracy = np.zeros((param_resol, param_resol), dtype=np.float32)
    for counter in idx:
        diff = ρ[counter] - \
               (u*+np.sin(φ[counter]) +
                v*+np.cos(φ[counter]))
        # Gaussian weighting
        vote = np.exp(-np.abs(diff*param_resol)/max_amp)
        #todo: outer product, to speed-up
        np.putmask(vote, np.isnan(vote), 0)
        democracy += vote
    return democracy

def _histogram_sample(φ, ρ, param_resol, max_amp, u, v):
    H,φ_h,ρ_h = np.histogram2d(φ, ρ, bins=param_resol,
                               range=[[-180, +180], [-max_amp, +max_amp]])

    democracy = np.zeros((param_resol, param_resol), dtype=np.float32)
    for i, j in np.ndindex(H.shape):
        if H[i,j]==0: continue
        diff = ρ_h[j] - \
               (u*+np.sin(φ_h[i]) + v*+np.cos(φ_h[i]))
        vote = np.exp(-np.abs(diff*param_resol)/max_amp)
        np.putmask(vote, np.isnan(vote), 0)
        democracy += H[i,j]*vote
    return democracy

def hough_sinus(φ, ρ,
                param_resol=100, max_amp=1, sample_fraction=0,
                num_estimates=1, indexing='polar'):
    """ estimates parameters of sinus curve through the Hough transform

    Parameters
    ----------
    φ : numpy.array, size=(m,n), unit=radians
        array with angle values, or argument of a polar expression
    ρ : numpy.array, size=(m,n)
        array with amplitude values
    param_resol : integer
        amount of bins for each axis to cover the Hough space
    max_amp : float
        maximum extent of the sinus curve, and the Hough space
    sample_fraction : float
        * ==0 : use histogram
        * < 1 : takes a random subset of the collection
        * ==1 : uses all data given
        * > 1 : uses a random collection of the number specified
    num_estimates : integer
        amount of displacement estimates

    Returns
    -------
    φ_H : float
        estimated argument of the curve
    ρ_H : float
        estimated amplitude of the curve
    score_H : float, range=0...1
        probability of support of the estimate

    References
    ----------
    .. [GL79] Guo & Lü, "Phase-shifting algorithm by use of Hough transform"
              Optics express vol.20(23) pp.26037-26049, 2012.
    """
    are_two_arrays_equal(φ, ρ)

    normalize = True
    IN = np.logical_and.reduce((~np.isnan(φ), ~np.isnan(ρ), np.abs(ρ)<max_amp))
    if np.sum(IN)<2: return 0, 0, 0
    φ, ρ = φ[IN], ρ[IN]

    if normalize:
        ρ -= np.nanmedian(ρ)

    sample_size = ρ.size
    if sample_fraction==1:
        idx = np.arange(0, sample_size)
    elif sample_fraction==0:
        idx = None
    elif sample_fraction>1: # use the amount given by sample_fraction
        idx = np.random.choice(sample_size,
           np.round(np.minimum(sample_size, sample_fraction)).astype(np.int32),
           replace=False)
    else: # sample random from collection
        idx = np.random.choice(sample_size,
           np.round(sample_size*sample_fraction).astype(np.int32),
           replace=False)

    u,v = np.meshgrid(np.linspace(-max_amp,+max_amp, param_resol),
                      np.linspace(-max_amp,+max_amp, param_resol))

    if idx is None: # histogram goes quicker
        democracy = _histogram_sample(φ, ρ, param_resol, max_amp, u, v)
    else:
        democracy = _point_sample(φ, ρ, idx, param_resol, max_amp, u, v)

    # find multiple peaks if present and wanted
    ind, score = get_peak_indices(democracy, num_estimates=num_estimates)
    score /= sample_size # normalize

    if indexing in ('polar', 'circular',):
        rho_H, φ_H = np.zeros(num_estimates), np.zeros(num_estimates)
        for cnt,sc in enumerate(score):
            if sc!=0:
                rho_H[cnt] = np.sqrt( u[ind[cnt,0]][ind[cnt,1]]**2 +
                                 v[ind[cnt,0]][ind[cnt,1]]**2 )
                φ_H[cnt] = np.arctan2(u[ind[cnt,0]][ind[cnt,1]],
                                   v[ind[cnt,0]][ind[cnt,1]])
        return φ_H, rho_H, score
    elif indexing in ('cartesian', 'xy',):
        u_H, v_H = np.zeros(num_estimates), np.zeros(num_estimates)
        for cnt,sc in enumerate(score):
            if sc!=0:
                u_H[cnt] = u[ind[cnt,0]][ind[cnt,1]]
                v_H[cnt] = v[ind[cnt,0]][ind[cnt,1]]

        return u_H, v_H, score

def differential_stacking(I_st, id_1, id_2, dt):
    """ differential stack with different time intervals

    Parameters
    ----------
    I_st : numpy.array, size=(_,_,k)
        stack of satellite image templates
    id_1 : numpy.array, size=(n,1)
        index of the first image to be matched in the stack, for all n pairs
    id_2 : numpy.array, size=(n,1)
        index of the second image to be matched in the stack, for all n pairs
    dt : numpy.array, size=(n,1)
        time difference of the pair

    Returns
    -------
    diff_stack : {float, np.array}
        sub-pixel displacement, amount depends upon "num_estimates"
    """
    if isinstance(dt, float): dt = np.array([dt])
    if isinstance(id_1, int): id_1 = np.array([id_1])
    if isinstance(id_2, int): id_2 = np.array([id_2])

    for i in range(len(dt)):
        I1, I2 = I_st[..., id_1[i]], I_st[..., id_2[i]]
        # get local gradients
        I_di, I_dj, I_dt = create_differential_data(I1, I2)
        # transform to Hough-space coordinates
        abs_grad = np.hypot(I_di, I_dj)
        θ_grad = np.arctan2(I_dj, I_di)

        ρ = np.divide(I_dt, abs_grad,
                      out=np.zeros_like(abs_grad),
                      where=abs_grad != 0)
        θ = np.divide(θ_grad, 1.)  # Q_scaling[i])
        θ *= dt[i]

        # bring into collection
        if i == 0:
            diff_stack = np.array([ρ.flatten(), θ.flatten()]).T
        else:
            diff_stack = np.vstack((diff_stack, np.array([ρ.flatten(),
                                                          θ.flatten()]).T))
    return diff_stack


#todo episolar_optical_flow

