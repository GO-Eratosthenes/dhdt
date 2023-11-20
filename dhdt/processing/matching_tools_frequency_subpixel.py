# general libraries
import warnings
import numpy as np

# image processing libraries
from scipy import ndimage
from skimage.transform import radon
from skimage.measure.fit import _dynamic_max_trials

from ..generic.data_tools import gradient_descent, secant
from ..preprocessing.multispec_transforms import principle_component_analysis
from .matching_tools_frequency_filters import \
    raised_cosine, thresh_masking, normalize_power_spectrum, local_coherence, \
    make_fourier_grid, construct_phase_plane, cross_spectrum_to_coordinate_list

def phase_jac(Q, m, W=np.array([]),
              F1=np.array([]), F2=np.array([]), rank=2): # wip
    """
    Parameters
    ----------
    Q : numpy.array, size=(_,_), dtype=complex
        cross spectrum
    m : numpy.array, size=(2,1), dtype=float
        displacement estimate,  in pixel coordinate system
    W : numpy.array, size=(m,n), dtype=float | boolean
        weigthing matrix, in a range of 0...1
    F1 : np,array, size=(m,n), dtype=integer
        coordinate of the first axis from the Fourier spectrum.
    F2 : np,array, size=(m,n), dtype=integer
        coordinate of the second axis from the Fourier spectrum
    rank : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    dQdm : numpy.array, size=(m,n)
        Jacobian of phase estimate
    """
    # metric system:      Fourier-based flip
    #        y            +------><------+
    #        ^            |              |
    #        |            |              |
    #        |            v              v
    # <------+-------> x
    #        |            ^              ^
    #        |            |              |
    #        v            +------><------+
    #
    # indexing   |           indexing    ^ y
    # system 'ij'|           system 'xy' |
    #            |                       |
    #            |       i               |       x
    #    --------+-------->      --------+-------->
    #            |                       |
    #            |                       |
    #            | j                     |
    #            v                       |
    assert type(Q)==np.ndarray, ("please provide an array")

    if Q.shape[0]==Q.shape[1]: # if Q is a cross-spectral matrix
        if W.size==0: # if W is not given
            W = np.ones((Q.shape[0], Q.shape[1]), dtype=float)
        if F1.size==0:
            F1,F2 = make_fourier_grid(Q, indexing='ij')
    else: # list format
        F1,F2 = Q[:,0], Q[:,1]
        Q = Q[:,-1]
        if W.size==0:
            W = np.ones_like(Q, dtype=float)

    if rank==2: # default
        dXY = 1 - np.multiply(np.real(Q), +np.cos(F1*m[0]+F2*m[1])) \
            - np.multiply(np.imag(Q), -np.sin(F1*m[0]+F2*m[1]))
    else:
        C_hat = construct_phase_plane(Q, m[0], m[1], indexing='ij')
        QC = Q-C_hat # convert complex vector difference to metric
        dXY = np.abs(np.multiply(W, QC)**rank)

    dQdm = np.array([np.multiply(2*W.flatten()*F1.flatten(),dXY.flatten()), \
                     np.multiply(2*W.flatten()*F2.flatten(),dXY.flatten())]).T
    return dQdm

def phase_secant(data, W=np.array([]), x_0=np.zeros((2))): # wip
    """get phase plane of cross-spectrum through secant

    find slope of the phase plane through secant method (or Newton's method)
    in multiple dimensions it is known as the Broyden's method.

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    or     numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last collumn
    W : numpy.array, size=(m,n), dtype=boolean
        index of data that is correct
    or     numpy.array, size=(m*n,1), dtype=boolean
        list with classification of correct data

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_gradient_descend

    References
    ----------
    .. [Br65] Broyden, "A class of methods for solving nonlinear simultaneous
              equations" Mathematics and computation. vol.19(92) pp.577--593,
              1965.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_secant(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(data)==np.ndarray, ("please provide an array")

    data = cross_spectrum_to_coordinate_list(data, W)
    J = phase_jac(data, x_0)
    x_hat,_ = secant(data[:,:-1], data[:,-1], J, x_0, \
                     n_iters=10)
    di,dj = 2*x_hat[0], 2*x_hat[1]
    return di,dj

def phase_gradient_descend(data, W=np.array([]), x_0=np.zeros((2)),
                           learning_rate=1, n_iters=50): # wip
    """get phase plane of cross-spectrum through principle component analysis

    find slope of the phase plane through
    principle component analysis

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    data : numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last collumn
    W : numpy.array, size=(m,n), dtype=boolean
        index of data that is correct
    W : numpy.array, size=(m*n,1), dtype=boolean
        list with classification of correct data

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_lsq

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_gradient_descend(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))

    """
    assert type(data)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    data = cross_spectrum_to_coordinate_list(data, W)
    x_hat,_ = gradient_descent(data[:,:-1], data[:,-1], x_0,
                               learning_rate=learning_rate,
                               n_iters=n_iters)
    di,dj = x_hat[1], x_hat[0]
    return di,dj

def phase_tpss(Q, W, m, p=1e-4, k=4, j=5, n=3): #wip
    """get phase plane of cross-spectrum through two point step size iteration

    find slope of the phase plane through
    two point step size for phase correlation minimization [BB88]_
    which is implemented by [Le07]_

    Parameters
    ----------
    Q : numpy.ndarray, size=(_,_), dtype=complex
        cross spectrum
    m0 : numpy.ndarray, size=(2,1)
        initial displacement estimate
    p : float, default=1e4
        closing error threshold
    k : integer, default=4
        number of refinements in iteration
    j : integer, default=5
        number of sub routines during an estimation
    n : integer, default=3
        mask convergence factor
    Returns
    -------
    di,dj : float, size=(2,1)
        sub-pixel displacement
    snr: float
        signal-to-noise ratio

    See Also
    --------
    phase_svd, phase_radon, phase_difference, phase_jac

    References
    ----------
    .. [BB88] Barzilai & Borwein. "Two-point step size gradient methods", IMA
              journal of numerical analysis. vol.8 pp.141--148, 1988.
    .. [Le07] Leprince, et.al. "Automatic and precise orthorectification,
              coregistration, and subpixel correlation of satellite images,
              application to ground deformation measurements", IEEE Transactions
              on geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.
    """
    m = np.squeeze(m)
    s = 1.#1.25#.5#1.#.5#2.

    Q = normalize_power_spectrum(Q)
    #W = W/np.sum(W) # normalize weights

    Fx,Fy = make_fourier_grid(Q)

    # initialize
    m_min = m.copy().ravel()
    m_min += np.array([-.1, -.1])

    J_min = phase_jac(Q, m_min, W=W)
    g_min = np.sum(J_min, axis=0)

    #print('di:{:+.4f}'.format(m[0])+' dj:{:+.4f}'.format(m[1]))
    for i in range(k):
        p = 1
        while True:

            J = phase_jac(Q, m, W=W)
            g = np.sum(J, axis=0)

            # difference
            dm,dg = m - m_min, g - g_min

            #alpha = np.dot(dm,dg)/np.dot(dg,dg)
            alpha = np.dot(dm,dm)/(s*np.dot(dm,dg))

            if (np.all(np.abs(m - m_min)<=p)) or (k>=j):
                break

            # update
            m_min, g_min = np.copy(m), np.copy(g)

            #if i ==0:
            m -= alpha*dg
            #else:
            #    m -= alpha*dg
            print('di:{:+.4f}'.format(m[0])+' dj:{:+.4f}'.format(m[1]))
            p += 1

        # optimize weighting matrix
        #φ = np.abs(QC*np.conjugate(QC))/2
        C = 1j*-np.sin(Fx*m[1] + Fy*m[0])
        C += np.cos(Fx*m[1] + Fy*m[0])
        QC = (Q-C)**2 # np.abs(Q-C)#np.abs(Q-C)
        dXY = np.abs(np.multiply(W, QC))
        W = W*(1-(dXY/4))**n
#        φ = np.multiply(2*W,\
#                          (1 - np.multiply(np.real(Q),
#                                           +np.cos(Fx*m[1] + Fy*m[0])) - \
#                           np.multiply(np.imag(Q),
#                                       -np.sin(Fx*m[1] + Fy*m[0]))))
        #W = np.multiply(W, (1-(φ/4))**n)
#    snr = 1 - (np.sum(φ)/(4*np.sum(W)))
    snr = 0
    m = -1*m
    return m[0], m[1], snr

def phase_slope_1d(t, rad=.1):
    """ estimate the slope and intercept for one-dimensional signal

    Parameters
    ----------
    t : numpy.array, size=(m,1), dtype=complex
        angle values.
    rad : float, range=(0.0,0.5)
        radial inclusion, seen from the center

    Returns
    -------
    x_hat : numpy.array, size=(2,1)
        estimated slope and intercept.

    See also
    --------
    phase_svd
    """
    assert type(t)==np.ndarray, ("please provide an array")

    idx_sub = np.arange(np.ceil((0.5-rad)*len(t)), \
                    np.ceil((0.5+rad)*len(t))+1).astype(int)
    y_ang = np.unwrap(np.angle(t[idx_sub]),axis=0)
    A = np.column_stack([np.transpose(idx_sub-1), np.ones((len(idx_sub)))])
    x_hat = np.linalg.lstsq(A, y_ang, rcond=None)[0]
    return x_hat

def phase_svd(Q, W, rad=0.1):
    """get phase plane of cross-spectrum through single value decomposition

    find slope of the phase plane through
    single value decomposition, see [Ho03]_

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        cross spectrum
    W : numpy.array, size=(m,n), dtype=float
        weigthing matrix
    rad : float, range=(0.0,0.5)
        radial inclusion, seen from the center

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_tpss, phase_radon, phase_difference

    References
    ----------
    .. [Ho03] Hoge, W.S. "A subspace identification extension to the phase
              correlation method", IEEE transactions on medical imaging,
              vol.22(2) pp.277-280, 2003.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_svd(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(Q)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    if np.abs(Q).ptp()>1:
        Q = normalize_power_spectrum(Q)

    rad = np.minimum(rad, 0.5)
    (m,n) = Q.shape
    Q,W = np.fft.fftshift(Q), np.fft.fftshift(W)

    # decompose axis
    n_elements = 1
    try:
        u,s,v = np.linalg.svd(W*Q) # singular-value decomposition
    except np.linalg.LinAlgError:
        return 0, 0
    sig = np.zeros((m,n))
    sig[:m,:m] = np.diag(s)
    sig = sig[:,:n_elements] # select first element only
    # v = v[:n_elements,:]
    # reconstruct
    # b = u.dot(sig.dot(v))
    t_m = np.transpose(v).dot(sig)
    t_n = u.dot(sig)# transform

    d_n = phase_slope_1d(t_n, rad)
    d_m = phase_slope_1d(t_m, rad)

    di = -d_n[0][0]*n / (2*np.pi)
    dj = -d_m[0][0]*m / (2*np.pi)
    return di, dj

def phase_difference_1d(Q, W=np.array([]), axis=0):
    """get displacement from phase plane along one axis through differencing

    find slope of the phase plane through
    local difference of the phase angles, see [Ka89]_

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    W : numpy.array, size=(m,n), dtype=boolean
        weigthing matrix

    Returns
    -------
    dj : float
        sub-pixel displacement

    See Also
    --------
    phase_tpss, phase_svd, phase_difference

    References
    ----------
    .. [Ka89] Kay, S. "A fast and accurate frequency estimator", IEEE
              transactions on acoustics, speech and signal processing,
              vol.37(12) pp.1987-1990, 1989.
    """
    assert type(Q)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    if axis==0:
        Q = np.transpose(Q)
    m,n = Q.shape

    #estimate period
    Q_dj = np.roll(Q, (0,1))

    Q_diff = np.multiply(np.conj(Q),Q_dj)

    Delta_dj = np.angle(Q_diff)/np.pi
    if W.size==0:
        # find coherent data
        Qn = normalize_power_spectrum(Q)
        C = local_coherence(Qn, ds=1)
        C = np.minimum(C, np.roll(C, (0,1)))

        W = C>np.quantile(C,0.9) # get the best 10%

    dj = np.median(Delta_dj[W])*(m//2)
    return dj

def phase_difference(Q, W=np.array([])):
    """get displacement from phase plane through neighbouring vector difference

    find slope of the phase plane through local difference of the phase angles

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    W : numpy.array, size=(m,n), dtype=boolean
        weigthing matrix

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_tpss, phase_svd, phase_difference

    References
    ----------
    .. [Ka89] Kay, S. "A fast and accurate frequency estimator", IEEE
              transactions on acoustics, speech and signal processing,
              vol.37(12) pp.1987-1990, 1989.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_svd(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))

    """
    assert type(Q)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    di = phase_difference_1d(Q, W, axis=0)
    dj = phase_difference_1d(Q, W, axis=1)

    return di,dj

def phase_lsq(data, W=np.array([])):
    """get phase plane of cross-spectrum through least squares plane fitting

    find slope of the phase plane through
    principle component analysis

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    or     numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last
    W : numpy.array, size=(m,n), dtype=boolean
        index of data that is correct
    or     numpy.array, size=(m*n,1), dtype=boolean
        list with classification of correct data

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_pca, phase_ransac, phase_hough

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_lsq(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(data)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    data = cross_spectrum_to_coordinate_list(data, W)
    A,y = data[:,:-1], data[:,-1]

    M = A.transpose().dot(A)
    V = A.transpose().dot(y)

    # pseudoinverse:
    Mp = np.linalg.inv(M.transpose().dot(M)).dot(M.transpose())

    #Least-squares Solution
    plane_normal = Mp.dot(V)

    di = 2*plane_normal[0]
    dj = 2*plane_normal[1]
    return di, dj

def phase_pca(data, W=np.array([])):
    """get phase plane of cross-spectrum through principle component analysis

    find slope of the phase plane through
    principle component analysis

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    or     numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last
    W : numpy.array, size=(m,n), dtype=boolean
        index of data that is correct
    or     numpy.array, size=(m*n,1), dtype=boolean
        list with classification of correct data

    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_lsq

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj = phase_pca(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))

    """
    assert type(data)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    data = cross_spectrum_to_coordinate_list(data, W)

    eigen_vecs, eigen_vals = principle_component_analysis(data)
    e3 = eigen_vecs[:,np.argmin(eigen_vals)] # normal vector
    di = (-2*e3[0]/e3[-1])
    dj = (-2*e3[1]/e3[-1])
    return di, dj
# PCA is sensative to data contamination, see
# Hubert, Rousseeuw, van Aelst. 2008
# High-breakdown robust multivariate methods

def phase_weighted_pca(Q, W): #todo
    """get phase plane of cross-spectrum through principle component analysis

    find slope of the phase plane through
    principle component analysis

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    or     numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last
    W : numpy.array, size=(m,n), dtype=boolean
        index of data that is correct
    or     numpy.array, size=(m*n,1), dtype=boolean
        list with classification of correct data
    Returns
    -------
    di,dj : float
        sub-pixel displacement

    See Also
    --------
    phase_lsq

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> W = gaussian_mask(Q)
    >>> di,dj,_,_ = phase_weighted_pca(Q, W)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(Q)==np.ndarray, ('please provide an array')
    assert type(W)==np.ndarray, ('please provide an array')

    data = cross_spectrum_to_coordinate_list(Q)
    weights = W.flatten()

    covar = np.dot(data.T, data)
    covar /= np.dot(weights.T, weights)

    try: # eigenvalues might not converge
        eigen_vals, eigen_vecs = np.linalg.eigh(covar)
        e3 = eigen_vecs[:,np.argmin(eigen_vals)] # normal vector
        di = (-2*e3[0]/e3[-1])
        dj = (-2*e3[1]/e3[-1])
    except np.linalg.LinAlgError:
        di, dj = 0, 0
    return di, dj

# from skimage.measure
def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None,
           params_bounds=0):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.
    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:
    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.
    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.
    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:
         * ``success = estimate(*data)``
         * ``residuals(*data)``
        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):
            N >= log(1 - probability) / log(1 - e**m)
        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples value.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation
    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------
    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> xc, yc = 20, 30
    >>> a, b = 5, 10
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> rng = np.random.default_rng(203560)  # do not copy this value
    >>> data += rng.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> np.round(model.params)  # doctest: +SKIP
    array([ 72.,  75.,  77.,  14.,   1.])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    >>> abs(np.round(ransac_model.params))
    array([20., 30., 10.,  6.,  2.])

    >>> inliers  # doctest: +SKIP
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)

    >>> sum(inliers) > 40
    True

    RANSAC can be used to robustly estimate a geometric transformation. In this section,
    we also show how to use a proportion of the total samples, rather than an absolute number.

    >>> from skimage.transform import SimilarityTransform
    >>> rng = np.random.default_rng()
    >>> src = 100 * rng.random((50, 2))
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1,
    ...                              translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> ratio = 0.5  # use half of the samples
    >>> min_samples = int(ratio * len(src))
    >>> model, inliers = ransac((src, dst), SimilarityTransform, min_samples,
    ...                         10,
    ...                         initial_inliers=np.ones(len(src), dtype=bool))

    >>> inliers  # doctest: +SKIP
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True])
    """

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = np.random.default_rng(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data, )
    num_samples = len(data[0])

    if not (0 < min_samples < num_samples):
        raise ValueError("`min_samples` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), num_samples))

    # for the first run use initial guess of inliers
    spl_idxs = (initial_inliers if initial_inliers is not None
                else random_state.choice(num_samples, min_samples, replace=False))

    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        spl_idxs = random_state.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples, params_bound=params_bounds)

        # loop through potential solutions
        for potential_param in np.arange(success):
            potential_model = model_class()
            potential_model.params = sample_model.params[potential_param][0:min_samples]

            potential_model_residuals = np.abs(potential_model.residuals(*data))

            # consensus set / inliers
            potential_model_inliers = potential_model_residuals < \
                residual_threshold
            potential_model_residuals_sum = np.sum(potential_model_residuals**2)

            # choose as new best model if number of inliers is maximal
            potential_inlier_num = np.sum(potential_model_inliers)
            if (
                # more inliers
                potential_inlier_num > best_inlier_num
                # same number of inliers but less "error" in terms of residuals
                or (potential_inlier_num == best_inlier_num
                    and potential_model_residuals_sum < best_inlier_residuals_sum)
            ):
                best_model = potential_model
                best_inlier_num = potential_inlier_num
                best_inlier_residuals_sum = potential_model_residuals_sum
                best_inliers = potential_model_inliers
                dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                         num_samples,
                                                         min_samples,
                                                         stop_probability)
                if (best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                    or num_trials >= dynamic_max_trials):
                    break

    # estimate final model using all inliers
    if not(best_inliers is not None and any(best_inliers)):
        best_model = None
        best_inliers = None
        warnings("No inliers found. Model not fitted")

    return best_model, best_inliers

class BaseModel(object):
    def __init__(self):
        self.params = None

class PlaneModel(BaseModel):
    """Least squares estimator for phase plane.

    Vectors/lines are parameterized using polar coordinates as functional model::
        z = x * dx + y * dy
    This estimator minimizes the squared distances from all points to the
    plane, independent of distance.
    """

    def estimate(self, data):
        """Estimate plane from data using least squares.
        Parameters
        ----------
        data : (N, 3) array
            N points with ``(x, y)`` coordinates of vector, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if data.shape[0] >= 2:  # well determined
            x_hat = np.linalg.lstsq(data[:,0:2], data[:,-1], rcond=None)[0]
        else:  # under-determined
            raise ValueError('At least two vectors needed.')
        self.params = (x_hat[0], x_hat[1])
        number_of_params = 1
        return number_of_params

    def residuals(self, data):
        """Determine residuals of data to model
        For each point the shortest distance to the plane is returned.

        Parameters
        ----------
        data : (N, 3) array
            N points with x, y coordinates and z values, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        x_hat = self.params

        Q_hat = data[:,0]*x_hat[0] + data[:,1]*x_hat[1]

        residuals = np.abs(data[:,-1] - Q_hat)
        return residuals

    def predict_xy(self, xy, params=None):
        """Predict vector using the estimated heading.
        Parameters
        ----------
        xy : numpy.array
            x,y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        Q_hat : numpy.array
            Predicted plane height at x,y-coordinates.
        """

        if params is None:
            params = self.params
        x_hat = params
        if xy.ndim<2:
            Q_hat = xy[0]*x_hat[0] + xy[1]*x_hat[1]
        else:
            Q_hat = xy[:,0]*x_hat[0] + xy[:,1]*x_hat[1]
        return Q_hat

class SawtoothModel(BaseModel):
    """Least squares estimator for phase sawtooth.
    """

    def estimate(self, data, params_bound=0):
        """Estimate plane from data using least squares.
        Parameters
        ----------
        data : (N, 3) array
            N points with ``(x, y)`` coordinates of vector, respectively.
        params_bound : float
            bound of the parameter space
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        if data.shape[0] >= 2:  # well determined
            if params_bound != 0:
                # create multitudes of cycles
                param_cycle = np.mgrid[-params_bound:+params_bound+1, \
                                       -params_bound:+params_bound+1]
                cycle_0 = param_cycle[:,:,0].flatten() # 2*np.pi() if in radians
                cycle_1 = param_cycle[:,:,1].flatten() # but -.5 ... +.5
                x_stack = np.zeros((cycle_0.size, 2))

                for val,idx in enumerate(cycle_0):
                    y = np.array([data[0,-1] + cycle_0[idx], \
                                  data[1,-1] + cycle_1[idx]])
                    x_hat = np.linalg.lstsq(data[:,0:2], \
                                            y, \
                                            rcond=None)[0]
                    x_stack[idx,0] = x_hat[0]
                    x_stack[idx,1] = x_hat[1]
                # multitutes = np.maximum(np.floor(params_bound/x_hat[0]), \
                #              np.floor(params_bound/x_hat[1]))
                # x_stack = np.stack((np.arange(1,multitutes+1)*x_hat[0], \
                #                     np.arange(1,multitutes+1)*x_hat[1])).T
            else:
                x_stack = np.linalg.lstsq(data[:,0:2], data[:,-1], rcond=None)[0]
        else:  # under-determined
            raise ValueError('At least two vectors needed.')
        self.params = tuple(map(tuple, x_stack))
        return x_stack.shape[0]

    def residuals(self, data):
        """Determine residuals of data to model
        For each point the shortest distance to the plane is returned.

        Parameters
        ----------
        data : (N, 3) array
            N points with x, y coordinates and z values, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        x_hat = self.params

        Q_hat = data[:,0]*x_hat[0] + data[:,1]*x_hat[1]
        Q_hat = np.remainder(Q_hat+.5,1)-.5 # phase wrapping
        residuals = np.abs(data[:,-1] - Q_hat)
        return residuals

    def predict_xy(self, xy, params=None):
        """Predict vector using the estimated heading.
        Parameters
        ----------
        xy : numpy.array
            x,y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        Q_hat : numpy.array
            Predicted plane height at x,y-coordinates.
        """

        if params is None:
            params = self.params
        x_hat = params
        if xy.ndim<2:
            Q_hat = xy[0]*x_hat[0] + xy[1]*x_hat[1]
            Q_hat = np.remainder(Q_hat+.5,1)-.5
        else:
            Q_hat = xy[:,0]*x_hat[0] + xy[:,1]*x_hat[1]
            Q_hat = np.remainder(Q_hat+.5,1)-.5
        return Q_hat

def phase_ransac(data, max_displacement=1, precision_threshold=.05):
    """robustly fit plane using RANSAC algorithm

    find slope of the phase plane through
    random sampling and consensus

    Parameters
    ----------
    data : numpy.array, size=(m,n), dtype=complex
        normalized cross spectrum
    or     numpy.array, size=(m*n,3), dtype=complex
        coordinate list with complex cross-sprectum at last

    Returns
    -------
    di : float
        sub-pixel displacement along vertical axis
    dj : float
        sub-pixel displacement along horizontal axis

    See Also
    --------
    phase_lsq, phase_svd, phase_hough, phase_pca

    References
    ----------
    .. [FB81] Fischler & Bolles. "Random sample consensus: a paradigm for model
              fitting with applications to image analysis and automated
              cartography" Communications of the ACM vol.24(6) pp.381-395, 1981.
    .. [To15] Tong et al. "A novel subpixel phase correlation method using
              singular value decomposition and unified random sample consensus"
              IEEE transactions on geoscience and remote sensing vol.53(8)
              pp.4143-4156, 2015.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_ransac(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(data)==np.ndarray, ('please provide an array')

    # what type of data? either list of coordinates or a cross-spectral matrix
    if data.shape[0]==data.shape[1]:
        (m,n) = data.shape
        Fx,Fy = make_fourier_grid(np.zeros((m,n)))
        Fx,Fy = np.fft.fftshift(Fx), np.fft.fftshift(Fy)
        Q = np.fft.fftshift(np.angle(data) / (2*np.pi))

        max_displacement = np.maximum(max_displacement, m//2)
        data = np.vstack((Fy.flatten(),
                          Fx.flatten(),
                          Q.flatten() )).T

    ransac_model, inliers = ransac(data, SawtoothModel, #PlaneModel,
                                   min_samples=int(2),
                                   residual_threshold=precision_threshold,
                                   max_trials=int(1e3),
                                   params_bounds=max_displacement)
#    IN = np.reshape(inliers, (m,n)) # what data is within error bounds

    di = ransac_model.params[0]
    dj = ransac_model.params[1]
    return di, dj

def phase_hough(data, max_displacement=64, param_spacing=1, sample_fraction=1.,
                W=np.array([])):
    assert type(data)==np.ndarray, ('please provide an array')

    # what type of data? either list of coordinates or a cross-spectral matrix
    if data.shape[0]==data.shape[1]:
        (m,n) = data.shape
        Fx,Fy = make_fourier_grid(np.zeros((m,n)))
        Fx,Fy = np.fft.fftshift(Fx), np.fft.fftshift(Fy)
        Q = np.fft.fftshift(np.angle(data) / (2*np.pi))

        if max_displacement==64:
            max_displacement = m//2
        data = np.vstack((Fx.flatten(),
                          Fy.flatten(),
                          Q.flatten() )).T

    # create voting space
    (dj,di) = np.meshgrid(np.arange(-max_displacement, \
                                    +max_displacement + param_spacing, \
                                    param_spacing),
                          np.arange(-max_displacement, \
                                    +max_displacement + param_spacing, \
                                    param_spacing))

    # create population that can vote
    sample_size = data.shape[0]
    if sample_fraction>=1:
        idx = np.arange(0, sample_size)
    elif W.size==0: # sample random from collection
        idx = np.random.choice(sample_size,
                               np.round(sample_size*sample_fraction).astype(np.int32),
                               replace=False)
    else: # use weights to select sampling
        idx = np.flip(np.argsort(data[:,-1]))
        idx = idx[0:np.round(sample_size*sample_fraction).astype(np.int32)]

    #vote = np.zeros_like(di, dtype=np.int32)
    vote = np.zeros_like(di, dtype=np.float32)
    for counter in idx:
        angle_diff = data[counter,-1] - \
            (np.remainder((data[counter,0]*dj + data[counter,1]*di)+.5,1)-.5)

        vote += 1 / (1 + (angle_diff/(.1*np.std(angle_diff)))**2) # cauchy weighting
        # hard threshold
        # vote += (np.abs(angle_diff) <= .1).astype(np.int32)


    (i_hough,j_hough) = np.unravel_index(np.argmax(vote), di.shape)
    di_hough,dj_hough = di[i_hough,j_hough], dj[i_hough,j_hough]

    return di_hough, dj_hough

def phase_radon(Q, coord_system='ij'):
    """get direction and magnitude from phase plane through Radon transform

    find slope of the phase plane through
    single value decomposition

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        cross spectrum

    Returns
    -------
    * di,dj : float, default
        cartesian displacement
    * θ,ρ : float
        magnitude and direction of displacement

    See Also
    --------
    phase_tpss, phase_svd, phase_difference

    References
    ----------
    .. [BF06] Balci & Foroosh. "Subpixel registration directly from the phase
              difference" EURASIP journal on advances in signal processing,
              pp.1-11, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from dhdt.testing.matching_tools import create_sample_image_pair

    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**5, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> di,dj,_,_ = phase_radon(Q)

    >>> assert(np.isclose(ti, di, atol=.2))
    >>> assert(np.isclose(tj, dj, atol=.2))
    """
    assert type(Q)==np.ndarray, ('please provide an array')

    (m, n) = Q.shape
    half = m // 2
    Q = np.fft.fftshift(Q)

    # estimate direction, through the radon transform
    W = np.fft.fftshift(raised_cosine(Q, beta=1e-5)).astype(bool)
    Q[~W] = 0 # make circular domain

    θ = np.linspace(0., 180., max(m,n), endpoint=False)
    R = radon(np.angle(Q), θ) # sinogram

    #plt.imshow(R[:half,:]), plt.show()
    #plt.imshow(np.flipud(R[half:,:])), plt.show()

    R_fold = np.abs(np.multiply(R[:half,:], R[half:,:]))
    radon_score = np.sum(R_fold, axis=0)
    score_idx = np.argmax(radon_score)
    θ = θ[score_idx]
    del R_fold, radon_score, score_idx

    # peaks can also be seen
    # plt.plot(R[:,score_idx]), plt.show()

    # estimate magnitude
    Q_rot = ndimage.rotate(Q, -θ,
                             axes=(1, 0), reshape=False, output=None,
                             order=3, mode='constant')
    rho = np.abs(phase_difference_1d(Q_rot, axis=1))
    if coord_system=='ij':
        θ = np.deg2rad(θ)
        di,dj = np.sin(θ)*rho, -np.cos(θ)*rho
        return di, dj
    else: # do polar coordinates
        return θ, rho


#def phase_binairy_stripe():
#    Notes
#    -----
#    [1] Zuo et al. "Registration method for infrared images under conditions
#    of fixed-pattern noise", Optics communications, vol.285 pp.2293-2302,
#    2012.
#    """


