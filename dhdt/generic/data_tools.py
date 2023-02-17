import numpy as np

#from ..processing.matching_tools_frequency_subpixel import phase_jac
from ..processing.matching_tools import \
    get_integer_peak_location, get_peak_indices

def squared_difference(A,B):
    """
    efficient computation of the squared difference

    Parameters
    ----------
    A : np.array, size=(m,n)
        data array
    B : np.array, size=(m,n)
        data array

    Returns
    -------
    sq_diff : float
        sum of squared difference.

    Notes
    -----
    .. math :: \Sigma{[\mathbf{A}-\mathbf{B}]^2}
    """
    diff = A-B
    sq_diff = np.einsum('ijk,ijk->',diff,diff)
    return sq_diff

def compute_cost(A, y, params):
    n_samples = len(y)
    hypothesis = np.squeeze(A @ params)
    return (1/(2*n_samples))*np.sum(np.abs(hypothesis-y))

def gradient_descent(A, y, params, learning_rate=0.01, n_iters=100):
    n_samples, history = y.shape[0], np.zeros((n_iters))

    for i in range(n_iters):
        ɛ = np.squeeze(A @ params) - y
        
        # ɛ = (1 / (1 + ɛ**2))*ɛ
        
        # IN = np.abs(ɛ)<=np.quantile(np.abs(ɛ), quantile)
        # ɛ = ɛ[IN]
        # params = params - (learning_rate/n_samples) * A[IN,:].T @ ɛ
        params = params - (learning_rate/n_samples) * A.T @ ɛ
        history[i] = compute_cost(A, y, params)
    if history[0]<history[-1]:
        print('no convergence')
    return params, history

def secant(A, y, J, params, n_iters=5, print_diagnostics=False):
    """
    also known as Boyden's method

    Parameters
    ----------
    A : np.array, size=(m,2), dtype=float
        design matrix
    y : np.array, size=(m,1), dtype=float
        phase angle vector
    J : np.array, size=(m,2), dtype=float
        Jacobian matrix
    params : np.array
        initial estimate
    n_iters : integer, optional
        amount of maximum iterations that need to be excuted, the default is 5.
    print_diagnostics : boolean
        print the solution at each iteration

    Returns
    -------
    params : np.array, dtype=float
        final estimate
    history : np.array, size=(n_iters)
        evolution of the cost per estimation in the iteration.

    """
    history = np.zeros((n_iters))
    
    if len(params)==1: # one dimensional problem
        x0 = params.copy()-.1
        x1 = params.copy()
        for i in range(n_iters):
            fx0 = np.squeeze(A @ x0) - y
            fx1 = np.squeeze(A @ x1) - y
    
            x2 = x0 - ((x1-x0)*np.sum(fx0))/( np.sum(fx1 - fx0) )
            
            # update
            x0 = x1.copy()
            x1 = x2.copy()
            history[i] = compute_cost(A, y, x1)
    else: # multi-variate case
        x0 = params.copy()-.1
        x1 = params.copy()
        
        for i in range(n_iters):
            fx0 = np.squeeze(A @ x0) - y
            fx1 = np.squeeze(A @ x1) - y
            
            delta_x = x1 - x0
            delta_f = fx1 - fx0
            # estimate Jacobian
            J_new = J + \
                np.outer( (delta_f - (J @ delta_x)), delta_x) / \
                np.dot(delta_x,delta_x)
            
            # estimate new parameter set
            dx = np.linalg.lstsq(J_new, -fx1, rcond=None)[0]
            x2 = x1 + dx
            
            if print_diagnostics:
                print('di:{:+.4f}'.format(x2[0])+' dj:{:+.4f}'.format(x2[1]))
            # update
            x0,x1,J = x1.copy(), x2.copy(), J_new.copy()
            history[i] = compute_cost(A, y, x1)
        params = x1
        
    if (history[0]<history[-1]) and print_diagnostics:
        print('no convergence')
    return params, history    

def estimate_sinus(φ, ρ, bias=True):
    """ least squares estimation of a sinus function

    Parameters
    ----------
    φ : numpy.array, size=(m,), unit=degrees
        angle
    ρ : numpy.array, size=(m,)
        amplitude
    bias : boolean
        estimate offset as well

    Returns
    -------
    phase_hat : float
        least squares estimate of the phase
    amp_hat : float
        least squares estimate of the amplitude
    bias : float
        offset of the curve
    """
    assert φ.size>=2, 'please provide enough data points'
    assert ρ.size>=2, 'please provide enough data points'

    # design matrix
    if bias==True:
        A = np.vstack([np.sin(np.deg2rad(φ)),
                       np.cos(np.deg2rad(φ)),
                       np.ones(len(φ))]).T
    else:
        A = np.vstack([np.sin(np.deg2rad(φ)),
                       np.cos(np.deg2rad(φ))]).T

    x_hat = np.linalg.lstsq(A, ρ, rcond=None)[0]

    phase_hat = np.rad2deg(np.arctan2(x_hat[0], x_hat[1]))
    amp_hat = np.hypot(x_hat[0], x_hat[1])
    bias = x_hat[-1]
    return phase_hat, amp_hat, bias

def four_pl_curve(X, a, b, c, d):
    """ four parameter logistic curve, also known as, 4PL,  Hill model, see [1].

    Parameters
    ----------
    X : numpy.array, size=(m,)
        coordinate axis
    a : float
        the minimum asymptote
    b : float
        the steepness parameter
    c : float
        the inflection point
    d : float
        the maximum asymptote

    Returns
    -------
    Y : numpy.array, size=(m,)
        four parameter logistic curve

    Notes
    -----
    The parameters of this curve have an intuitive meaning and are as follows:

        .. code-block:: text

                       b
              ^ y     <->
              |         ┌------ <- d
              |        /
          a-> |-------┘
              +--------^--------> x
                       |c

    References
    ----------
    .. [1] Hill, "A new mathematical treatment of changes of ionic concentration
       in muscle and nerve under the action of electric currents, with a theory
       as to their mode of excitation", The Journal of physiology, vol.40(3)
       pp.190, 1910.
    """
    Y = d + np.divide(a-d, 1.0+np.power(X/c,b))
    return Y

def five_pl_curve(X, a, b, c, d, g):
    """ four parameter logistic curve, also known as, 4PL

    Parameters
    ----------
    X : numpy.array, size=(m,)
        coordinate axis
    a : float
        the minimum asymptote
    b : float
        the steepness parameter
    c : float
        the inflection point
    d : float
        the maximum asymptote
    g : float
        asymmetry of the curve

    Returns
    -------
    Y : numpy.array, size=(m,)
        four parameter logistic curve

    See Also
    --------
    four_pl_curve, gompertz_curve, s_curve
    """
    Y = d + np.divide(a-d, np.power(1.0+np.power(X/c, b), g))
    return Y

def gompertz_curve(X, a, b):
    Y = np.exp(-a*np.exp(-b*X))
    return Y

def s_curve(X, a=10, b=.5):
    """ transform intensity in non-linear way, also known as, sigmoid function

    enhances high and low intensities

    Parameters
    ----------
    X : np.array, size=(m,n)
        array with intensity values
    a : float, default=10
        slope, default is fitted for the domain of 0...1
    b : float, default=.5
        intercept, default is fitted for the domain of 0...1

    Returns
    -------
    Y : np.array, size=(m,n)
        array with transform

    See Also
    --------
    four_pl_curve, five_pl_curve, gompertz_curve

    References
    ----------
    .. [1] Fredembach & Süsstrunk. "Automatic and accurate shadow detection from
       (potentially) a single image using near-infrared information", EPFL Tech
       Report 165527, 2010.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0,1,.001)
    >>> fx = s_curve(x)
    >>> plt.plot(x,fx), plt.show()
    shows the mapping function
    """
    fe = -a * (X - b)
    Y = np.divide(1, 1 + np.exp(fe))
    return Y

def logit(Y):
    """ logit function, or the inverse of the standard logistic function
    """
    return np.log10(np.divide(Y, 1-Y))

def logit_weighting(Y, sigma_Y=1., sigma_0=1.):
    """ calculate transformed weighting values for logit function

    Parameters
    ----------
    Y : numpy.array, size=(m,)
        y-values prior to logit transformation
    sigma_Y : {float, numpy.array}, size={(m,),1}
        variance of the data set, or for the individual elements
    sigma_0 : float
        proportionality factor, that is, the variance of a function of
        unit weight

    Returns
    -------
    W : numpy.array, size=(m,)
        corresponding transformed weighting

    See Also
    --------
        See Also
    --------
    ln_weighting, reciprocal_weighting, exponential_weighting, squared_weighting

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """

    W = np.multiply(Y**2,(1-Y)**2) * np.divide(sigma_Y, sigma_0)
                                                        # from Table 9 in [1]
    return W

def ln_weighting(Y, sigma_Y=1., sigma_0=1.):
    """ calculate transformed weighting values for natural logarithm function

    Parameters
    ----------
    Y : numpy.array, size=(m,)
        y-values prior to logit transformation
    sigma_Y : {float, numpy.array}, size={(m,),1}
        variance of the data set, or for the individual elements
    sigma_0 : float
        proportionality factor, that is, the variance of a function of
        unit weight

    Returns
    -------
    W : numpy.array, size=(m,)
        corresponding transformed weighting

    See Also
    --------
    logit_weighting, reciprocal_weighting, exponential_weighting,
    squared_weighting

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """
    W = Y**2 * np.divide(sigma_Y, sigma_0)  # from Table 9 in [1]
    return W

def reciprocal_weighting(Y, sigma_Y=1., sigma_0=1.):
    """ calculate transformed weighting values for reciprocal function

    Parameters
    ----------
    Y : numpy.array, size=(m,)
        y-values prior to logit transformation
    sigma_Y : {float, numpy.array}, size={(m,),1}
        variance of the data set, or for the individual elements
    sigma_0 : float
        proportionality factor, that is, the variance of a function of
        unit weight

    Returns
    -------
    W : numpy.array, size=(m,)
        corresponding transformed weighting

    See Also
    --------
    logit_weighting, ln_weighting, exponential_weighting, squared_weighting

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """
    W = Y**4 * np.divide(sigma_Y, sigma_0) # from Table 9 in [1]
    return W

def exponential_weighting(Y, sigma_Y=1., sigma_0=1.):
    """ calculate transformed weighting values for exponential function

    Parameters
    ----------
    Y : numpy.array, size=(m,)
        y-values prior to logit transformation
    sigma_Y : {float, numpy.array}, size={(m,),1}
        variance of the data set, or for the individual elements
    sigma_0 : float
        proportionality factor, that is, the variance of a function of
        unit weight

    Returns
    -------
    W : numpy.array, size=(m,)
        corresponding transformed weighting

    See Also
    --------
    logit_weighting, ln_weighting, reciprocal_weighting, squared_weighting

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """
    W = np.divide(sigma_0, np.multiply(np.exp(2*Y) ,sigma_Y))
                                                    # from Table 9 in [1]
    return W

def squared_weighting(Y, sigma_Y=1., sigma_0=1.):
    """ calculate transformed weighting values for squared function

    Parameters
    ----------
    Y : numpy.array, size=(m,)
        y-values prior to logit transformation
    sigma_Y : {float, numpy.array}, size={(m,),1}
        variance of the data set, or for the individual elements
    sigma_0 : float
        proportionality factor, that is, the variance of a function of
        unit weight

    Returns
    -------
    W : numpy.array, size=(m,)
        corresponding transformed weighting

    See Also
    --------
    logit_weighting, ln_weighting, reciprocal_weighting, exponential_weighting

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> X = np.linspace(0,10,100).astype(float)
    >>> m = 10.
    >>> Y = m*X**2
    >>> Y += np.random.rand(100)*10 # corrupt with some white noise
    >>> A = (X**2)[:,np.newaxis]

    calculate via weithed least-squares adjustment, taking exponential function
    into account
    >>> w = squared_weighting(Y)
    >>> W = np.sqrt(np.diag(w))
    >>> Aw, Yw = np.dot(W,A), np.dot(Y,W)
    >>> m_wls = np.linalg.lstsq(Aw, Yw, rcond=-1)[0]

    via normal least squares
    >>> m_lsq = np.linalg.lstsq(A, y, rcond=-1)[0]

    plot the different results
    >>> plt.figure()
    >>> plt.plot(X,Y), plt.plot(X,m_wls*X**2), plt.plot(X,m_lsq*X**2)
    >>> plt.axis([X[0], X[-1], Y[0], Y[-1]])
    >>> plt.legend(['raw','weighted least squares','normal least squares']);

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """
    W = np.divide(sigma_0**2, np.multiply(4*np.power(Y,2) ,sigma_Y))
                                                    # from Table 9 in [1]
    return W

def hough_transf(x, y, w=None,
                 param_resol=100, bnds=[-1, +1, -1, +1], sample_fraction=1,
                 num_estimates=1):
    """ estimates parameters of a line through the Hough transform

    Parameters
    ----------
    x,y : numpy.array, size=(m,)
        coordinates on a line
    w : numpy.array, size=(m,)
        array with weights
    param_resol : integer
        amount of bins for each axis to cover the Hough space
    bnds : list, default=[-1,+1,-1,+1]
        bounds or extent of the parameter/feature space
    sample_fraction : float
        * < 1 : takes a random subset of the collection
        * ==1 : uses all data given
        * > 1 : uses a random collection of the number specified
    num_estimates : integer
        amount of parameter set estimates

    Returns
    -------
    a : float
        estimated slope
    b : float
        estimated intercept
    score_H : float, range=0...1
        probability of support of the estimate

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> a,b,_ = hough_transf(x, y)

    >>> plt.figure()
    >>> plt.plot(x,y), plt.plot(x,a*x+b)
    >>> plt.show()
    """
    if w is None: w = np.ones_like(x)
    x, y, w = x.flatten(), y.flatten(), w.flatten()

    sample_size = x.size
    if sample_fraction == 1:
        idx = np.arange(0, sample_size)
    elif sample_fraction > 1:  # use the amount given by sample_fraction
        idx = np.random.choice(sample_size,
                               np.round(np.minimum(sample_size, sample_fraction)).astype(np.int32),
                               replace=False)
    else:  # sample random from collection
        idx = np.random.choice(sample_size,
                               np.round(sample_size * sample_fraction).astype(np.int32),
                               replace=False)

    ρ_max = np.maximum(np.hypot(bnds[0], bnds[1]),
                         np.hypot(bnds[2], bnds[3]))

    (θ, ρ) = np.meshgrid(np.deg2rad(np.linspace(-90., +90., param_resol)),
                               np.linspace(-ρ_max, +ρ_max, param_resol))
    democracy = np.zeros((param_resol, param_resol), dtype=np.float32)

    cos_th, sin_th = np.cos(θ), np.sin(θ)

    for counter in idx:
        diff = (cos_th * x[counter] + sin_th * y[counter]) - ρ
        # Gaussian weighting
        vote = np.exp(-np.abs(diff * w[counter]) / ρ_max)
        np.putmask(vote, np.isnan(vote), 0)
        democracy += vote

    # get integer peak location
    idx, val = get_peak_indices(democracy, num_estimates=1)
    idx_flat = np.ravel_multi_index(idx.T, democracy.shape, 'C')

    # transform to slope and intercept
    slope_hat = np.divide(-cos_th.flatten()[idx_flat],
                          +sin_th.flatten()[idx_flat])
#    intercept_hat = np.divide(ρ.flatten()[idx_flat],
#                              cos_th.flatten()[idx_flat]) # x-intercept
    intercept_hat = np.divide(rho.flatten()[idx_flat],
                              sin_th.flatten()[idx_flat]) # y-intercept
    return slope_hat, intercept_hat, val