import numpy as np

def online_steady_state(y_ol, X_fi=None, v_fi=None, d_fi=None,
                        lambda_1=.2, lambda_2=.1, lambda_3=.1):
    """ statistical online test if proces is in steady state, see [CR95]_.
    Which is a simplified F-test based upon the ratio of variances

    Parameters
    ----------
    y_ol : numpy.ndarray, size=(m,)
        variable of interest
    lambda_1, lambda_2, lambda_3 : float
        tuning paramters
    R_thres : float
        critical value for steady state detection

    Returns
    -------
    in_steady_state : boolean
        specifies if proces is in steady state
    R_i : numpy.ndarray, size=(m,)
        steady state metric

    References
    ----------
    .. [CR95] Cao & Rhinehart, "An efficient method for on-line identification
              of steady state" Journal of proces control, vol.5(6) pp.363-374,
              1995.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> N,c = 100, 1
    >>> x = np.linspace(0,7,N)
    >>> y = np.exp(-c*x) # create exponential decay
    >>> y += np.random.normal(loc=0.0, scale=0.05, size=N) # add noise

    >>> R_i = np.zeros_like(y)
    >>> X,v,d = None, None, None
    >>> for idx,val in enumerate(y):
    >>>     y_ol = y[:idx+1] # simulate online
    >>>     R_i[idx],X,v,d = online_steady_state(y_ol,X,v,d)

    >>> plt.figure(), plt.plot(x,R_i)
    """
    if len(y_ol)==1:  # initialization
        X_fi, v_fi = y_ol, 1
        d_fi = 2*v_fi
    else:
        v_fi = np.multiply(lambda_2, (y_ol[-1] - X_fi) ** 2) + \
               np.multiply(1.-lambda_2, v_fi)           # eq.9 in [1]
        X_fi = np.multiply(lambda_1, y_ol[-1]) + \
               np.multiply(1.-lambda_1, y_ol[-2])       # eq.2 in [1]
        d_fi = np.multiply(lambda_3, (y_ol[-1] - y_ol[-2]) ** 2) + \
               np.multiply(1.-lambda_3, d_fi)           # eq.13 in [1]

    # eq.15 in [1]
    R_i = np.divide(np.multiply((2.-lambda_1), v_fi), d_fi)
    return R_i, X_fi, v_fi, d_fi

def online_T_test(y_ol):
    n = len(y_ol)
    if n == 1: return 0
    x_1, x_2 = np.mean(y_ol[:-1]), np.mean(y_ol)
    s_1, s_2 = np.var(y_ol[:-1]), np.var(y_ol)

    T = np.divide(np.abs(x_1-x_2), np.hypot(s_1,s_2))*np.sqrt(n)
    return T

def online_F_test(y_ol):
    n = len(y_ol)
    if n <= 2: return np.nan
    s_1, s_2 = np.nanvar(y_ol[:-1]), np.nanvar(y_ol)
    max_s, min_s = np.maximum(s_1, s_2), np.minimum(s_1, s_2)
    if min_s==0: return np.nan
    F = np.divide(max_s, min_s)
    return F

def make_2D_Gaussian(size, fwhm=3.):
    r"""make a 2D Gaussian kernel.
    
    find slope of the phase plane through 
    two point step size for phase correlation minimization
    
    Parameters
    ----------    
    size : list, size=(1,2), dtype=integer
        size of the resulting numpy array
    fwhm : float, size=(2,1)
        a.k.a.: full-width-half-maximum
        can be thought of as an effective radius
      
    Returns
    -------
    M : np.array, size=(size,size)   
        array with Gaussian peak in the center
    
    See Also
    --------
    phase_svd, phase_radon, phase_difference   
    
    Notes
    ----- 
    .. math:: M = e^{ -4 \ln(2) \frac{x^2 + y^2}{fwhm}}
    """
    if len(size)==1:
        size = (size, size)
    
    x = np.linspace(-(size[0]-1)/2,+(size[0]-1)/2, size[0], float)
    y = np.linspace(-(size[1]-1)/2,+(size[1]-1)/2, size[0], float)
    y = y[:,np.newaxis]
    M = np.exp(-4*np.log(2) * (x**2 + y**2) / fwhm**2)
    return M

def make_2D_Laplacian(alpha=.5):
    alpha = max(0, min(alpha, 1))
    cross_weight = np.divide(1.-alpha, 1.+alpha)
    diag_weight = np.divide(alpha, (alpha + 1.))
    center = np.divide(-4., 1.+alpha)
    M = np.array([
        [diag_weight, cross_weight, diag_weight],
        [cross_weight, center, cross_weight],
        [diag_weight, cross_weight, diag_weight]])
    return M

def sigma_filtering(y,thres=3):
    """ 3-sigma filtering

    finds the traditional statistics and filters the 1% outliers
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest
    thres : float, size=1
        multiplication factor of the sigma
        1: 68%
        2: 95%
        3: 99%
      
    Returns
    -------
    IN : np.array, size=(_,1) , dtype=boolean  
        array with classification
    
    See Also
    --------
    weighted_sigma_filtering, mad_filtering
    """
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    IN = (y > (mean_y - thres*std_y)) & \
        (y < mean_y + thres*std_y)
    return IN

def weighted_sigma_filtering(y,w,thres=3):
    """ weighted 3-sigma filtering

    finds the weighted statistics and filters with these statistics
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest
    w : np.array, size=(_,1), dtype=float
        weights to be used for the statistics calculation       
    thres : float, size=1
        multiplication factor of the sigma
        1: 68%
        2: 95%
        3: 99%
      
    Returns
    -------
    IN : np.array, size=(_,1) , dtype=boolean  
        array with classification
    
    See Also
    --------
    sigma_filtering, mad_filtering
    """
    mean_y = np.average(y,weights=w)
    std_y = np.sqrt(np.average((y-mean_y)**2, weights=w))

    IN = (y > (mean_y - thres*std_y)) & \
        (y < mean_y + thres*std_y)
    return IN

def mad_filtering(y,thres=1.4826):
    """ robust 3-sigma filtering

    finds the robust statistics and filters with these statistics
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest       
    thres : float, size=1
        multiplication factor of the median absolute difference.
        if the distribution is normal then 1.4826*thresh = sigma
      
    Returns
    -------
    IN : np.array, size=(_,1) , dtype=boolean  
        array with classification
    
    See Also
    --------
    sigma_filtering, weighted_sigma_filtering, 
    huber_filtering, cauchy_filtering
    """
    med_y = np.median(y)
    mad_y = np.median(np.abs(y - med_y))

    IN = (y > (med_y - thres*mad_y)) & \
        (y < med_y + thres*mad_y)
    return IN

def normalize_variance(y):
    """ normalize data through its variance

    project/scale data towards a uniform space, 
    by means of classical statistics 
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest       
      
    Returns
    -------
    r : np.array, size=(_,1) , dtype=float
        projected data, centered around the origin
    
    See Also
    --------
    normalize_variance_robust
    """    
    mean_y = np.mean(y)
    std_y = np.std(y)
    r = (y-mean_y)/std_y
    return r

def normalize_variance_robust(y):
    """ normalize data through its robust statistics

    project/scale data towards a uniform space, 
    by means of robust statistics 
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest       
      
    Returns
    -------
    r : np.array, size=(_,1) , dtype=float
        projected data, centered around the origin
    
    See Also
    --------
    normalize_variance
    """
    med_y = np.median(y)
    mad_y = np.median(np.abs(med_y - med_y))
    r = (y-med_y)/mad_y
    return r

def huber_filtering(y,thres=1.345,preproc='normal'):    
    """ filter with statitics following a Huber function

    finds the robust statistics and filters with these statistics
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest       
    thres : float, size=1
        multiplication factor of the median absolute difference.
        if the distribution is normal then 1.345*thresh = sigma
    preproc: string, default='normal'
        'normal'
            using classical statistics to estimate the mean
        else:
            using robust statistics
      
    Returns
    -------
    IN : np.array, size=(_,1) , dtype=boolean  
        array with classification
    
    See Also
    --------
    sigma_filtering, weighted_sigma_filtering, 
    mad_filtering, cauchy_filtering
    """
    if preproc == 'normal':
        r = normalize_variance(y)
    else:
        r = normalize_variance_robust(y)
    w = 1 / max(1, abs(r))
    IN = w<thres
    return IN

def cauchy_filtering(y,thres=2.385,preproc='normal'):
    """ filter with statitics following a Cauchy function

    finds the robust statistics and filters with these statistics
    
    Parameters
    ----------    
    y : np.array, size=(_,1), dtype=float
        data of interest       
    thres : float, size=1
        multiplication factor of the median absolute difference.
        if the distribution is normal then 2.385*thresh = sigma
    preproc: string, default='normal'
        'normal'
            using classical statistics to estimate the mean
        else:
            using robust statistics
      
    Returns
    -------
    IN : np.array, size=(_,1) , dtype=boolean  
        array with classification
    
    See Also
    --------
    sigma_filtering, weighted_sigma_filtering, 
    mad_filtering, huber_filtering
    """
    if preproc == 'normal':
        r = normalize_variance(y)
    else:
        r = normalize_variance_robust(y)    
    w = 1 / (1 + r**2)
    IN = w<thres
    return IN
    
def normalized_sampling_histogram(I):
    """ sample histogram based upon Sturges' rule, see also [StXX]_

    Parameters
    ----------
    I : np.array, dim={1,n}, dtype={integer,float}

    Returns
    -------
    values : np.array, size=(m)
        occurances within an interval
    base : np.array, size=(m)
        central value of the interval

    References
    ----------
    .. [StXX] Sturges, "The choice of a class interval". Journal of the american
       statistical association. vol.21(153) pp.65–66.
    """
    sturge = 1.6 * (np.log2(I.size) + 1)
    values, base = np.histogram(I.flatten(),
                                bins=int(np.ceil(sturge)))
    # transform to centers
    base = base[:-1] + np.diff(base)
    return values, base

def generalized_histogram_thres():
    asd
    return asd