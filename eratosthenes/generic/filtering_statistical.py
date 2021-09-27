import numpy as np

def make_2D_Gaussian(size, fwhm=3):
    """make a 2D Gaussian kernel.
    
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
    std_y = math.sqrt(np.average((y-mean_y)**2, weights=w))

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
    

