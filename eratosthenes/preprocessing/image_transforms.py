import numpy as np

def mat_to_gray(I, notI=None):
    """
    Transform matix to float, omitting nodata values
    input:   I              array (n x m)     matrix of integers with data
             notI           array (n x m)     matrix of boolean with nodata
    output:  Inew           array (m x m)     linear transformed floating point
                                              [0...1]
    """
    if notI is None:
        yesI = np.ones(I.shape, dtype=bool)
        notI = ~yesI
    else:
        yesI = ~notI
    Inew = np.float64(I)  # /2**16
    Inew[yesI] = np.interp(Inew[yesI],
                           (Inew[yesI].min(),
                            Inew[yesI].max()), (0, +1))
    Inew[notI] = 0
    return Inew

def gamma_adjustment(I, gamma=1.0):
    """ transform intensity in non-linear way
    
    enhances high intensities when gamma>1
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensity values
    gamma : float
        power law parameter
        
    Returns
    -------
    I_new : np.array, size=(_,_)
        array with transform
    """
    if I.dtype.type == np.uint8:
        radio_range = 2**8-1
        look_up_table = np.array([((i / radio_range) ** gamma) * radio_range
                                  for i in np.arange(0, radio_range+1)]
                                 ).astype("uint8")
    else:
        radio_range = 2**16-1
        look_up_table = np.array([((i / radio_range) ** gamma) * radio_range
                                  for i in np.arange(0, radio_range+1)]
                                 ).astype("uint16")
    return look_up_table[I]

def log_adjustment(I):
    """ transform intensity in non-linear way
    
    enhances low intensities
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensity values
        
    Returns
    -------
    I_new : np.array, size=(_,_)
        array with transform
    """
    if I.dtype.type == np.uint8:
        radio_range = 2**8-1
    else:
        radio_range = 2**16-1
    
    c = radio_range/(np.log(1 + np.amax(I)))
    I_new = c * np.log(1 + I)
    if I.dtype.type == np.uint8:
        I_new = I_new.astype("uint8")
    else:
        I_new = I_new.astype("uint16")
    return I_new

def s_curve(x, a=10, b=.5):
    """ transform intensity in non-linear way
    
    enhances high and low intensities
    
    Parameters
    ----------    
    x : np.array, size=(m,n)
        array with intensity values
    a : float
        slope
    b : float
        intercept
        
    Returns
    -------
    fx : np.array, size=(_,_)
        array with transform
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0,1,.001)
    >>> y = s_curve(x)
    >>> plt.plot(x,fx), plt.show()   
    shows the mapping function   
    
    Notes
    ----- 
    [1] Fredembach & Süsstrunk. "Automatic and accurate shadow detection from 
    (potentially) a single image using near-infrared information" 
    EPFL Tech Report 165527, 2010.
    """
    fe = -a * (x - b)
    fx = np.divide(1, 1 + np.exp(fe))
    return fx

def inverse_tangent_transformation(x):
    """ enhances high and low intensities
    
    Parameters
    ----------    
    x : np.array, size=(m,n)
        array with intensity values
        
    Returns
    -------
    fx : np.array, size=(_,_)
        array with transform
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0,1,.001)
    >>> y = inverse_tangent_transformation(x)
    >>> plt.plot(x,fx), plt.show()   
    shows the mapping function   
    """   
    fx = np.arctan(x)
    return fx

def get_histogram(img):
    """ calculate the normalized histogram of an image

    Parameters
    ----------
    img : np.array, size=(m,n)
        gray scale image array

    Returns
    -------
    hist : np.array, size=(2**8|2**16,2)
        histogram array
    """
    m, n = img.shape
    if img.dtype.type==np.uint8:
        hist = [0.] * 2**8
    else:
        hist = [0.] * 2**16
        
    for i in range(m):
      for j in range(n):
        hist[img[i, j]]+=1
    hist = np.array(hist)/(m*n)
    return hist

def normalize_histogram(img):
    """ transform image to have a uniform distribution

    Parameters
    ----------
    img : np.array, size=(m,n), dtype={uint8,unit16}
        image array

    Returns
    -------
    new_img : np.array, size=(m,n), dtype={uint8,unit16}
        transfrormed image

    """
    hist = get_histogram(img)
    cdf = np.cumsum(hist) 
    
    # determine the normalization values for each unit of the cdf    
    if img.dtype.type==np.uint8:
        scaling = np.uint8(cdf * 2**8)
    else:
        scaling = np.uint16(cdf * 2**16)    
    
    # normalize the normalization values
    new_img = scaling[img]
    return new_img



