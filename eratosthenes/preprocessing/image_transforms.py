import numpy as np

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