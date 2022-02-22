import numpy as np

def mat_to_gray(I, notI=None):
    """ transform matix array  to float, omitting nodata values

    Parameters
    ----------
    I : np.array, size=(m,n), dtype={integer,float}
        matrix of integers with data
    notI : np.array, size=(m,n), dtype=bool
        matrix assigning which is a no data. The default is None.

    Returns
    -------
    Inew : np.array, size=(m,n), dtype=float, range=0...1
        array with normalized intensity values

    See Also
    --------
    gamma_adjustment, log_adjustment, s_curve, inverse_tangent_transformation

    Example
    -------
    >>> import numpy as np
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> np.max(img)
    255
    >>> gray = mat_to_gray(img[:,:,0], img[:,:,0]==0)
    >>> np.max(gray)
    1.0
    """
    if notI is None:
        yesI = np.ones(I.shape, dtype=bool)
        notI = ~yesI
    else:
        yesI = ~notI
    Inew = np.float64(I)  # /2**16
    if np.ptp(I) != 0:
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
    I : np.array, size=(m,n), dtype=integer
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
        I_new = np.take(look_up_table, I, out=np.zeros_like(I))
    elif I.dtype.type == np.uint16:
        radio_range = 2**16-1
        look_up_table = np.array([((i / radio_range) ** gamma) * radio_range
                                  for i in np.arange(0, radio_range+1)]
                                 ).astype("uint16")
        I_new = np.take(look_up_table, I, out=np.zeros_like(I))
    else: # float
        radio_range = 2**16-1
        I = np.round(I*radio_range).astype(np.uint16)
        look_up_table = np.array([((i / radio_range) ** gamma) * radio_range
                                  for i in np.arange(0, radio_range+1)]
                                 ).astype("uint16")
        I_new = np.take(look_up_table, I, out=np.zeros_like(I))
        I_new = (I_new/radio_range).astype(np.float64)
    return I_new

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
    fx : np.array, size=(m,n)
        array with transform

    References
    ----------
    [1] Fredembach & Süsstrunk. "Automatic and accurate shadow detection from
    (potentially) a single image using near-infrared information"
    EPFL Tech Report 165527, 2010.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0,1,.001)
    >>> fx = s_curve(x)
    >>> plt.plot(x,fx), plt.show()
    shows the mapping function
    """
    fe = -a * (x - b)
    fx = np.divide(1, 1 + np.exp(fe))
    return fx

def hyperbolic_adjustment(I, intercept):
    """ transform intensity through an hyperbolic sine function

    Parameters
    ----------
    I : np.array, size=(m,n), dtype=float
        array with intensity values

    Returns
    -------
    I_new : np.array, size=(m,n), dtype=float, range=-1...+1
        array with transform
    """

    bottom = np.min(I)
    ptp = np.ptp(np.array([bottom, intercept]))

    I_new = np.arcsinh((I-intercept)*(10/ptp)) / 3
    I_new = np.minimum(I_new, 1)
    return I_new

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

    See Also
    --------
    histogram_equalization
    """
    if np.issubdtype(img.dtype, np.integer):
        hist = get_histogram(img)
        cdf = np.cumsum(hist)

        # determine the normalization values for each unit of the cdf
        if img.dtype.type==np.uint8:
            scaling = np.uint8(cdf * 2**8)
        else:
            scaling = np.uint16(cdf * 2**16)

        # normalize the normalization values
        new_img = scaling[img]
    else: # when array is a float
        cdf = np.argsort(img.flatten(order='C'))
        (m, n) = img.shape
        new_img = np.zeros_like(img).flatten()
        new_img[cdf] = np.linspace(1,m*n,m*n)/(m*n)
        new_img = np.reshape(new_img,(m,n), 'C')
    return new_img

def histogram_equalization(img, img_ref):
    """ transform the intensity values of an array, so they have a similar
    histogram as the reference.

    Parameters
    ----------
    img : np.array, ndim=2
        asd
    img_ref : np.array, ndim=2

    Returns
    -------
    img : np.array, ndim=2
        transformed image

    See Also
    --------
    normalize_histogram
    """
    mn = img.shape
    img, img_ref = img.flatten(), img_ref.flatten()

    # make resistant to NaN's
    IN, IN_ref = ~np.isnan(img), ~np.isnan(img_ref)
    # do not do processing if there is a lack of data
    if (np.sum(IN)<4) or (np.sum(IN_ref)<4): return img.reshape(mn)

    new_img = np.zeros_like(img)
    val_img,val_idx,cnt_img = np.unique(img[IN],
                                        return_counts=True, return_inverse=True)
    val_ref,cnt_ref = np.unique(img_ref[IN_ref], return_counts=True)

    qnt_img,qnt_ref = np.cumsum(cnt_img).astype(np.float64), \
                      np.cumsum(cnt_ref).astype(np.float64)
    qnt_img /= qnt_img[-1] # normalize
    qnt_ref /= qnt_ref[-1]

    # implement the mapping
    intp_img = np.interp(qnt_img, qnt_ref, val_ref)

    new_img[IN] = intp_img[val_idx]
    new_img[~IN] = np.nan
    return new_img.reshape(mn)

def high_pass_im(Im, radius=10):
    (m, n) = Im.shape
    If = np.fft.fft2(Im)

    # fourier coordinate system
    (I, J) = np.mgrid[:m, :n]
    Hi = np.fft.fftshift(np.divide(1 - np.exp(-((I - m/2)**2 + (J - n/2)**2)),
                                   (2 * radius)**2)
                         )
    Inew = np.real(np.fft.ifft2(If * Hi))
    return Inew

