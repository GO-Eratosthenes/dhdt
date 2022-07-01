# general libraries
import warnings
import numpy as np

# image processing libraries
from scipy import ndimage, interpolate, fft, signal
from skimage.transform import radon
from skimage.measure import ransac
from sklearn.cluster import KMeans

from ..generic.filtering_statistical import make_2D_Gaussian, mad_filtering
from ..generic.handler_im import get_grad_filters

# frequency preparation
def perdecomp(img):
    """calculate the periodic and smooth components of an image
       
    Parameters
    ----------    
    img : numpy.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    per : numpy.array, size=(m,n)
        periodic component
    cor : numpy.array, size=(m,n)
        smooth component   

    References
    ----------    
    .. [1] Moisan, L. "Periodic plus smooth image decomposition", Journal of 
       mathematical imaging and vision vol. 39.2 pp. 161-179, 2011.    


    Example
    -------
    >>> import numpy as np
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,_,_,_,_ = create_sample_image_pair(d=2**7, max_range=1)
    >>> per,cor = perdecomp(im1)
    
    >>> spec1 = np.fft.fft2(per)
    """
    assert type(img)==np.ndarray, ("please provide an array")

    # don't need to process empty arrays
    if (0 in img.shape): return img, np.zeros_like(img)

    img = img.astype(float)
    if img.ndim==2:
        (m, n) = img.shape
        per = np.zeros((m, n), dtype=float)

        per[+0,:] = +img[0,:] -img[-1,:]
        per[-1, :] = -per[0, :]

        per[:,+0] = per[:,+0] +img[:,+0] -img[:,-1]
        per[:,-1] = per[:,-1] -img[:,+0] +img[:,-1]
    
    elif img.ndim==3:
        (m, n, b) = img.shape
        per = np.zeros((m, n, b), dtype=float)
        
        per[+0,:,:] = +img[0,:,:] -img[-1,:,:]
        per[-1,:,:] = -per[0,:,:]
        
        per[:,+0,:] = per[:,+0,:] +img[:,+0,:] -img[:,-1,:]
        per[:,-1,:] = per[:,-1,:] -img[:,+0,:] +img[:,-1,:]
    
    fy = np.cos( 2*np.pi*( np.arange(0,m) )/m )
    fx = np.cos( 2*np.pi*( np.arange(0,n) )/n )

    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx[0,0] = 0
    if img.ndim==3:
        Fx = np.repeat(Fx[:,:,np.newaxis], b, axis=2)
        Fy = np.repeat(Fy[:,:,np.newaxis], b, axis=2)
        cor = np.real( np.fft.ifftn( np.fft.fft2(per) *.5/ (2-Fx-Fy)))
    else:
        cor = np.real( np.fft.ifft2( np.fft.fft2(per) *.5/ (2-Fx-Fy)))
    per = img-cor
    return per, cor

def normalize_power_spectrum(Q):
    """transform spectrum to complex vectors with unit length 
       
    Parameters
    ----------    
    Q : numpy.array, size=(m,n), dtype=complex
        cross-spectrum
    
    Returns
    -------
    Qn : numpy.array, size=(m,n), dtype=complex
        normalized cross-spectrum, that is elements with unit length 

    Example
    -------
    >>> import numpy as np
    >>> from dhdt.generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    >>> Q = spec1 * np.conjugate(spec2) # fourier based image matching
    >>> Qn = normalize_spectrum(Q)
  
    """    
    assert type(Q)==np.ndarray, ("please provide an array")
    Qn = np.divide(Q, abs(Q), out=np.zeros_like(Q), where=Q!=0)
    return Qn

def make_fourier_grid(Q, indexing='ij', system='radians'):
    """
    The four quadrants of the coordinate system of the discrete Fourier 
    transform are flipped. This function gives its coordinate system as it 
    would be in a map (xy) or pixel based (ij) system.

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
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
    
    Returns
    -------
    F_1 : np,array, size=(m,n), dtype=integer
        first coordinate index of the Fourier spectrum in a map system.
    F_2 : np,array, size=(m,n), dtype=integer
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
    assert type(Q)==np.ndarray, ("please provide an array")
    (m,n) = Q.shape
    if indexing=='ij':
        (I_grd,J_grd) = np.meshgrid(np.arange(0,m)-(m//2),
                                    np.arange(0,n)-(n//2),
                                    indexing='ij')
        F_1, F_2 = I_grd/m, J_grd/n
    else:
        fy = np.flip((np.arange(0,m)-(m/2)) /m)
        fx = (np.arange(0,n,n)-(n/2)) /n
            
        F_1 = np.repeat(fx[np.newaxis,:],m,axis=0)
        F_2 = np.repeat(fy[:,np.newaxis],n,axis=1)

    if system=='radians': # what is the range of the axis
        F_1 *= 2*np.pi
        F_2 *= 2*np.pi
    elif system=='pixel':
        F_1 *= n
        F_1 *= m
    elif system=='unit':
        F_1 *= 2
        F_2 *= 2
        
    F_1, F_2 = np.fft.fftshift(F_1), np.fft.fftshift(F_2)
    return F_1, F_2

def construct_phase_plane(I, di, dj, indexing='ij'):
    """given a displacement, create what its phase plane in Fourier space

    Parameters
    ----------
    I : np.array, size=(m,n)
        image domain
    di : float
        displacment along the vertical axis
    dj : float
        displacment along the horizantal axis
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
    (m,n) = I.shape

    (I_grd,J_grd) = np.meshgrid(np.arange(0,n)-(n//2),
                                np.arange(0,m)-(m//2), \
                                indexing='ij')
    I_grd,J_grd = I_grd/m, J_grd/n

    Q_unwrap = ((I_grd*di) + (J_grd*dj) ) * (2*np.pi)   # in radians
    Q = np.cos(-Q_unwrap) + 1j*np.sin(-Q_unwrap)

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
    assert type(data)==np.ndarray, ("please provide an array")
    assert type(W)==np.ndarray, ("please provide an array")

    if data.shape[0]==data.shape[1]:
        (m,n) = data.shape
        F1,F2 = make_fourier_grid(np.zeros((m,n)),
                                  indexing='ij', system='unit')

        # transform from complex to -1...+1
        Q = np.fft.fftshift(np.angle(data) / np.pi) #(2*np.pi))

        data_list = np.vstack((F1.flatten(),
                               F2.flatten(),
                               Q.flatten() )).T
        if W.size>0: # remove masked data
            data_list = data_list[W.flatten()==1,:]
    elif W.size!= 0:
        data_list = data[W.flatten()==1,:]
    else:
        data_list = data
    return data_list

def construct_phase_values(IJ, di, dj, indexing='ij', system='radians'): #todo implement indexing
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

    if system=='radians': # -pi ... +pi
        scaling = 1
    elif system=='unit': # -1 ... +1
        scaling = np.pi
    else: # normalized -0.5 ... +0.5
        scaling = 2*np.pi

    Q_unwrap = ((IJ[:,0]*di) + (IJ[:,1]*dj) )*scaling
    Q = np.cos(-Q_unwrap) + 1j*np.sin(-Q_unwrap)
    return Q

def gradient_fourier(I):
    """ spectral derivative estimation

    Parameters
    ----------
    I : numpy.array, size=(m,n), dtype=float
        intensity array

    Returns
    -------
    dIdx_1, dIdx_2 : numpy.array, size=(m,n), dtype=float
        first order derivative along both axis
    """
    m, n = I.shape[0], I.shape[1]

    x_1 = np.linspace(0, 2*np.pi, m, endpoint=False)
    dx = x_1[1] - x_1[0]

    k_1, k_2 = 2*np.pi*np.fft.fftfreq(m, dx), 2*np.pi*np.fft.fftfreq(n, dx)
    K_1, K_2 = np.meshgrid(k_1, k_2, indexing='ij')

    # make sure Gibbs phenomena are reduced
    I_g = perdecomp(I)[0]
    W_1, W_2 = hanning_window(K_1), hanning_window(K_2)

    dIdx_1 = np.fft.ifftn(W_1*K_1*1j * np.fft.fftn(I)).real
    dIdx_2 = np.fft.ifftn(W_2*K_2*1j * np.fft.fftn(I)).real
    return dIdx_1, dIdx_2

# frequency matching filters
def raised_cosine(I, beta=0.35):
    """ raised cosine filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    .. [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
       registration of images." IEEE Transactions on geoscience and remote 
       sensing. vol. 39(10) pp. 2235-2243, 2001.
    .. [2] Leprince, et.al. "Automatic and precise orthorectification, 
       coregistration, and subpixel correlation of satellite images, 
       application to ground deformation measurements", IEEE Transactions on 
       geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.

    Example
    -------
    >>> import numpy as np
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    
    >>> rc1 = raised_cosine(spec1, beta=0.35)
    >>> rc2 = raised_cosine(spec2, beta=0.50)
    
    >>> Q = (rc1*spec1) * np.conjugate((rc2*spec2)) # Fourier based image matching
    >>> Qn = normalize_spectrum(Q)    
    """ 
    assert type(I)==np.ndarray, ("please provide an array")
    (m, n) = I.shape
   
    Fx,Fy = make_fourier_grid(I, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy) # radius
    # filter formulation 
    Hamm = np.cos( (np.pi/(2*beta)) * (R - (.5-beta)))**2
    selec = np.logical_and((.5 - beta) <= R , R<=.5)
    
    # compose filter
    W = np.zeros((m,n))
    W[(.5 - beta) > R] = 1
    W[selec] = Hamm[selec]
    return W 

def hamming_window(I):
    """ create two-dimensional Hamming filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.hamming(m), np.hamming(n)))
    W = np.fft.fftshift(W)
    return W 

def hanning_window(I):
    """ create two-dimensional Hanning filter, also known as Cosine Bell
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.hanning(m), np.hanning(n)))
    W = np.fft.fftshift(W)
    return W 

def blackman_window(I):
    """ create two-dimensional Blackman filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.blackman(m), np.blackman(n)))
    W = np.fft.fftshift(W)
    return W 

def kaiser_window(I, beta=14.):
    """ create two dimensional Kaiser filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.kaiser(m, beta), np.kaiser(n, beta)))
    W = np.fft.fftshift(W)
    return W 

def low_pass_rectancle(I, r=0.50):
    """ create hard two dimensional low-pass filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    .. [1] Takita et al. "High-accuracy subpixel image registration based on 
       phase-only correlation" IEICE transactions on fundamentals of 
       electronics, communications and computer sciences, vol.86(8) 
       pp.1925-1934, 2003.
    """
    assert type(I)==np.ndarray, ("please provide an array")
    Fx,Fy = make_fourier_grid(I, indexing='xy', system='normalized')

    # filter formulation 
    W = np.logical_and(np.abs(Fx)<=r, np.abs(Fy)<=r) 
    return W 

def low_pass_pyramid(I, r=0.50):
    """ create low-pass two-dimensional filter with pyramid shape
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    .. [1] Takita et al. "High-accuracy subpixel image registration based on 
       phase-only correlation" IEICE transactions on fundamentals of 
       electronics, communications and computer sciences, vol.86(8) 
       pp.1925-1934, 2003.    
    """
    assert type(I)==np.ndarray, ("please provide an array")
    R = low_pass_rectancle(I, r)
    W = signal.convolve2d(R.astype(float), R.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_bell(I, r=0.50):
    """ create low-pass two-dimensional filter with a bell shape
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : numpy.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_pyramid
    
    References
    ----------
    .. [1] Takita et al. "High-accuracy subpixel image registration based on 
       phase-only correlation" IEICE transactions on fundamentals of 
       electronics, communications and computer sciences, vol.86(8) 
       pp.1925-1934, 2003.
    """
    assert type(I)==np.ndarray, ("please provide an array")
    R1 = low_pass_rectancle(I, r)
    R2 = low_pass_pyramid(I, r)
    W = signal.convolve2d(R1.astype(float), R2.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_circle(I, r=0.50):
    """ create hard two-dimensional low-pass filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    Fx,Fy = make_fourier_grid(I, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy) # radius
    # filter formulation 
    W = R<=r
    return W


def low_pass_ellipse(I, r1=0.50, r2=0.50):
    """ create hard low-pass filter

    Parameters
    ----------
    I : numpy.array, size=(m,n)
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
    sc = r1/r2
    assert type(I) == np.ndarray, ("please provide an array")
    Fx, Fy = make_fourier_grid(I, indexing='xy', system='normalized')
    R = np.hypot(sc*Fx, Fy)  # radius
    # filter formulation
    W = R <= r1
    return W

def high_pass_circle(I, r=0.50):
    """ create hard high-pass filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
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
    assert type(I)==np.ndarray, ("please provide an array")
    Fx,Fy = make_fourier_grid(I, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy) # radius
    # filter formulation 
    W = R>=r
    return W

def cosine_bell(I):
    """ cosine bell filter
    
    Parameters
    ----------    
    I : numpy.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : numpy.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    raised_cosine     
    """
    assert type(I)==np.ndarray, ("please provide an array")
    Fx,Fy = make_fourier_grid(I, indexing='xy', system='normalized')
    R = np.hypot(Fx, Fy) # radius
    
    # filter formulation 
    W = .5*np.cos(2*R*np.pi) + .5
    W[R>.5] = 0
    return W

def cross_shading_filter(Q): #, az_1, az_2): #todo
    assert type(Q)==np.ndarray, ("please provide an array")
    (m,n) = Q.shape    
    Coh = local_coherence(np.fft.fftshift(Q))
    R = np.fft.fftshift(low_pass_circle(Q, r=0.50))
    Coh[R==0] = 0
    
    theta = np.linspace(0., 180., max(m,n), endpoint=False)
    S = radon(Coh, theta)/m # sinogram
    
    # classify
    s = S[m//2,:]
    min_idx,max_idx = np.argmin(s), np.argmax(s)
    # create circle
    x,y = np.sin(np.radians(2*theta)), np.cos(np.radians(2*theta))
    coh_circle = np.vstack((x,y,(s+.1)**2)).T
    kmeans = KMeans(n_clusters=2, \
                    init=np.array([coh_circle[min_idx,:], 
                                  coh_circle[max_idx,:]]),
                    n_init=1
                    ).fit(coh_circle)
    grouping = kmeans.labels_ #.astype(np.float)
    OUT = grouping==grouping[min_idx]     
       

    Fx,Fy = make_fourier_grid(Q)    
    Theta = np.round(np.degrees(np.arctan2(Fx,Fy) % np.pi)/360 *m) *360 /m 
    W = np.isin(Theta, theta[~OUT])
    return W

# cross-spectral and frequency signal metrics for filtering
def thresh_masking(S, m=1e-4, s=10):
    """ mask significant intensities in spectrum
    
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
    .. [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
        registration of images." IEEE Transactions on geoscience and remote 
        sensing vol. 39(10) pp. 2235-2243, 2001.
    .. [2] Leprince, et.al. "Automatic and precise orthorectification, 
        coregistration, and subpixel correlation of satellite images, 
        application to ground deformation measurements", IEEE Transactions on 
        geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.    
    """    
    assert type(S)==np.ndarray, ("please provide an array")
    S_bar = np.abs(S)
    th = np.max(S_bar)*m
    
    # compose filter
    M = S_bar>th
    M = ndimage.median_filter(M, size=(s,s))
    return M

def adaptive_masking(S, m=.9):
    """ mark significant intensities in spectrum
    
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
    .. [1] Leprince, et.al. "Automatic and precise orthorectification, 
        coregistration, and subpixel correlation of satellite images, 
        application to ground deformation measurements", IEEE Transactions on 
        geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.    
    """ 
    assert type(S)==np.ndarray, ("please provide an array")
    np.seterr(divide = 'ignore') 
    LS = np.log10(np.abs(S))
    LS[np.isinf(LS)] = np.nan
    np.seterr(divide = 'warn') 
    
    NLS = LS - np.nanmax(LS.flatten())
    mean_NLS = m*np.nanmean(NLS.flatten())

    M = NLS>mean_NLS
    return M

def gaussian_mask(S):
    """ mask significant intensities in spectrum
    
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
    .. [1] Eckstein et al. "Phase correlation processing for DPIV 
       measurements", Experiments in fluids, vol.45 pp.485-500, 2008.

    Example
    --------
    >>> import numpy as np
    >>> from dhdt.generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)      
    >>> Q = spec1 * np.conjugate(spec2) # Fourier based image matching
    >>> Qn = normalize_spectrum(Q)

    >>> W = gaussian_mask(Q)
    >>> C = np.fft.ifft2(W*Q)
    """
    assert type(S)==np.ndarray, ("please provide an array")
    (m,n) = S.shape
    Fx,Fy = make_fourier_grid(S, indexing='xy', system='normalized')
    
    M = np.exp(-.5*((Fy*np.pi)/m)**2) * np.exp(-.5*((Fx*np.pi)/n)**2)
    return M