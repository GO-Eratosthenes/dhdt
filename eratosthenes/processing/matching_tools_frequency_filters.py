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
    img : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    per : np.array, size=(m,n)
        periodic component
    cor : np.array, size=(m,n)
        smooth component   
    
    Notes
    -----    
    [1] Moisan, L. "Periodic plus smooth image decomposition", Journal of 
    mathematical imaging and vision vol. 39.2 pp. 161-179, 2011.    
    """
    img = img.astype(float)
    if img.ndim==2:
        (m, n) = img.shape
        per = np.zeros((m, n), dtype=float)
        
        per[+0,:] = +img[0,:] -img[-1,:]
        per[-1,:] = -per[0,:]
        
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
    return (per, cor)    

def normalize_power_spectrum(Q):
    Qn = np.divide(Q, abs(Q), out=np.zeros_like(Q), where=Q!=0)
    return Qn

def make_fourier_grid(Q):
    (m,n) = Q.shape
    fy = 2*np.pi*(np.arange(0,m)-(m/2)) /m
    fx = 2*np.pi*(np.arange(0,n)-(n/2)) /n
        
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx = np.fft.fftshift(Fx)
    Fy = np.fft.fftshift(Fy)
    return Fx, Fy
    
# frequency matching filters
def raised_cosine(I, beta=0.35):
    """ raised cosine filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    beta : float, default=0.35
        roll-off factor
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    tpss   
    
    Notes
    -----    
    [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
    registration of images." IEEE Transactions on geoscience and remote sensing 
    vol. 39(10) pp. 2235-2243, 2001.
    [2] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    Hamm = np.cos( (np.pi/(2*beta)) * (R - (.5-beta)))**2
    selec = np.logical_and((.5 - beta) <= R , R<=.5)
    
    # compose filter
    W = np.zeros((m,n))
    W[(.5 - beta) > R] = 1
    W[selec] = Hamm[selec]
    return W 

def hamming_window(I):
    """ create hanning filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, blackman_window,
    hamming_window
        
    """
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.hamming(m), np.hamming(n)))
    W = np.fft.fftshift(W)
    return W 

def hanning_window(I):
    """ create hanning filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, blackman_window,
    hamming_window
        
    """
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.hanning(m), np.hanning(n)))
    W = np.fft.fftshift(W)
    return W 

def blackman_window(I):
    """ create blackman filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, hamming_window, 
    hanning_window
        
    """
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.blackman(m), np.blackman(n)))
    W = np.fft.fftshift(W)
    return W 

def kaiser_window(I, beta=14.):
    """ create kaiser filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    beta: float
        0.0 - rectangular window
        5.0 - similar to Hamming window
        6.0 - similar to Hanning window
        8.6 - similar to Blackman window
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle, hamming_window, 
    hanning_window
        
    """
    (m, n) = I.shape
    W = np.sqrt(np.outer(np.kaiser(m, beta), np.kaiser(n, beta)))
    W = np.fft.fftshift(W)
    return W 

def low_pass_rectancle(I, r=0.50):
    """ create hard low-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_circle, low_pass_pyramid, low_pass_bell
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.
    """
    
    (m, n) = I.shape
    fy = 2*np.mod(.5 + np.arange(0,m)/m , 1) -1 # fft shifted coordinate frame
    fx = 2*np.mod(.5 + np.arange(0,n)/n , 1) -1
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    # filter formulation 
    W = np.logical_and(np.abs(Fx)<=r, np.abs(Fy)<=r) 
    return W 

def low_pass_pyramid(I, r=0.50):
    """ create low-pass filter with pyramid shape
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_bell
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.    
    """
    R = low_pass_rectancle(I, r)
    W = signal.convolve2d(R.astype(float), R.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_bell(I, r=0.50):
    """ create low-pass filter with a bell shape
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_pyramid
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.
    """
    R1 = low_pass_rectancle(I, r)
    R2 = low_pass_pyramid(I, r)
    W = signal.convolve2d(R1.astype(float), R2.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_circle(I, r=0.50):
    """ create hard low-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle
        
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    W = R<=r
    return W 

def high_pass_circle(I, r=0.50):
    """ create hard high-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, low_pass_circle
        
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    W = R>=r
    return W

def cosine_bell(I):
    """ cosine bell filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    raised_cosine     
    """
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    
    # filter formulation 
    W = .5*np.cos(2*R*np.pi) + .5
    W[R>.5] = 0
    return W

def cross_shading_filter(Q): #, az_1, az_2): # wip
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
    S : np.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
    m : float, default=1e-3
        cut-off intensity in respect to maximum
    s : integer, default=10
        kernel size of the median filter
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=bool
        frequency mask
    
    See Also
    --------
    tpss   
    
    Notes
    ----- 
    [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
    registration of images." IEEE Transactions on geoscience and remote sensing 
    vol. 39(10) pp. 2235-2243, 2001.
    [2] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """    
    Sbar = np.abs(S)
    th = np.max(Sbar)*m
    
    # compose filter
    M = Sbar>th
    M = ndimage.median_filter(M, size=(s,s))
    return M

def adaptive_masking(S, m=.9):
    """ mark significant intensities in spectrum
    
    Parameters
    ----------    
    S : np.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
    m : float, default=.9
        cut-off intensity in respect to maximum
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=bool
        frequency mask
    
    See Also
    --------
    tpss   
    
    Notes
    ----- 
    [1] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """    
    LS = np.log10(np.abs(S))
    LS[np.isinf(LS)] = np.nan
    NLS = LS - np.nanmax(LS.flatten())
    mean_NLS = m*np.nanmean(NLS.flatten())

    M = NLS>mean_NLS
    return M

def local_coherence(Q, ds=1): 
    """ estimate the local coherence of a spectrum
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        array with cross-spectrum
    ds : integer, default=10
        kernel radius to describe the neighborhood
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=float
        vector coherence from no to ideal, i.e.: 0...1
    
    See Also
    --------
    thresh_masking   
      
    """
    diam = 2*ds+1
    C = np.zeros_like(Q)
    (isteps,jsteps) = np.meshgrid(np.linspace(-ds,+ds,2*ds+1, dtype=int), \
                          np.linspace(-ds,+ds,2*ds+1, dtype=int))
    IN = np.ones(diam**2, dtype=bool)  
    IN[diam**2//2] = False
    isteps,jsteps = isteps.flatten()[IN], jsteps.flatten()[IN]
    
    for idx, istep in enumerate(isteps):
        jstep = jsteps[idx]
        Q_step = np.roll(Q, (istep,jstep))
        # if the spectrum is normalized, then no division is needed
        C += Q*np.conj(Q_step)
    C = np.abs(C)/np.sum(IN)
    return C

def gaussian_mask(S):
    """ mask significant intensities in spectrum
    
    Parameters
    ----------    
    S : np.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=bool
        frequency mask
    
    See Also
    --------
    tpss   
    
    Notes
    ----- 
    [1] Eckstein et al. "Phase correlation processing for DPIV measurements",
    Experiments in fluids, vol.45 pp.485-500, 2008.
    """
    (m,n) = S.shape
    Fx,Fy = make_fourier_grid(S)
    
    M = np.exp(-.5*((Fy*np.pi)/m)**2) * np.exp(-.5*((Fx*np.pi)/n)**2)
    return M