import math
import numpy as np
import random

from scipy import ndimage # for image filters

def enhance_shadow(method, Blue, Green, Red, RedEdge, Nir, Shw):
    """
    Given a specific method, employ shadow transform
    input:   method         string            method name to be implemented, 
                                              can be one of the following:
                                              - ruffenacht
                                              - nsvi
                                              - sdi
                                              - sisdi
                                              - mpsi
                                              - shadow_index
                                              - shade_index
                                              - levin
             Blue           array (n x m)     blue band of satellite image
             Green          array (n x m)     green band of satellite image
             Red            array (n x m)     red band of satellite image
             Nir            array (n x m)     near-infrared band of satellite image   
             Shw            array (n x m)     synthetic shading from for example
                                              a digital elevation model
    output:  M              array (n x m)     shadow enhanced satellite image
    """
    
    method = method.lower()
    
    if method=='ruffenacht':
        M = ruffenacht(Blue, Green, Red, Nir)
    elif len([n for n in ['shadow','index'] if n in method])==2:
        M = shadow_index(Blue, Green, Red, Nir)
    elif len([n for n in ['shade','index'] if n in method])==2:
        M = shade_index(Blue, Green, Red, Nir)
    elif method=='nsvi':
        M = nsvi(Blue, Green, Red)
    elif method=='sdi':
        M = sdi(Blue, Green, Red)
    elif method=='sisdi':
        M = sisdi(Blue, RedEdge, Nir)
    elif method=='mpsi':
        M = mpsi(Blue, Green, Red)    
    elif method=='gevers':
        M = gevers(Blue, Green, Red) 
    elif method=='finlayson':
        M = finlayson(Blue, Green, Red)
        
    return M

# generic transforms
def pca(X):
    """ principle component analysis (PCA)
    
    input:   data           array (n x m)     array of the band image
    output:  eigen_vecs     array (m x m)     array with eigenvectors
             eigen_vals     array (m x 1)     vector with eigenvalues
    """
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    # removed, because a sample is taken:
    #    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    return eigen_vecs, eigen_vals

def pca_rgb_preparation(Red, Green, Blue, min_samp=1e4):
    OK = Blue != 0  # boolean array with data

    r,g,b = np.mean(Red[OK]), np.mean(Green[OK]), np.mean(Blue[OK])
    # shift axis origin
    Red,Green,Blue = Red-r, Green-g, Blue-b
    del r, g, b

    # sampleset
    Nsamp = int(
        min(min_samp, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp

    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')

    X = np.transpose(np.array(
        [Red[sampling], Green[sampling], Blue[sampling]]))
    return X    

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

# generic metrics
def shannon_entropy(X,band_width=100):
    num_bins = np.round(np.nanmax(X)-np.nanmin(X)/band_width)
    hist, bin_edges = np.histogram(X.flatten(), \
                                   bins=num_bins.astype(int), \
                                   density=True)
    hist = hist[hist!=0]    
    shannon = -np.sum(hist * np.log2(hist))
    return shannon

# image transforms
def gamma_adjustment(I, gamma=1.0):
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

def s_curve(x, a, b):
    """
    Transform intensity in non-linear way
    
    see Fredembach 2010
        info: EPFL Tech Report 165527
    
    input:   x              array (n x 1)     array with values
             a              array (1 x 1)     slope
             b              array (1 x 1)     intercept
    output:  fx             array (n x 1)     array with transform
    """
    fe = -a * (x - b)
    fx = np.divide(1, 1 + np.exp(fe))
    return fx

def inverse_tangent_transformation(I, zeta): # wip
    """enhance shadow transform, through classification knowledge
    
    Parameters
    ----------      
    I : np.array, size=(m,n)
        shadow enhanced imagery
        
    Returns
    -------
    I_new : np.array, size=(m,n)
        shadow enriched imagery
       
    Notes
    -----   
    [1] Li et al. "Shadow detection in remotely sensed images based on 
    self-adaptive feature selection" IEEE transactions on geoscience and 
    remote sensing vol.49(12) pp.5092-5103, 2011.
    """    
    I_new = (np.random) * np.sqrt(I)
    return I_new

# color transforms
def rgb2hcv(Blue, Green, Red):
    """transform red green blue arrays to a color space
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        Blue band of satellite image
    Green : np.array, size=(m,n)
        Green band of satellite image
    Red : np.array, size=(m,n)
        Red band of satellite image
        
    Returns
    -------
    V : np.array, size=(m,n)
        array with dominant frequency
    H : np.array, size=(m,n)
        array with amount of color
    C : np.array, size=(m,n)
        luminance   
    
    See also
    --------
    rgb2yiq, rgb2ycbcr, rgb2hsi, rgb2xyz, rgb2lms
        
    Notes
    -----
    [1] Smith, "Putting colors in order", Dr. Dobb’s Journal, pp 40, 1993.    
    [2] Tsai, "A comparative study on shadow compensation of color aerial 
    images in invariant color models", IEEE transactions in geoscience and
    remote sensing, vol. 44(6) pp. 1661--1671, 2006. 
    """
    NanBol = Blue == 0
    Blue, Green = mat_to_gray(Blue, NanBol), mat_to_gray(Green, NanBol)
    Red = Red = mat_to_gray(Red, NanBol)
    
    
    np.amax( np.dstack((Red, Green)))
    V = 0.3*(Red + Green + Blue)
    H = np.arctan2( Red-Blue, np.sqrt(3)*(V-Green))
    
    IN = abs(np.cos(H))<= 0.2
    C = np.divide(V-Green, np.cos(H))
    C2 = np.divide(Red-Blue, np.sqrt(3)*np.sin(H))
    C[IN] = C2[IN]
    return V, H, C

def rgb2yiq(Red, Green, Blue):
    """transform red, green, blue to luminance, inphase, quadrature values
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    Y : np.array, size=(m,n)
        luminance 
    I : np.array, size=(m,n)
        inphase
    Q : np.array, size=(m,n)
        quadrature
    
    See also
    --------
    rgb2hcv, rgb2ycbcr, rgb2hsi, rgb2xyz, rgb2lms
    
    Notes
    -----
    [1] Gonzalez & Woods "Digital image processing", 1992.
    """

    Y, I, Q = np.zeros(Red.shape), np.zeros(Red.shape), np.zeros(Red.shape)

    L = np.array([(+0.299, +0.587, +0.114),
                  (+0.596, -0.275, -0.321),
                  (+0.212, -0.523, +0.311)])

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            yiq = np.matmul(L, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            Y[i][j] = yiq[0]
            I[i][j] = yiq[1]
            Q[i][j] = yiq[2]
    return Y, I, Q

def rgb2ycbcr(Red, Green, Blue):   
    """transform red, green, blue arrays to luna and chroma values
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    Y : np.array, size=(m,n)
        luma
    Cb : np.array, size=(m,n)
        chroma
    Cr : np.array, size=(m,n)
        chroma
    
    See also
    --------
    rgb2hcv, rgb2yiq, rgb2hsi, rgb2xyz, rgb2lms
    
    Notes
    -----
    [1] Tsai, "A comparative study on shadow compensation of color aerial 
    images in invariant color models", IEEE transactions in geoscience and
    remote sensing, vol. 44(6) pp. 1661--1671, 2006.  
    """

    Y, Cb, Cr = np.zeros(Red.shape), np.zeros(Red.shape), np.zeros(Red.shape)

    L = np.array([(+0.257, +0.504, +0.098),
                  (-0.148, -0.291, +0.439),
                  (+0.439, -0.368, -0.071)])
    C = np.array([16, 128, 128])/2**8
    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            ycc = np.matmul(L, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            Y[i][j] = ycc[0] + C[0]
            Cb[i][j] = ycc[1] + C[1]
            Cr[i][j] = ycc[2] + C[2]
    return Y, Cb, Cr

def rgb2hsi(R, G, B):
    """transform red, green, blue arrays to hue, saturation, intensity arrays
    
    Parameters
    ----------    
    R : np.array, size=(m,n)
        red band of satellite image
    G : np.array, size=(m,n)
        green band of satellite image
    B : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    H : np.array, size=(m,n)
        Hue
    S : np.array, size=(m,n)
        Saturation
    I : np.array, size=(m,n)
        Intensity
    
    See also
    --------
    erdas2hsi, rgb2hcv, rgb2yiq, rgb2ycbcr, rgb2xyz, rgb2lms
    """

    Red = mat_to_gray(R)   # np.float64(R) / 2 ** 16
    Green = mat_to_gray(G) # np.float64(G) / 2 ** 16
    Blue = mat_to_gray(B)  # np.float64(B) / 2 ** 16

    Hue, Sat, Int = np.zeros(Red.shape()), np.zeros(Red.shape()), np.zeros(Red.shape())

    # See Tsai, IEEE Trans. Geo and RS, 2006.
    # from Pratt, 1991, Digital Image Processing, Wiley
    Tsai = np.array([(1 / 3, 1 / 3, 1 / 3),
                     (-math.sqrt(6) / 6, -math.sqrt(6) / 6, -math.sqrt(6) / 3),
                     (1 / math.sqrt(6), 2 / -math.sqrt(6), 0)])

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            hsi = np.matmul(Tsai, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            Int[i][j] = hsi[0]
            Sat[i][j] = math.sqrt(hsi[1] ** 2 + hsi[2] ** 2)
            Hue[i][j] = math.atan2(hsi[1], hsi[2])

    return Hue, Sat, Int

def erdas2hsi(Blue, Green, Red):
    """transform red, green, blue arrays to hue, saturation, intensity arrays
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

        
    Returns
    -------
    Hue : np.array, size=(m,n)
        hue
    Sat : np.array, size=(m,n)
        saturation
    Int : np.array, size=(m,n)
        intensity
        
    See also
    --------
    rgb2hsi
        
    Notes
    -----
    [1] ERDAS, "XXX", 2013. 
    """
    NanBol = Blue == 0
    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)

    Stack = np.dstack((Blue, Green, Red))
    min_Stack = np.amin(Stack, axis=2)
    max_Stack = np.amax(Stack, axis=2)
    Int = (max_Stack + min_Stack)/2

    Sat = np.copy(Blue)
    Sat[Int==0] = 0
    Sat[Int<=.5] = (max_Stack[Int<=.5] - 
                    min_Stack[Int<=.5]) / (max_Stack[Int<=.5] + 
                                           min_Stack[Int<=.5])
    Sat[Int>.5] = (max_Stack[Int>.5] - 
                   min_Stack[Int>.5]) / ( 2 - max_Stack[Int>.5] + 
                                         min_Stack[Int>.5])

    Hue = np.copy(Blue)
    Hue[Blue==max_Stack] = (1/6) *(6 
                                   + Green[Blue==max_Stack] 
                                   - Red[Blue==max_Stack])
    Hue[Green==max_Stack] = (1/6) *(4 
                                      + Red[Green==max_Stack] 
                                      - Blue[Green==max_Stack])
    Hue[Red==max_Stack] = (1/6) *(2
                                   + Blue[Red==max_Stack] 
                                   - Green[Red==max_Stack])    
    return Hue, Sat, Int

def rgb2xyz(Red, Green, Blue):
    """transform red, green, blue arrays to XYZ tristimulus values
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
    method : 
        'reinhardt'
            XYZitu601-1 axis
        'ford'
            D65 illuminant
        
    Returns
    -------
    X : np.array, size=(m,n)
    Y : np.array, size=(m,n)
    Z : np.array, size=(m,n)
    
    See also
    --------
    rgb2hcv, rgb2ycbcr, rgb2hsi, rgb2yiq, rgb2lms, xyz2lms
    
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics 
    and applications vol.21(5) pp.34-41, 2001.
    [2] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.    
    """
    
    X, Y, Z = np.copy(Red), np.copy(Red), np.copy(Red)

    if method=='ford':
        M = np.array([(0.4124564, 0.3575761, 0.1804375),
                      (0.2126729, 0.7151522, 0.0721750),
                      (0.0193339, 0.1191920, 0.9503041)])
    else:
        M = np.array([(0.5141, 0.3239, 0.1604),
                      (0.2651, 0.6702, 0.0641),
                      (0.0241, 0.1228, 0.8444)])


    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            xyz = np.matmul(M, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            X[i][j] = xyz[0]
            Y[i][j] = xyz[1]
            Z[i][j] = xyz[2]

    return X, Y, Z

def xyz2lms(X, Y, Z): 
    """transform XYZ tristimulus arrays to LMS values
    
    Parameters
    ----------    
    X : np.array, size=(m,n)
        modified XYZitu601-1 axis
    Y : np.array, size=(m,n)
        modified XYZitu601-1 axis
    Z : np.array, size=(m,n)
        modified XYZitu601-1 axis
        
    Returns
    -------
    L : np.array, size=(m,n)
    M : np.array, size=(m,n)
    S : np.array, size=(m,n)
    
    See also
    --------
    rgb2hcv, rgb2ycbcr, rgb2hsi, rgb2yiq, rgb2lms
    
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics 
    and applications vol.21(5) pp.34-41, 2001.
    """

    L, M, S = np.copy(X), np.copy(X), np.copy(X)
    
    N = np.array([(+0.3897, +0.6890, -0.0787),
                  (-0.2298, +1.1834, +0.0464),
                  (+0.0000, +0.0000, +0.0000)])

    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            lms = np.matmul(N, np.array(
                [[X[i][j]], [Y[i][j]], [Z[i][j]]]))
            L[i][j] = lms[0]
            M[i][j] = lms[1]
            S[i][j] = lms[2]
    return L, M, S

def xyz2lab(X, Y, Z, th=0.008856): 
    """transform XYZ tristimulus arrays to Lab values
    
    Parameters
    ----------    
    X : np.array, size=(m,n)
    Y : np.array, size=(m,n)
    Z : np.array, size=(m,n)
        
    Returns
    -------
    L : np.array, size=(m,n)
    a : np.array, size=(m,n)
    b : np.array, size=(m,n)
    
    See also
    --------
    rgb2xyz, xyz2lms, lms2lch
    
    Notes
    -----
    [1] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.
    [2] Silva et al. "Near real-time shadow detection and removal in aerial 
    motion imagery application" ISPRS journal of photogrammetry and remote
    sensing, vol.140 pp.104--121, 2018.
    """
    Xn,Yn,Zn = 95.047, 100.00, 108.883 # D65 illuminant
    
    YYn = Y/Yn
    
    L_1 = 116* YYn**(1/3.) 
    L_2 = 903.3 * YYn
    
    L = L_1
    L[YYn<=th] = L_2[YYn<=th]
    
    def f(tau, th):    
        fx = X**(1/3.)
        fx[X<=th] = 7.787*X[X<th] + 16/116
        return fx
    
    a = 500*( f(X/Xn, th) - f(Z/Zn, th) )
    b = 200*( f(Y/Yn, th) - f(Z/Zn, th) )
    return L, a, b

def lab2lch(L, a, b): 
    """transform XYZ tristimulus arrays to Lab values
    
    Parameters
    ----------    
    L : np.array, size=(m,n)
    a : np.array, size=(m,n)
    b : np.array, size=(m,n)
        
    Returns
    -------
    C : np.array, size=(m,n)
    h : np.array, size=(m,n)
    
    See also
    --------
    rgb2xyz, xyz2lms, xyz2lab
    
    Notes
    -----
    [1] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.
    [2] Silva et al. "Near real-time shadow detection and removal in aerial 
    motion imagery application" ISPRS journal of photogrammetry and remote
    sensing, vol.140 pp.104--121, 2018.
    """
    C = np.sqrt( a**2 + b**2)

    # calculate angle, and let it range from 0...1    
    h = ((np.arctan2(b, a) + 2*np.pi)% 2*np.pi) / 2*np.pi
    return C, h

def rgb2lms(Red, Green, Blue):
    """transform red, green, blue arrays to XYZ tristimulus values
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    L : np.array, size=(m,n)
    M : np.array, size=(m,n)
    S : np.array, size=(m,n)
    
    See also
    --------
    rgb2hcv, rgb2ycbcr, rgb2hsi, rgb2yiq, rgb2xyz, xyz2lms
    
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images", 2001.
    """
    
    L, M, S = np.copy(Red), np.copy(Red), np.copy(Red)

    I = np.array([(0.3811, 0.5783, 0.0402),
                  (0.1967, 0.7244, 0.0782),
                  (0.0241, 0.1228, 0.8444)])

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            lms = np.matmul(I, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            L[i][j] = lms[0]
            M[i][j] = lms[1]
            S[i][j] = lms[2]
    return L, M, S

def lms2lab(L, M, S):
    """transform L, M, S arrays to lab color space
    
    Parameters
    ----------    
    L : np.array, size=(m,n)
    M : np.array, size=(m,n)
    S : np.array, size=(m,n)
        
    Returns
    -------
    l : np.array, size=(m,n)
    a : np.array, size=(m,n)
    b : np.array, size=(m,n)
    
    See also
    --------
    rgb2hcv, rgb2ycbcr, rgb2hsi, rgb2yiq, rgb2xyz, xyz2lms
    
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images", 2001.
    """
    l = np.copy(L)
    a = np.copy(L)
    b = np.copy(L)

    I = np.matmul(np.array([(1/math.sqrt(3), 0, 0),
                            (0, 1/math.sqrt(6), 0),
                            (0, 0, 1/math.sqrt(2))]), \
                  np.array([(+1, +1, +1),
                            (+1, +1, -2),
                            (+1, -1, +0)]))

    for i in range(0, L.shape[0]):
        for j in range(0, L.shape[1]):
            lab = np.matmul(I, np.array(
                [[L[i][j]], [M[i][j]], [S[i][j]]]))
            l[i][j] = lab[0]
            a[i][j] = lab[1]
            b[i][j] = lab[2]
    return l, a, b

# color-based shadow functions
def shadow_index_zhou(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
        
    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform
    
    Notes
    -----
    [1] Zhou et al. "Shadow detection and compensation from remote sensing 
    images under complex urban conditions", 2021.
    """  
    
    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)    
    SI = np.divide(Cb-Y, Cb+Y)
#    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)    
#    SI = np.divide(Cr+1, Y+1)    
#    H, C, V = rgb2hcv(Red, Green, Blue)    
#    SI = np.divide(H+1, V+1)
#    Y, I, Q = rgb2yiq(Red, Green, Blue)    
#    SI = np.divide(Q+1, Y+1)
#    H, S, I = rgb2hsi(Red, Green, Blue)    
#    SI = np.divide(H+1, I+1)  
    return SI

def improved_shadow_index(Blue, Green, Red, Near):
    """transform red, green, blue arrays to improved shadow index
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    ISI : np.array, size=(m,n)
        array with shadow transform
    
    Notes
    -----
    [1] Zhou et al. "Shadow detection and compensation from remote sensing 
    images under complex urban conditions", 2021.
    """        
    SI = shadow_index_zhou(Blue, Green, Red)
    ISI = np.divide( SI + (1 - Near), SI + (1 + Near))
    
    return ISI

def shadow_enhancement_index(Blue,Green,Red,Near):
    """transform red, green, blue arrays to sei? index
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    SEI : np.array, size=(m,n)
        array with sei transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{SEI}=\frac{(B_{1}+B_{9})-(B_{3}+B_{8})}{(B_{1}+B_{9})+(B_{3}+B_{8})} 
    
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """   
    SEI = np.divide( (Blue + Near)-(Green + Red), (Blue + Near)+(Green + Red))
    return SEI

def false_color_shadow_difference_index(Green,Red,Near):
    """transform red and near infrared arrays to shadow index
    
    Parameters
    ----------    
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    FCSI : np.array, size=(m,n)
        array with shadow enhancement
        
    See also
    --------
    normalized_sat_value_difference_index   
    
    Notes
    -----
    [1] Teke et al. "Multi-spectral false color shadow detection" Proceedings
    of the ISPRS conference on photogrammetric image analysis, pp.109-119, 
    2011.
    """ 
    
    (H, S, I) = rgb2hsi(Near, Red, Green)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI
    
    SEI = np.divide( (Blue + Near)-(Green + Red), (Blue + Near)+(Green + Red))
    return SEI

def normalized_difference_water_index(Green, Near):
    """transform green and near infrared arrays NDW-index
    
    Parameters
    ----------    
    Green : np.array, size=(m,n)
        green band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    NDWI : np.array, size=(m,n)
        array with NDWI transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{NDWI}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    [1] Gao, "NDWI - a normalized difference water index for remote sensing of 
    vegetation liquid water from space" Remote sensing of environonment, 
    vol. 58 pp. 257–266, 1996
    """   
    NDWI = np.divide( (Green - Near), (Green + Near))
    return NDWI

def combinational_shadow_index(Blue,Green,Red,Near):
    """transform red, green, blue arrays to combined shadow index
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    CSI : np.array, size=(m,n)
        array with shadow transform
    
    Notes
    -----
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """
    SEI = shadow_enhancement_index(Blue,Green,Red,Near) 
    NDWI = normalized_difference_water_index(Green, Near)
    
    option_1 = SEI-Near
    option_2 = SEI-NDWI
    IN_1 = Near >= NDWI
    
    CSI = option_2
    CSI[IN_1] = option_1[IN_1]    
    return CSI    

def normalized_sat_value_difference_index(Blue, Green, Red):
    """transform red, green, blue arrays to normalized sat-value difference
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
        
    Returns
    -------
    ISI : np.array, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index
    
    Notes
    -----
    [1] Ma et al. "Shadow segmentation and compensation in high resolution 
    satellite images", Proceedings of IEEE IGARSS, pp. II-1036--II-1039, 2008.
    """    
    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI

def shadow_index_liu(Red, Green, Blue):
    """transform red, green, blue arrays to shadow index
    
    Parameters
    ----------    
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
        
    Returns
    -------
    SEI : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----     
    [1] Liu & Xie. "Study on shadow detection in high resolution remote sensing
    image of PCA and HIS model", Remote sensing technology and application, 
    vol.28(1) pp. 78--84, 2013.    
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """     
    (H,S,I) = rgb2hsi(Red, Green, Blue)    

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))
    
    SI = np.divide((PC1 - I) * (1 + S), (PC1 + I + S))
    return SI

def specthem_ratio(Red, Green, Blue):
    """transform red, green, blue arrays to specthem ratio
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image    
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    Sr : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    In the paper the logarithmic is also used further enhance the shadows,
    
    .. math:: \tilde{\text{Sr}}=\log{\text{Sr}+1} 
    
    [1] Silva et al. "Near real-time shadow detection and removal in aerial 
    motion imagery application" ISPRS journal of photogrammetry and remote
    sensing, vol.140 pp.104--121, 2018.
    """    
    X,Y,Z = rgb2xyz(Red, Green, Blue, method='Ford')
    L,a,b = xyz2lab(X,Y,Z)
    _,h = lab2lch(L,a,b)
    Sr = np.divide(h+1 , L+1)
    return Sr

def shadow_detector_index(Blue, Green, Red): 
    """transform red, green, blue arrays to shadow detector index
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image      

    Returns
    -------
    SDI : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Mustafa & Abedehafez. "Accurate shadow detection from high-resolution 
    satellite images" IEEE geoscience and remote sensing letters, vol.14(4) 
    pp. 494--498, 2017.
    """    
    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))
        
    SDI = ((1 - PC1) + 1)/(((Green - Blue) * Red) + 1)
    return SDI

def sat_int_shadow_detector_index(Blue, RedEdge, Near):
    """transform red, green, blue arrays to sat-int shadow detector index
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    RedEdge : np.array, size=(m,n)
        rededge band of satellite image      
    Near : np.array, size=(m,n)
        near infrared band of satellite image   
        
    Returns
    -------
    SISDI : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Mustafa & Abedelwahab. "Corresponding regions for shadow restoration 
    in satellite high-resolution images" International journal of remote 
    sensing, vol.39(20) pp.7014--7028, 2018.
    """                 
    (H, S, I) = erdas2hsi(Blue, RedEdge, Near)
    SISDI = S - (2*I)
    return SISDI

def mixed_property_based_shadow_index(Blue, Green, Red):
    """transform Red Green Blue arrays to Mixed property-based shadow index
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image   
        
    Returns
    -------
    MPSI : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Han et al. "A mixed property-based automatic shadow detection approach 
    for VHR multispectral remote sensing images" Applied sciences vol.8(10) 
    pp.1883, 2018.
    """    
    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    MPSI = (H - I) * (Green - Blue)
    return MPSI

def color_invariant(Blue, Green, Red):
    """transform Red Green Blue arrays to color invariant (c3)
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image   
        
    Returns
    -------
    c3 : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Gevers & Smeulders. "Color-based object recognition" Pattern 
    recognition, vol.32 pp.453--464, 1999 
    [2] Arévalo González & G. Ambrosio. "Shadow detection in colour 
    high‐resolution satellite images" International journal of remote sensing, 
    vol.29(7) pp.1945--1963, 2008
    """ 
    c3 = np.arctan2( Blue, np.amax( np.dstack((Red, Green))))
    return c3

def modified_color_invariant(Blue, Green, Red, Near): #wip
    """transform Red Green Blue arrays to color invariant (c3)
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image   
        
    Returns
    -------
    c3 : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Gevers & Smeulders. "Color-based object recognition" Pattern 
    recognition, vol.32 pp.453--464, 1999 
    [2] Besheer & G. Abdelhafiz. "Modified invariant colour model for shadow 
    detection" International journal of remote sensing, vol.36(24) 
    pp.6214--6223, 2015.
    10.1080/01431161.2015.1112930
    """     
    c3 = []
    return c3

def reinhard(Blue, Green, Red):
    """transform Red Green Blue arrays to luminance
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image  
        
    Returns
    -------
    l : np.array, size=(m,n)
        shadow enhanced band
       
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics 
    and applications vol.21(5) pp.34-41, 2001.
    """
    (L,M,S) = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    (l,alfa,beta) = lms2lab(L, M, S)
    
    return l

def finlayson(Ia, Ib, Ic, a=-1):
    """
    Transform ...
    
    see Finlayson 2009
    Entropy Minimization for Shadow Removal
    
    input:   Ia             array (n x m)     highest frequency band of satellite image
             Ib             array (n x m)     lower frequency band of satellite image
             Ic             array (n x m)     lowest frequency band of satellite image
             a              float             angle to use [in degrees]
    output:  S              array (m x m)     ...
             R              array (m x m)     ...    
    """
    Ia[Ia==0] = 1e-6    
    Ib[Ib==0] = 1e-6    
    sqIc = np.power(Ic, 1./2)
    sqIc[sqIc==0] = 1    
        
    chi1 = np.log(np.divide( Ia, sqIc))
    chi2 = np.log(np.divide( Ib, sqIc))
    
    if a==-1: # estimate angle from data if angle is not given
        (m,n) = chi1.shape
        mn = np.power((m*n),-1./3)
        # loop through angles
        angl = np.arange(0,180)
        shan = np.zeros(angl.shape)
        for i in angl:
            chi_rot = np.cos(np.radians(i))*chi1 + \
                np.sin(np.radians(i))*chi2
            band_w = 3.5*np.std(chi_rot)*mn   
            shan[i] = shannon_entropy(chi_rot, band_w)
        
        a = angl[np.argmin(shan)]
        #a = angl[np.argmax(shan)]
    # create imagery    
    S = - np.cos(np.radians(a))*chi1 - np.sin(np.radians(a))*chi2
    #a += 90
    #R = np.cos(np.radians(a))*chi1 + np.sin(np.radians(a))*chi2
    return S


def shade_index(Blue, Green, Red, Near):
    """transform blue, green, red and near infrared arrays to shade index
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image  
    Near : np.array, size=(m,n)
        near infrared band of satellite image 
        
    Returns
    -------
    l : np.array, size=(m,n)
        shadow enhanced band
       
    Notes
    -----
    Amplify dark regions by simple multiplication:
        
    .. math:: \text{SI}= \Pi_{i}^{N}{1 - B_{i}}    
        
    [1] Altena "Filling the white gap on the map: Photoclinometry for glacier 
    elevation modelling" MSc thesis TU Delft, 2012.
    """
    SI = np.prod(np.dstack([(1 - Red), (1 - Green), (1 - Blue), (1 - Near)]),
                 axis=2)
    return SI

def shadow_probabilities(Blue, Green, Red, Near, ae = 1e+1, be = 5e-1):
    """transform blue, green, red and near infrared to shade probabilities
    
    Parameters
    ----------      
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image      
    Red : np.array, size=(m,n)
        red band of satellite image  
    Near : np.array, size=(m,n)
        near infrared band of satellite image  
        
    Returns
    -------
    M : np.array, size=(m,n)
        shadow enhanced band
       
    Notes
    -----   
    [2] Fredembach & Süsstrunk. "Automatic and accurate shadow detection from 
    (potentially) a single image using near-infrared information" 
    EPFL Tech Report 165527, 2010.
    [2] Rüfenacht et al. "Automatic and accurate shadow detection using 
    near-infrared information" IEEE transactions on pattern analysis and 
    machine intelligence, vol.36(8) pp.1672-1678, 2013.
    """
    Fk = np.amax(np.stack((Red, Green, Blue), axis=2), axis=2)
    F = np.divide(np.clip(Fk, 0, 2), 2)  # (10), see Fredembach 2010
    L = np.divide(Red + Green + Blue, 3)  # (4), see Fredembach 2010
    del Red, Green, Blue, Fk

    Dvis = s_curve(1 - L, ae, be)
    Dnir = s_curve(1 - Near, ae, be)  # (5), see Fredembach 2010
    D = np.multiply(Dvis, Dnir)  # (6), see Fredembach 2010
    del L, Near, Dvis, Dnir

    M = np.multiply(D, (1 - F))
    return M


# recovery - normalized color composite
# histogram matching....

# TO DO:
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
# https://github.com/yaksoy/AffinityBasedMattingToolbox

# to look at:
# Modified invariant colour model for shadow detection, Besheer    