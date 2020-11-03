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
    elif method=='levin':
        M = levin(Blue, Green, Red, Nir, Shw)
        
    return M

# shadow functions
def matting_laplacian(I, Label, Mask=None, win_rad=1, eps=-10**7):
    """
    see Levin et al. 2006
    A closed form solution to natural image matting    
    """
    
    # dI = np.arange(start=-win_rad,stop=+win_rad+1)
    # dI,dJ = np.meshgrid(dI,dI)
    
    # dI = dI.ravel()
    # dJ = dJ.ravel()
    # N = dI.size
    # for i in range(dI.size):
    #     if i == 0:
    #         Istack = np.expand_dims( np.roll(I, 
    #                                          [dI[i], dJ[i]], axis=(0, 1)), 
    #                                 axis=3)
    #     else:
    #         Istack = np.concatenate((Istack, 
    #                                 np.expand_dims(np.roll(I, 
    #                                     [dI[i], dJ[i]], axis=(0, 1)),
    #                                                axis=3)), axis=3)
                                    
    # muI = np.mean(Istack, axis=3)
    # dI = Istack - np.repeat(np.expand_dims(muI, axis=3), 
    #                         N, axis=3)
    # del Istack
    # aap = np.einsum('ijkl,qrs->km',dI,dI) /(N - 1)
    # aap = np.einsum('ijkl,ijkn->ijk',dI,dI) /(N - 1)
    
    # win_siz = 2*win_rad+1
    
    
    
    
    
    # mean_kernel = np.ones((win_siz, win_siz))/win_siz**2
    
    # muI = I.copy()
    # varI = I.copy()
    # for i in range(I.shape[2]):
    #     muI[:,:,i] = ndimage.convolve(I[:,:,i], mean_kernel, mode='reflect')
    #     dI = (I[:,:,i] - muI[:,:,i])**2
    #     varI[:,:,i] = ndimage.convolve(dI, mean_kernel, mode='reflect')
    #     del dI
    
    # local mean colors
    L = 0
    return L

def levin(Blue, Green, Red, Near, Shade):
    """
    Transform Red Green Blue arrays to Hue Saturation Intensity arrays
    
    see Levin et al. 2006
    A closed form solution to natural image matting
    
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
             Near           array (n x m)     Near infrared band of satellite image
             Shade          array (n x m)     shadow estimate from DEM
    output:  alpha          array (m x m)     alpha matting channel
    """
    NanBol = Blue == 0
    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)
    Near = mat_to_gray(Near, NanBol)

    Stack = np.dstack((Blue, Green, Red, Near))
    
    L = matting_laplacian(Stack, Shade)
    alpha = 0*L
    return alpha

def rgb2hsi(R, G, B):  # pre-processing
    """
    Transform Red Green Blue arrays to Hue Saturation Intensity arrays
    input:   R              array (n x m)     Red band of satellite image
             G              array (n x m)     Green band of satellite image
             B              array (n x m)     Blue band of satellite image
    output:  Hue            array (m x m)     Hue band
             Sat            array (n x m)     Saturation band
             Int            array (n x m)     Intensity band
    """
    Red = np.float64(R) / 2 ** 16
    Green = np.float64(G) / 2 ** 16
    Blue = np.float64(B) / 2 ** 16

    Hue = np.copy(Red)
    Sat = np.copy(Red)
    Int = np.copy(Red)

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

def erdas2hsi(Blue, Green, Red):  # pre-processing
    """
    Transform Red Green Blue arrays to Hue Saturation Intensity arrays
    following ERDAS 2013 (pp.197)
    input:   Blue           array (n x m)     Blue band of satellite image
             Green        array (n x m)     Green band of satellite image
             Red           array (n x m)     Red infrared band of satellite image
    output:  Hue            array (m x m)     Hue band
             Sat            array (n x m)     Saturation band
             Int            array (n x m)     Intensity band
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

def rgb2xyz(Red, Green, Blue):  # pre-processing
    """
    Transform Red Green Blue arrays to XYZ tristimulus values
    
    see Reinhard et al. 2001
    Color Transfer between images
    
    input:   Red            array (n x m)     Red band of satellite image
             Green          array (n x m)     Green band of satellite image
             Blue           array (n x m)     Blue band of satellite image
    output:  X              array (m x m)     modified XYZitu601-1 axis
             Y              array (n x m)     modified XYZitu601-1 axis
             Z              array (n x m)     modified XYZitu601-1 axis
    """
    X = np.copy(Red)
    Y = np.copy(Red)
    Z = np.copy(Red)

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

def xyz2lms(X, Y, Z):  # pre-processing
    """
    Transform XYZ tristimulus values to  LMS arrays
    
    see Reinhard et al. 2001
    Color Transfer between images
    
    input:   X              array (m x m)     modified XYZitu601-1 axis
             Y              array (n x m)     modified XYZitu601-1 axis
             Z              array (n x m)     modified XYZitu601-1 axis
    output:  L              array (m x m)     
             M              array (n x m)     
             S              array (n x m)     
    """
    L = np.copy(X)
    M = np.copy(X)
    S = np.copy(X)

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

def rgb2lms(Red, Green, Blue):  # pre-processing
    """
    Transform Red Green Blue arrays to XYZ tristimulus values
    
    see Reinhard et al. 2001
    Color Transfer between images
    
    input:   Red            array (n x m)     Red band of satellite image
             Green          array (n x m)     Green band of satellite image
             Blue           array (n x m)     Blue band of satellite image
    output:  L              array (m x m)     modified XYZitu601-1 axis
             M              array (n x m)     modified XYZitu601-1 axis
             S              array (n x m)     modified XYZitu601-1 axis
    """
    L = np.copy(Red)
    M = np.copy(Red)
    S = np.copy(Red)

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
    l = np.copy(L)
    a = np.copy(L)
    b = np.copy(L)

    I = np.matmul(np.array([(1/math.sqrt(3), 0, 0),
                            (0, 1/math.sqrt(6), 0),
                            (0, 0, 1/math.sqrt(2))]), \
                  np.array([(+1, +1, +1),
                            (+1, +1, -2),
                            (+1, -1, +0)]))

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            lab = np.matmul(I, np.array(
                [[L[i][j]], [M[i][j]], [S[i][j]]]))
            l[i][j] = lab[0]
            a[i][j] = lab[1]
            b[i][j] = lab[2]
    return l, a, b

def pca(X):  # pre-processing
    """
    principle component analysis (PCA)
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


def mat_to_gray(I, notI):  # pre-processing or generic
    """
    Transform matix to float, omitting nodata values
    input:   I              array (n x m)     matrix of integers with data
             notI           array (n x m)     matrix of boolean with nodata
    output:  Inew           array (m x m)     linear transformed floating point
                                              [0...1]
    """
    yesI = ~notI
    Inew = np.float64(I)  # /2**16
    Inew[yesI] = np.interp(Inew[yesI],
                           (Inew[yesI].min(),
                            Inew[yesI].max()), (0, +1))
    Inew[notI] = 0
    return Inew


def nsvi(Blue, Green, Red):  # pre-processing
    """
    Transform Red Green Blue arrays to normalized saturation-value difference
    index (NSVI) following Ma et al. 2008
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
    output:  NSVDI          array (m x m)     array with shadow transform
    """
    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI

def sdi(Blue, Green, Red):  # pre-processing
    """
    Transform Red Green Blue arrays to Shadow detector index (SDI)
    following Mustafa and Abedehafez 2017
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
    output:  SDI           array (m x m)     array with shadow transform
    """
    OK = Blue != 0  # boolean array with data

    R = np.float64(Red)
    r = np.mean(Red[OK])
    G = np.float64(Green)
    g = np.mean(Green[OK])
    B = np.float64(Blue)
    b = np.mean(Blue[OK])

    Red = R - r
    Green = G - g
    Blue = B - b
    del r, g, b

    # sampleset
    Nsamp = int(
        min(1e4, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp

    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')

    X = np.transpose(np.array(
        [Red[sampling], Green[sampling], Blue[sampling]]))
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))
    
    Red = mat_to_gray(Red, not(OK))
    Green = mat_to_gray(Green, not(OK))
    Blue = mat_to_gray(Blue, not(OK))
        
    SDI = ((1 - PC1) + 1)/(((Green - Blue) * Red) + 1)
    return SDI

def sisdi(Blue, RedEdge, Near):
    """
    Transform Near RedEdge Blue arrays to 
    Saturation-intensity Shadow detector index (SISDI)
    following Mustafa et al. 2018
    input:   Blue           array (n x m)     Blue band of satellite image
             RedEdge        array (n x m)     RedEdge band of satellite image
             Near           array (n x m)     Near band of satellite image
    output:  SISDI          array (m x m)     array with shadow transform
    """       
    NanBol = Blue == 0
    Blue = mat_to_gray(Blue, NanBol)
    RedEdge = mat_to_gray(RedEdge, NanBol)
    Near = mat_to_gray(Near, NanBol)
    
    (H, S, I) = erdas2hsi(Blue, RedEdge, Near)
    SISDI = S - (2*I)
    return SISDI

def mpsi(Blue, Green, Red):  # pre-processing
    """
    Transform Red Green Blue arrays to Mixed property-based shadow index (MPSI)
    following Han et al. 2018
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
    output:  MPSI           array (m x m)     array with shadow transform
    """
    NanBol = Blue == 0
    # Green = np.float64(Green) / 2 ** 16
    # Blue = np.float64(Blue) / 2 ** 16
    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)
    
    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    MPSI = (H - I) * (Green - Blue)
    return MPSI

def gevers(Blue, Green, Red):  # pre-processing
    """
    Transform Red Green Blue arrays to color invariant (c3)
    
    following Gevers & Smeulders 1999
        Color-based object recognition
        Pattern Recognition 32 (1999) 453—464
    from Arévalo González & G. Ambrosio 2008
        Shadow detection in colour high‐resolution satellite images
        DOI: 10.1080/01431160701395302
        
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
    output:  c3             array (m x m)     array with color invariant (c3)
    """
    NanBol = Blue == 0
    # Green = np.float64(Green) / 2 ** 16
    # Blue = np.float64(Blue) / 2 ** 16
    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)
    
    c3 = np.arctan2( Blue, np.amax( np.dstack((Red, Green))))
    return c3

def shadow_index(Blue, Green, Red, Nir):  # pre-processing
    """
    Calculates the shadow index, basically the first component of PCA,
    following Hui et al. 2013
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
             Nir            array (n x m)     Near-Infrared band of satellite
                                              image
    output:  SI             array (m x m)     Shadow band
    """
    OK = Blue != 0  # boolean array with data

    Red = np.float64(Red)
    r = np.mean(Red[OK])
    Green = np.float64(Green)
    g = np.mean(Green[OK])
    Blue = np.float64(Blue)
    b = np.mean(Blue[OK])
    Near = np.float64(Nir)
    n = np.mean(Near[OK])
    del Nir

    Red = Red - r
    Green = Green - g
    Blue = Blue - b
    Near = Near - n
    del r, g, b, n

    # sampleset
    Nsamp = int(
        min(1e4, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp

    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')
    Near = Near.ravel(order='F')

    X = np.transpose(np.array(
        [Red[sampling], Green[sampling], Blue[sampling], Near[sampling]]))
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue, Near]))
    SI = np.transpose(np.reshape(PC1, OK.shape))
    return SI

def reinhard(Blue, Green, Red):  # pre-processing
    """
    transform colour space
    following Reinhardt 2001
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
    output:  l              array (m x m)     Shadow band
    """
    NanBol = Blue == 0  # boolean array with no data

    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)

    (L,M,S) = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    (l,alfa,beta) = lms2lab(L, M, S)
    
    return l,alfa,beta


def shade_index(Blue, Green, Red, Nir):  # pre-processing
    """
    Amplify dark regions by simple multiplication
    following Altena 2008
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
             Nir            array (n x m)     Near-Infrared band of satellite
    output:  SI             array (m x m)     Shadow band
    """
    NanBol = Blue == 0  # boolean array with no data

    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)
    Near = mat_to_gray(Nir, NanBol)
    del Nir

    SI = np.prod(np.dstack([(1 - Red), (1 - Green), (1 - Blue), (1 - Near)]),
                 axis=2)
    return SI


def ruffenacht(Blue, Green, Red, Nir):  # pre-processing
    """
    Transform Red Green Blue NIR to shadow intensities/probabilities
    
    following Fredembach 2010
        info: EPFL Tech Report 165527
    and Rüfenacht et al. 2013
        DOI: 10.1109/TPAMI.2013.229
    
    input:   Blue           array (n x m)     Blue band of satellite image
             Green          array (n x m)     Green band of satellite image
             Red            array (n x m)     Red band of satellite image
             Nir            array (n x m)     Near-Infrared band of satellite
    output:  M              array (m x m)     Shadow band
    """
    ae = 1e+1
    be = 5e-1

    NanBol = Blue == 0  # boolean array with no data

    Blue = mat_to_gray(Blue, NanBol)
    Green = mat_to_gray(Green, NanBol)
    Red = mat_to_gray(Red, NanBol)
    Near = mat_to_gray(Nir, NanBol)
    del Nir

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


def s_curve(x, a, b):  # pre-processing
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
    del fe, x
    return fx

# TO DO:
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
