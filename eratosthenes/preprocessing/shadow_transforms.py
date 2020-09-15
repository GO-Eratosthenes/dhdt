import math
import numpy as np
import random


# shadow functions
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
    following Ruffenacht
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
    return I


def nsvi(B2, B3, B4):  # pre-processing
    """
    Transform Red Green Blue arrays to normalized saturation-value difference
    index (NSVI) following Ma et al. 2008
    input:   B2             array (n x m)     Blue band of satellite image
             B3             array (n x m)     Green band of satellite image
             B4             array (n x m)     Red band of satellite image
    output:  NSVDI          array (m x m)     array with shadow transform
    """
    (H, S, I) = rgb2hsi(B4, B3, B2)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI


def mpsi(B2, B3, B4):  # pre-processing
    """
    Transform Red Green Blue arrays to Mixed property-based shadow index (MPSI)
    following Han et al. 2018
    input:   B2             array (n x m)     Blue band of satellite image
             B3             array (n x m)     Green band of satellite image
             B4             array (n x m)     Red band of satellite image
    output:  MPSI           array (m x m)     array with shadow transform
    """
    Green = np.float64(B3) / 2 ** 16
    Blue = np.float64(B2) / 2 ** 16

    (H, S, I) = rgb2hsi(B4, B3, B2)  # create HSI bands
    MPSI = (H - I) * (Green - Blue)
    return MPSI


def shadowIndex(R, G, B, NIR):  # pre-processing
    """
    Calculates the shadow index, basically the first component of PCA,
    following Hui et al. 2013
    input:   R              array (n x m)     Red band of satellite image
             G              array (n x m)     Green band of satellite image
             B              array (n x m)     Blue band of satellite image
             NIR            array (n x m)     Near-Infrared band of satellite
                                              image
    output:  SI             array (m x m)     Shadow band
    """
    OK = R != 0  # boolean array with data

    Red = np.float64(R)
    r = np.mean(Red[OK])
    Green = np.float64(G)
    g = np.mean(Green[OK])
    Blue = np.float64(B)
    b = np.mean(Blue[OK])
    Near = np.float64(NIR)
    n = np.mean(Near[OK])
    del R, G, B, NIR

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


def shadeIndex(R, G, B, NIR):  # pre-processing
    """
    Amplify dark regions by simple multiplication
    following Altena 2008
    input:   R              array (n x m)     Red band of satellite image
             G              array (n x m)     Green band of satellite image
             B              array (n x m)     Blue band of satellite image
             NIR            array (n x m)     Near-Infrared band of satellite
                                              image
    output:  SI             array (m x m)     Shadow band
    """
    NanBol = R == 0  # boolean array with no data

    Blue = mat_to_gray(B, NanBol)
    Green = mat_to_gray(G, NanBol)
    Red = mat_to_gray(R, NanBol)
    Near = mat_to_gray(NIR, NanBol)
    del R, G, B, NIR

    SI = np.prod(np.dstack([(1 - Red), (1 - Green), (1 - Blue), (1 - Near)]),
                 axis=2)
    return SI


def ruffenacht(R, G, B, NIR):  # pre-processing
    """
    Transform Red Green Blue NIR to shadow intensities/probabilities
    following Fedembach 2010
    input:   R              array (n x m)     Red band of satellite image
             G              array (n x m)     Green band of satellite image
             B              array (n x m)     Blue band of satellite image
             NIR            array (n x m)     Near-Infrared band of satellite
                                              image
    output:  M              array (m x m)     Shadow band
    """
    ae = 1e+1
    be = 5e-1

    NanBol = R == 0  # boolean array with no data

    Blue = mat_to_gray(B, NanBol)
    Green = mat_to_gray(G, NanBol)
    Red = mat_to_gray(R, NanBol)
    Near = mat_to_gray(NIR, NanBol)
    del R, G, B, NIR

    Fk = np.amax(np.stack((Red, Green, Blue), axis=2), axis=2)
    F = np.divide(np.clip(Fk, 0, 2), 2)  # (10), see Fedembach 2010
    L = np.divide(Red + Green + Blue, 3)  # (4), see Fedembach 2010
    del Red, Green, Blue, Fk

    Dvis = S_curve(1 - L, ae, be)
    Dnir = S_curve(1 - Near, ae, be)  # (5), see Fedembach 2010
    D = np.multiply(Dvis, Dnir)  # (6), see Fedembach 2010
    del L, Near, Dvis, Dnir

    M = np.multiply(D, (1 - F))
    return M


def S_curve(x, a, b):  # pre-processing
    """
    Transform intensity in non-linear way
    see Fedembach 2010
    input:   x              array (n x 1)     array with values
             a              array (1 x 1)     slope
             b              array (1 x 1)     intercept
    output:  fx             array (n x 1)     array with transform
    """
    fe = -a * (x - b)
    fx = np.divide(1, 1 + np.exp(fe))
    del fe, x
    return fx
