import math
import numpy as np
import random


def enhance_shadow(method, Blue, Green, Red, Nir):
    """
    Given a specific method, employ shadow transform
    input:   method         string            method name to be implemented,
                                              can be one of the following:
                                              - ruffenacht
                                              - nsvi
                                              - mpsi
                                              - shadow_index
                                              - shade_index
             Blue           array (n x m)     blue band of satellite image
             Green          array (n x m)     green band of satellite image
             Red            array (n x m)     red band of satellite image
             Nir            array (n x m)     near-infrared band of satellite
                                              image
    output:  M              array (n x m)     shadow enhanced satellite image
    """

    method = method.lower()

    if method == 'ruffenacht':
        M = ruffenacht(Blue, Green, Red, Nir)
    elif len([n for n in ['shadow', 'index'] if n in method]) == 2:
        M = shadow_index(Blue, Green, Red, Nir)
    elif len([n for n in ['shade', 'index'] if n in method]) == 2:
        M = shade_index(Blue, Green, Red, Nir)
    elif method == 'nsvi':
        M = nsvi(Blue, Green, Red)
    elif method == 'mpsi':
        M = mpsi(Blue, Green, Red)
    elif method == 'gevers':
        M = gevers(Blue, Green, Red)

    return M


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

    matrix = np.array([(0.3811, 0.5783, 0.0402),
                       (0.1967, 0.7244, 0.0782),
                       (0.0241, 0.1228, 0.8444)])

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            lms = np.matmul(matrix, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            L[i][j] = lms[0]
            M[i][j] = lms[1]
            S[i][j] = lms[2]
    return L, M, S


def lms2lab(L, M, S):
    k = np.copy(L)
    a = np.copy(L)
    b = np.copy(L)

    matrix = np.matmul(np.array([(1 / math.sqrt(3), 0, 0),
                                 (0, 1 / math.sqrt(6), 0),
                                 (0, 0, 1 / math.sqrt(2))]),
                       np.array([(+1, +1, +1),
                                 (+1, +1, -2),
                                 (+1, -1, +0)]))

    for i in range(0, L.shape[0]):
        for j in range(0, L.shape[1]):
            lab = np.matmul(matrix, np.array(
                [[L[i][j]], [M[i][j]], [S[i][j]]]))
            k[i][j] = lab[0]
            a[i][j] = lab[1]
            b[i][j] = lab[2]
    return k, a, b


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


def mat_to_gray(matrix, mask_no_data):  # pre-processing or generic
    """
    Transform matix to float, omitting nodata values
    following Ruffenacht
    input:   matrix              array (n x m)     matrix of integers with data
             mask_no_data        array (n x m)     matrix of boolean with
                                                   nodata
    output:  matrix_new          array (m x m)     linear transformed floating
                                                   point [0...1]
    """
    mask_data = ~mask_no_data
    matrix_new = np.float64(matrix)  # /2**16
    matrix_new[mask_data] = np.interp(matrix_new[mask_data],
                                      (matrix_new[mask_data].min(),
                                       matrix_new[mask_data].max()), (0, +1))
    matrix_new[mask_no_data] = 0
    return matrix_new


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

    c3 = np.arctan2(Blue, np.amax(np.dstack((Red, Green))))
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

    L, M, S = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    (l, alfa, beta) = lms2lab(L, M, S)

    return l, alfa, beta


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

# Gevers and Smulders 1999
# Colour-based object recognition. Pattern Recognition, 32, pp. 453–464.
# c3 = arctan(B/max(G,R))


# TO DO:
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
# Reinhard 2001, Color transfer between images DOI: 10.1109/38.946629
