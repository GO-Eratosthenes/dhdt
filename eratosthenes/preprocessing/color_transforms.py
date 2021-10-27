import numpy as np

from .image_transforms import mat_to_gray

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
    .. [1] Smith, "Putting colors in order", Dr. Dobb’s Journal, pp 40, 1993.
    .. [2] Tsai, "A comparative study on shadow compensation of color aerial
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
    .. [1] Gonzalez & Woods "Digital image processing", 1992.
    """

    Y, I, Q = np.zeros_like(Red), np.zeros_like(Red), np.zeros_like(Red)

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
    .. [1] Tsai, "A comparative study on shadow compensation of color aerial
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

def rgb2hsi(Red, Green, Blue):
    """transform red, green, blue arrays to hue, saturation, intensity arrays

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
    Hue : np.array, size=(m,n)
        Hue
    Sat : np.array, size=(m,n)
        Saturation
    Int : np.array, size=(m,n)
        Intensity

    See also
    --------
    erdas2hsi, rgb2hcv, rgb2yiq, rgb2ycbcr, rgb2xyz, rgb2lms

    Notes
    -----
    .. [1] Tsai, "A comparative study on shadow compensation of color aerial
       images in invariant color models", IEEE transactions in geoscience and
       remote sensing, vol. 44(6) pp. 1661--1671, 2006.
    .. [2] Pratt, "Digital image processing" Wiley, 1991.
    """
    if np.ptp(Red.flatten())>1:
        Red = mat_to_gray(Red)
    if np.ptp(Green.flatten())>1:
        Green = mat_to_gray(Green)
    if np.ptp(Blue.flatten())>1:
        Blue = mat_to_gray(Blue)

    Hue,Sat,Int = np.zeros_like(Red), np.zeros_like(Red), np.zeros_like(Red)

    Tsai = np.array([(1 / 3, 1 / 3, 1 / 3),
                     (-np.sqrt(6) / 6, -np.sqrt(6) / 6, -np.sqrt(6) / 3),
                     (1 / np.sqrt(6), 2 / -np.sqrt(6), 0)])

    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            hsi = np.matmul(Tsai, np.array(
                [[Red[i][j]], [Green[i][j]], [Blue[i][j]]]))
            Int[i][j] = hsi[0]
            Sat[i][j] = np.sqrt(hsi[1] ** 2 + hsi[2] ** 2)
            Hue[i][j] = np.atan2(hsi[1], hsi[2])

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
    Hue : np.array, size=(m,n), float
        hue
    Sat : np.array, size=(m,n), float
        saturation
    Int : np.array, size=(m,n), float
        intensity

    See also
    --------
    rgb2hsi

    Notes
    -----
    .. [1] ERDAS, "User handbook", 2013.
    """
    if np.ptp(Red.flatten())>1:
        Red = mat_to_gray(Red)
    if np.ptp(Green.flatten())>1:
        Green = mat_to_gray(Green)
    if np.ptp(Blue.flatten())>1:
        Blue = mat_to_gray(Blue)

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

def rgb2xyz(Red, Green, Blue, method='reinhardt'):
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
    .. [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics
       and applications vol.21(5) pp.34-41, 2001.
    .. [2] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.
    """

    X, Y, Z = np.zeros_like(Red), np.zeros_like(Red), np.zeros_like(Red)

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
    .. [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics
       and applications vol.21(5) pp.34-41, 2001.
    """

    L, M, S = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

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
    .. [1] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.
    .. [2] Silva et al. "Near real-time shadow detection and removal in aerial
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
    .. [1] Ford & Roberts. "Color space conversion", pp. 1--31, 1998.
    .. [2] Silva et al. "Near real-time shadow detection and removal in aerial
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
    .. [1] Reinhard et al. "Color transfer between images", 2001.
    """

    L, M, S = np.zeros_like(Red), np.zeros_like(Red), np.zeros_like(Red)

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
    .. [1] Reinhard et al. "Color transfer between images", 2001.
    """
    l,a,b = np.zeros_like(L), np.zeros_like(L), np.zeros_like(L)

    I = np.matmul(np.array([(1/np.sqrt(3), 0, 0),
                            (0, 1/np.sqrt(6), 0),
                            (0, 0, 1/np.sqrt(2))]), \
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
