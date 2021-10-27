import numpy as np
import random

from skimage import data
from scipy.interpolate import griddata

from ..preprocessing.image_transforms import mat_to_gray
from ..processing.matching_tools_frequency_filters import \
    normalize_power_spectrum
from ..processing.matching_tools import get_integer_peak_location

# assert np.isclose()

def create_sample_image_pair(d=2**7, max_range=1, integer=False):
    """ create an image pair with random offset

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    max_range : float
        maximum offset of the random displacement.
    integer : bool
        should the offset be whole numbers. The default is False.

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : size=(1)
        displacement in the vertical direction
         * "integer" : integer
         * otherwise : float
    random_dj : size=(1)
        displacement in the horizontal direction
         * "integer" : integer
         * otherwise : float
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.
    """
    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    if integer : random_di,random_dj = np.round(random_di),np.round(random_dj)

    (grd_j1,grd_i1) = np.meshgrid(np.linspace(1, mI, mI), np.linspace(1, nI, nI))
    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = stk_1 + np.array([random_di, random_dj])
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[:,0], (mI, nI))
    grd_j2 = np.reshape(grd_2[:,1], (mI, nI))

    # do interpolation
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                    grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                   method='cubic')
    im1_same = im1[mI//2-d:mI//2+d,nI//2-d:nI//2+d]
    return im1_same, im2, random_di, random_dj, im1

def create_sheared_image_pair(d=2**7,sh_x=0.00, sh_y=0.00, max_range=1):
    """ create an image pair with random offset and shearing

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    sh_x : float, unit=image
        horizontal shear to apply, scalar is based on the image width, ranging
        from -1...1 .
    sh_y : float, unit=image
        vertical shear to apply, scalar is based on the image width, ranging
        from -1...1 .
    max_range : float
        maximum offset of the random displacement.

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : float, unit=pixel
        displacement in the vertical direction
    random_dj : float, unit=pixel
        displacement in the horizontal direction
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.

    Notes
    -----
    .. code-block:: text

            sh_x = 0         sh_x > 0
            +--------+      +--------+
            |        |     /        /
            |        |    /        /
            |        |   /        /
            |        |  /        /
            +--------+ +--------+

    """
    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    A = np.array([[1, sh_x], [sh_y, 1]]) # transformation matrix
    (mI,nI) = im1.shape

    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI), np.linspace(-1, 1, nI))

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(A, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))

    # introduce offset
    grd_i2 += random_di/mI
    grd_j2 += random_dj/nI

    # do shearing
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[d:-d,d:-d],grd_j2[d:-d,d:-d]),
                   method='cubic')
    im1_same = im1[d:-d,d:-d]
    return im1_same, im2, random_di, random_dj, im1

def create_scaled_image_pair(d=2**7,sc_x=1.00, sc_y=1.00, max_range=1):
    """ create an image pair with random offset and scaling

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    sc_x : float, unit=image
        horizontal scale to apply, scalar is based on the image width, ranging
        from -1...1 .
    sc_y : float, unit=image
        vertical scale to apply, scalar is based on the image height, ranging
        from -1...1 .
    max_range : float
        maximum offset of the random displacement.

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : float, unit=pixel
        displacement in the vertical direction
    random_dj : float, unit=pixel
        displacement in the horizontal direction
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.

    Notes
    -----
    .. code-block:: text

            sc_x = 1         sc_x > 1
            +--------+      +------------+
            |        |      |            |
            |        |      |            |
            |        |      |            |
            |        |      |            |
            +--------+      +------------+

    """
    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    A = np.array([[1/sc_x, 0], [0, 1/sc_y]]) # transformation matrix
    (mI,nI) = im1.shape

    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI), np.linspace(-1, 1, nI))

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(A, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))

    # introduce offset
    grd_i2 += random_di/mI
    grd_j2 += random_dj/nI

    # do shearing
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[d:-d,d:-d],grd_j2[d:-d,d:-d]),
                   method='cubic')
    im1_same = im1[d:-d,d:-d]
    return im1_same, im2, random_di, random_dj, im1

def construct_correlation_peak(I, di, dj, fwhm=3.):
    """given a displacement, create a gaussian peak

    Parameters
    ----------
    I : np.array, size=(m,n)
        image domain
    di : float
        displacment along the vertical axis
    dj : float
        displacment along the horizantal axis
    fwhm: float
        full width half maximum

    Returns
    -------
    C : np.array, size=(m,n), complex
        array with correlation peak in the form of a circular Gaussian
    """
    (m,n) = I.shape

    (I_grd,J_grd) = np.meshgrid(np.arange(0,n)-(n//2),
                                np.arange(0,m)-(m//2), \
                                indexing='ij')
    I_grd,J_grd = I_grd.astype('float64'), J_grd.astype('float64')
    I_grd -= di
    J_grd -= dj

    C = np.exp(-4*np.log(2) * (I_grd**2 + J_grd**2) / fwhm**2)
    return C

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

def construct_phase_values(IJ, di, dj, indexing='ij', system='radians'):
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

def signal_to_noise(Q,C,norm=2):
    """ calculate the signal to noise from a theoretical and a cross-spectrum

    Parameters
    ----------
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    C : np.array, size=(m,n), dtype=complex
        phase plane
    norm : integer
        norm for the difference

    Returns
    -------
    snr : float, range=0...1
        signal to noise ratio
    """
    Qn = normalize_power_spectrum(Q)
    Q_diff = np.abs(Qn-C)**norm
    snr = 1 - (np.sum(Q_diff) / (2*norm*np.prod(C.shape)))
    return snr

def test_phase_plane_localization(Q, di, dj, tolerance=1.):
    C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    di_hat,dj_hat,_,_ = get_integer_peak_location(C)

    assert np.isclose(np.round(di), di_hat, tolerance)
    assert np.isclose(np.round(dj), dj_hat, tolerance)

    return (di-di_hat, dj-dj_hat)

def test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=.1):
    assert np.isclose(di, di_hat, tolerance)
    assert np.isclose(dj, dj_hat, tolerance)

    return (di_hat-di, dj_hat-dj)

def test_phase_direction(theta, di, dj, tolerance=5):
    tolerance = np.radians(tolerance)
    theta = np.radians(theta)
    theta_tilde = np.arctan2(di, dj)

    # convert to complex domain, so angular difference can be done
    a,b = 1j*np.sin(theta), 1j*np.sin(theta_tilde)
    a += np.cos(theta)
    b += np.cos(theta_tilde)

    assert np.isclose(a,b, tolerance)

def test_normalize_power_spectrum(Q, tolerance=.001):
    # trigonometric version
    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    Qn[Q==0] = 0

    Qd = normalize_power_spectrum(Q)
    assert np.all(np.isclose(Qd,Qn, tolerance))

#todo construct shadowing_caster_casted():
