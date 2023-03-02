import numpy as np
from scipy.interpolate import griddata
from skimage import data

from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.processing.matching_tools import get_integer_peak_location
from dhdt.processing.matching_tools_frequency_filters import normalize_power_spectrum


def create_sample_image_pair(d=2**7, max_range=1, integer=False, ndim=1):
    """ create an image pair with random offset

    Parameters
    ----------
    d : integer, {x ∈ ℕ}, range=0...256
        radius of the template. The default is 2**7.
    max_range : float
        maximum offset of the random displacement.
    integer : bool
        should the offset be whole numbers. The default is False.
    ndim : integer, {x ∈ ℕ}, {1,3}
        number of bands making up an image

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

    Notes
    -----
    The following coordinate system is used here:

        .. code-block:: text

          indexing   |
          system 'ij'|
                     |
                     |       j
             --------+-------->
                     |
                     |
          image      | i
          based      v

    """
    if ndim==1:
        im1 = mat_to_gray(data.grass())
    else:
        im1 = data.astronaut()
        ndim = np.maximum(im1.shape[-1], ndim)
    mI,nI = im1.shape[0:2]

    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    if integer: random_di,random_dj = np.round(random_di),np.round(random_dj)

    (grd_j1,grd_i1) = np.meshgrid(np.linspace(1, mI, mI), np.linspace(1, nI, nI))
    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = stk_1 + np.array([random_di, random_dj])
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[:,0], (mI, nI))
    grd_j2 = np.reshape(grd_2[:,1], (mI, nI))

    # do interpolation
    if ndim==1:
        im2 = griddata(stk_1, im1.flatten().T,
                       (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                        grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                       method='cubic')
        im1_same = im1[mI//2-d:mI//2+d,nI//2-d:nI//2+d]

    else:
        im2 = np.zeros((2*d,2*d,ndim))
        for i in range(ndim):
            im2[...,i] = griddata(stk_1, im1[...,i].flatten().T,
                                  (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                                   grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                                  method='linear')
        im2 = im2.astype('uint8')
        im1_same = im1[mI//2-d:mI//2+d, nI//2-d:nI//2+d,:]
    return im1_same, im2, random_di, random_dj, im1


def create_sheared_image_pair(d=2**7,sh_i=0.00, sh_j=0.00, max_range=1):
    """ create an image pair with random offset and shearing

    Parameters
    ----------
    d : integer, {x ∈ ℕ}, range=0...256
        radius of the template. The default is 2**7.
    sh_j : float, unit=image
        horizontal shear to apply, scalar is based on the image width, ranging
        from -1...1 .
    sh_i : float, unit=image
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
    The following coordinate system is used here:

    .. code-block:: text

      indexing   |
      system 'ij'|
                 |
                 |       j
         --------+-------->
                 |
                 |
      image      | i
      based      v

    The shear is applied in the following manner:

    .. code-block:: text

            sh_j = 0         sh_j > 0
            +--------+      +--------+
            |        |     /        /
            |        |    /        /
            |        |   /        /
            |        |  /        /
            +--------+ +--------+

    The shear parameter is based upon a centered unit image domain, that is, the
    image extent spans -1...+1
    """
    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    A = np.array([[1, sh_i], [sh_j, 1]]) # transformation matrix
    (mI,nI) = im1.shape

    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI),
                                  np.linspace(-1, 1, nI),
                                  indexing='ij')

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(A, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))

    # introduce offset
    grd_i2 += random_di/mI
    grd_j2 += random_dj/nI

    # do sheared interpolation
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                    grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                   method='cubic')
    im1_same = im1[mI//2-d:mI//2+d,nI//2-d:nI//2+d]
    return im1_same, im2, random_di, random_dj, im1


def create_scaled_image_pair(d=2**7,sc_x=1.00, sc_y=1.00, max_range=1):
    """ create an image pair with random offset and scaling

    Parameters
    ----------
    d : integer, {x ∈ ℕ}, range=0...256
        radius of the template. The default is 2**7.
    sc_x : float, {x ∈ ℝ | -1 ≥ x ≥ +1}, unit=image
        horizontal scale to apply, scalar is based on the image width, ranging
        from -1...1 .
    sc_y : float, {x ∈ ℝ | -1 ≥ x ≥ +1}, unit=image
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


def construct_correlation_peak(I, di, dj, fwhm=3., origin='center'):
    """given a displacement, create a gaussian peak

    Parameters
    ----------
    I : np.array, size=(m,n)
        image domain
    di : {float, np.array}
        displacement along the vertical axis
    dj : {float, np.array}
        displacement along the horizontal axis
    fwhm: float
        full width half maximum

    Returns
    -------
    C : np.array, size=(m,n), complex
        array with correlation peak in the form of a circular Gaussian
    """
    (m,n) = I.shape

    (I_grd, J_grd) = np.meshgrid(np.arange(0, m),
                                 np.arange(0, n),
                                 indexing='ij')
    if origin in ('center'):
        I_grd -= m//2
        J_grd -= n//2
    I_grd,J_grd = I_grd.astype('float64'), J_grd.astype('float64')

    if len(di)==1:
        I_grd -= di
        J_grd -= dj
        C = np.exp(-4*np.log(2) * (I_grd**2 + J_grd**2) / fwhm**2)
        return C

    C = np.zeros((m,n), dtype=float)
    for idx,delta_i in enumerate(di):
        delta_j = dj[idx]
        C = np.maximum(C,
                       np.real(
                       np.exp(-4*np.log(2) * ((I_grd-delta_i)**2 +
                                              (J_grd-delta_j)**2) / fwhm**2)))
    return C


def test_phase_plane_localization(Q, di, dj, tolerance=1.):
    """ test if an estimate is within bounds

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum
    di,dj : {float, integer}, unit=pixels
        displacement estimation
    tolerance : float
        absolute tolerance that is allowed
    """
    C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    di_hat,dj_hat,_,_ = get_integer_peak_location(C)

    assert np.isclose(np.round(di), di_hat, tolerance)
    assert np.isclose(np.round(dj), dj_hat, tolerance)
    return

def test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=.1):
    assert np.isclose(di, di_hat, tolerance)
    assert np.isclose(dj, dj_hat, tolerance)
    return

def test_phase_direction(θ, di, dj, tolerance=5):
    tolerance = np.deg2radrad(tolerance)
    θ = np.deg2rad(θ)
    θ_tilde = np.arctan2(di, dj)

    # convert to complex domain, so angular difference can be done
    a,b = 1j*np.sin(θ), 1j*np.sin(θ_tilde)
    a += np.cos(θ)
    b += np.cos(θ_tilde)
    assert np.isclose(a,b, tolerance)
    return

def test_normalize_power_spectrum(Q, tolerance=.001):
    # trigonometric version
    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    np.putmask(Qn, Q==0, 0)

    Qd = normalize_power_spectrum(Q)
    assert np.all(np.isclose(Qd,Qn, tolerance))
    return