import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate
from scipy.linalg import hankel
from scipy.optimize import fsolve
from skimage.measure import find_contours

from ..generic.attitude_tools import rot_mat
from ..generic.mapping_tools import cart2pol, pol2cart
from .matching_tools_frequency_filters import make_fourier_grid

# radon > direction > sign > shear


def rotated_coord_system(I_grd, J_grd, θ):
    """ create rotated coordinate system

    Parameters
    ----------
    I_grd, J_grd : np.array, size=(m,n), float
        grid with coordinates
    θ : float, unit=angle
        rotation angle

    Returns
    -------
    I_new, J_new : np.array, size=(m,n), float
        new sheared coordinate system
    """
    (m, n) = I_grd.shape
    R = rot_mat(θ)

    stack_grd = np.column_stack([I_grd.ravel(), J_grd.ravel()])

    # calculate new interpolation grid
    grd_new = np.matmul(R, stack_grd.T)

    I_new = np.reshape(grd_new[0, :], (m, n))
    J_new = np.reshape(grd_new[1, :], (m, n))
    return I_new, J_new


def sheared_coord_system(I_grd, J_grd, shear):
    """ create sheared coordinate system

    Parameters
    ----------
    I_grd, J_grd : np.array, size=(m,n), float
        grid with coordinates
    shear : float
        per pixel shear

    Returns
    -------
    I_new, J_new : np.array, size=(m,n), float
        new sheared coordinate system
    """
    (m, n) = I_grd.shape
    A = np.array([[1, (2 * shear) / m], [0, 1]])  # transformation matrix

    stack_grd = np.column_stack([I_grd.ravel(), J_grd.ravel()])

    # calculate new interpolation grid
    grd_new = np.matmul(A, stack_grd.T)

    I_new = np.reshape(grd_new[0, :], (m, n))
    J_new = np.reshape(grd_new[1, :], (m, n))
    return I_new, J_new


def sheared_cross_spectrum(Q, shear, θ):
    (m, n) = Q.shape
    F_1, F_2 = make_fourier_grid(Q, indexing='ij', system='pixel')

    F_1r, F_2r = rotated_coord_system(F_1, F_2, +θ)
    F_1s, F_2s = sheared_coord_system(F_1r, F_2r, shear)
    F_1n, F_2n = rotated_coord_system(F_1s, F_2s, -θ)
    return F_1n, F_2n


def compute_E(cirus, θ):
    # following Kadyrov '06
    a_0 = (1 / 8) * np.sum(cirus**4)
    a_2 = (1 / 8) * np.sum(cirus**4 * np.cos(2 * θ))
    b_2 = (1 / 8) * np.sum(cirus**4 * np.sin(2 * θ))  # (25)

    ab_dist = np.hypot(a_2, b_2)
    cos_omega = np.divide(a_2, ab_dist)
    sin_omega = np.divide(a_2, ab_dist)  # (26)
    omega = np.arctan2(cos_omega, sin_omega)

    φ_min, φ_max = (omega + np.pi) / 2, omega / 2  # (27)
    ψ_min, ψ_max = a_0 - ab_dist, a_0 + ab_dist  # (28)

    a_tilde = np.sqrt(ψ_max) * (np.cos(ψ_max)) ** 2 + \
        np.sqrt(ψ_min) * (np.cos(ψ_min)) ** 2
    b_tilde = np.sqrt(ψ_max) * np.sin(φ_max) * np.cos(φ_max) + \
        np.sqrt(ψ_min) * np.sin(φ_min) * np.cos(φ_min)
    d_tilde = np.sqrt(ψ_max) * (np.sin(ψ_max)) ** 2 + \
        np.sqrt(ψ_min) * (np.sin(ψ_min)) ** 2  # (29)
    D = np.power(ψ_max * ψ_min, 1. / 4)  # (30)
    a = np.divide(a_tilde, D)
    b = np.divide(b_tilde, D)
    d = np.divide(d_tilde, D)

    E_inv = np.array([[a, b], [b, d]])
    E = np.linalg.inv(E_inv)
    return E


def create_cirus_array(ρ, θ, d):
    # polar to coordinates
    (x1, y1) = pol2cart(ρ, θ)
    xy = np.transpose(np.vstack((x1, y1)))
    xy += d  # translate to center

    # imagedraw
    im = Image.new('1', (2 * d, 2 * d), color=0)  # create binary image
    ImageDraw.Draw(im).polygon(xy.flatten().tolist(), fill=1, outline=None)
    B = np.array(im)  # binary circus array
    return B


def affine_binairy_center(B1, B2):  # todo: docstring
    # preparation
    pT, pO = np.sum(B1), np.sum(B2)  # Lebesgue integral

    Jac = pO / pT  # Jacobian

    x = np.linspace(0, B1.shape[1] - 1, B1.shape[1])
    y = np.linspace(0, B1.shape[0] - 1, B1.shape[0])
    X1, Y1 = np.meshgrid(x, y)
    del x, y

    # calculating moments of the template
    x12, x13 = Jac * np.sum(X1**2 * B1), Jac * np.sum(X1**3 * B1)
    x22, x23 = Jac * np.sum(Y1**2 * B1), Jac * np.sum(Y1**3 * B1)
    del X1, Y1

    x = np.linspace(0, B2.shape[1] - 1, B2.shape[1])
    y = np.linspace(0, B2.shape[0] - 1, B2.shape[0])
    X2, Y2 = np.meshgrid(x, y)
    del x, y

    # calculating moments of the observation
    y12 = np.sum(X2**2 * B2)
    y13 = np.sum(X2**3 * B2)
    y12y2 = np.sum(X2**2 * Y2 * B2)
    y22 = np.sum(Y2**2 * B2)
    y23 = np.sum(Y2**3 * B2)
    y1y22 = np.sum(X2 * Y2**2 * B2)
    y1y2 = np.sum(X2 * Y2 * B2)
    del X2, Y2

    # estimation
    def func1(x):
        q12, q13 = x
        return [
            y12 * q12**2 + y22 * q13**2 + 2 * y1y2 * q12 * q13 - x12,
            y13 * q12**3 + y23 * q13**3 + 3 * y12y2 * q12**2 * q13 +
            3 * y1y22 * q12 * q13**2 - x13
        ]

    Q12, Q13 = fsolve(func1, (1.0, 0.0))

    # test for complex solutions, which should be excluded

    def func2(x):
        q22, q23 = x
        return [
            y12 * q22**2 + y22 * q23**2 + 2 * y1y2 * q22 * q23 - x22,
            y13 * q22**3 + y23 * q23**3 + 3 * y12y2 * q22**2 * q23 +
            3 * y1y22 * q22 * q23**2 - x23
        ]

    Q22, Q23 = fsolve(func2, (0.0, 1.0))
    # test for complex solutions, which should be excluded

    Q = np.array([[Q12, Q13], [Q22, Q23]])
    return Q


def compute_K_R(G, θ):
    # following Kadyrov '06
    K = np.hypot(G[0, 0] * np.cos(θ) + G[0, 1] * np.sin(θ),
                 G[1, 0] * np.cos(θ) + G[1, 1] * np.sin(θ))
    cos_R = np.divide(G[0, 0] * np.cos(θ) + G[0, 1] * np.sin(θ), K)
    sin_R = np.divide(G[1, 0] * np.cos(θ) + G[1, 1] * np.sin(θ), K)
    R = np.arctan2(cos_R, sin_R)
    return K, R


def moment(Z, p, q):
    (i_grd, j_grd) = np.meshgrid(np.arange(Z.shape[0]),
                                 np.arange(Z.shape[1]),
                                 sparse=False,
                                 indexing='ij')
    im_mom = np.sum(np.sum((i_grd**p) * (j_grd**q) * Z, axis=1), axis=0)
    return im_mom


def mom_mat(Z):
    M_00 = moment(Z, 0, 0)
    i_bar, j_bar = moment(Z, 1, 0) / M_00, moment(Z, 0, 1) / M_00

    mu_20 = moment(Z, 2, 0) / M_00 - i_bar**2
    mu_02 = moment(Z, 0, 2) / M_00 - j_bar**2
    mu_11 = moment(Z, 1, 1) / M_00 - i_bar * j_bar

    M = hankel([mu_20, mu_11], [mu_11, mu_02])
    return M


def polygon2list(Z, d):
    ij = find_contours(Z, .5)[0]
    ij -= d
    (ρ, φ) = cart2pol(ij[:, 1], ij[:, 0])
    φ = (φ + 2 * np.pi) % (2 * np.pi)  # convert to 0 ... 2pi
    return (ρ, φ)


def central_im_transform(Z, Aff):
    (mI, nI) = Z.shape
    (grd_i, grd_j) = np.meshgrid(np.linspace(-1, 1, mI),
                                 np.linspace(-1, 1, nI),
                                 indexing='ij')

    grd_idx = np.column_stack([grd_i.flatten(), grd_j.flatten()])
    grd_idx_new = np.matmul(Aff, grd_idx.T)
    grd_new_i = np.resize(grd_idx_new[0, :], (mI, nI))
    grd_new_j = np.resize(grd_idx_new[1, :], (mI, nI))

    Z_new = interpolate.griddata(grd_idx,
                                 Z.flatten().T, (grd_new_i, grd_new_j),
                                 method='cubic',
                                 fill_value=0)
    return Z_new


def moment_transform_array(Z):
    M = mom_mat(Z)  # moment matrix
    # sqrtm(M1), np.sqrt(np.linalg.det(M1))
    (w, v) = np.linalg.eig(M)  # eigen values and eigen matrix

    Z_tilde = central_im_transform(Z, v)
    return Z_tilde


def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be real sequences with the same length.
    """
    C = np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y)).real
    return C


# def trace_transform(S1,S2):
#    return a,b,c,d


def scaling_through_power_summation(S1, S2):
    """ refine two spectra through integration/summation

    Parameters
    ----------
    S1 : np.array, size=(m,n), dtype=complex
        Fourier power spectrum of an array
    S2 : np.array, size=(m,n), dtype=complex
        Fourier power spectrum of an array

    Returns
    -------
    a,b : float
        affine parameters

    References
    ----------
    .. [PB97] Pla & Bober. "Estimating translation/deformation motion through
              phase correlation", International conference on image analysis
              and processing. in Lecture notes in computer science, vol.1310,
              1997.
    """
    assert type(S1) == np.ndarray, "please provide an array"
    assert type(S2) == np.ndarray, "please provide an array"

    F_1, F_2 = make_fourier_grid(S1, indexing='ij', system='pixel')

    sc_x = \
        np.sum(np.abs(F_1 * S2).flatten()) / np.sum(np.abs(F_1 * S1).flatten())
    sc_y = \
        np.sum(np.abs(F_2 * S2).flatten()) / np.sum(np.abs(F_2 * S1).flatten())
    return sc_x, sc_y
