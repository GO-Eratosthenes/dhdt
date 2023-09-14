import numpy as np


def ordered_merge(t_0, t_1, x_0, x_1):
    """ merge two arrays that describe an (overlapping) time-series

    Parameters
    ----------
    t_0 : numpy.ndarray, size=(m,), dtype={float, numpy.datetime64}
        array with time instances
    t_1 : numpy.ndarray, size=(k,), dtype={float, numpy.datetime64}
        array with time instances
    x_0 : numpy.ndarray, size=(m,n)
        array with data instances at timestamp given by "t_0"
    x_1 : numpy.ndarray, size=(k,n)
        array with data instances at timestamp given by "t_1"

    Returns
    -------
    t : numpy.ndarray, size=(p,), dtype={float, numpy.datetime64}
        array with ordered time instances
    x : numpy.ndarray, size=(p,l)
        array with ordered data instances
    """
    assert t_0.shape[0] == x_0.shape[0], \
        'array should have the same amount of entries in the first dimension'
    assert t_1.shape[0] == x_1.shape[0], \
        'array should have the same amount of entries in the first dimension'
    assert x_0.shape[1] == x_1.shape[1], \
        'data arrays should have the same amount of fields'

    t_stack = np.hstack((t_0, t_1))
    x_stack = np.vstack((np.atleast_2d(x_0), np.atleast_2d(x_1)))

    Idx = np.argsort(t_stack)
    t_stack, x_stack = t_stack[Idx], x_stack[Idx, :]
    if t_stack.dtype.type == np.datetime64:
        IN = (np.diff(t_stack) / np.timedelta64(1, 's')) != 0
    else:
        IN = np.diff(t_stack) != 0
    IN = np.hstack((True, IN))

    t_merge, x_merge = t_stack[IN], x_stack[IN, :]
    return t_merge, x_merge


# forward attitude functions
def rot_mat(θ):
    r""" build a 2x2 rotation matrix

    Parameters
    ----------
    θ : float
        angle in degrees

    Returns
    -------
    R : np.array, size=(2,2)
        2D rotation matrix

    Notes
    -----
    Matrix is composed through,

        .. math:: \mathbf{R}[1,:] = [+\cos(\omega), -\sin(\omega)]
        .. math:: \mathbf{R}[2,:] = [+\sin(\omega), +\cos(\omega)]

    The angle(s) are declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x

    """
    θ = np.deg2rad(θ)
    R = np.array([[+np.cos(θ), -np.sin(θ)], [+np.sin(θ), +np.cos(θ)]])
    return R


def euler_rot_x(ψ):
    """ create three dimensional rotation matrix, along X-axis

    Parameters
    ----------
    ψ : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_x : numpy.ndarray
        rotation matrix along the X-axis (1st dimension)
    """

    R_x = np.diag(3)
    R_x[1:, 1:] = rot_mat(ψ)
    return R_x


def euler_rot_y(θ):
    """ create three dimensional rotation matrix, along Y-axis

    Parameters
    ----------
    θ : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_y : numpy.ndarray
        rotation matrix along the X-axis (2nd dimension)
    """
    R_y = np.diag(3)
    R_2 = rot_mat(θ)
    R_y[0, 0] = R_2[0, 0]
    R_y[0, -1] = R_2[0, -1]
    R_y[-1, 0] = R_2[-1, 0]
    R_y[-1, -1] = R_2[-1, -1]
    return R_y


def euler_rot_z(φ):
    """ create three dimensional rotation matrix, along Z-axis

    Parameters
    ----------
    φ : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_x : numpy.ndarray
        rotation matrix along the X-axis (3rd dimension)
    """
    R_z = np.diag(3)
    R_z[:-1, :-1] = rot_mat(φ)
    return R_z


# backward attitude methods
def rot_2_euler(R):
    """ convert to euler angels, rotating order is via Z->Y->X, see also
    [Ki13]_

    Parameters
    ----------
    R : numpy.ndarray, size=(3,3)
        rotation matrix, direction cosine matrix

    Returns
    -------
    φ, θ, ψ : float, unit=degrees
        rotation angles

    Notes
    -----
    These angles go also by other definitions, namely
        phi : roll, rotation about the X-axis
        θ : pitch, rotation about the Y-axis
        ψ : yaw, rotaion about the Z-axis

    References
    ----------
    .. [Ki13] Kim, "Rigid body dynamics for beginners", 2013.
    """
    φ = np.rad2deg(np.arctan2(R[1, 2], R[2, 2]))  # p.60 in [1]
    θ = np.rad2deg(np.arcsin(-R[0, 2]))  # eq.5.13 in [1]
    ψ = np.rad2deg(np.arctan2(R[0, 1], R[0, 0]))  # eq.5.14 in [1]
    return φ, θ, ψ


def quat_2_euler(q1, q2, q3, q4):
    """ convert to euler angels, rotating order is via Z->Y->X, see also
    [Ki13]_

    Parameters
    ----------
    q1,q2,q3,q4 : {float, numpy.ndarray}
        quaternions

    Returns
    -------
    φ, θ, ψ : {float, numpy.ndarray}, unit=degrees
        rotation angles

    Notes
    -----
    These angles go also by other definitions, namely
        phi : roll, rotation about the X-axis
        θ : pitch, rotation about the Y-axis
        ψ : yaw, rotaion about the Z-axis

    References
    ----------
    .. [Ki13] Kim, "Rigid body dynamics for beginners", 2013.
    """
    φ = np.rad2deg(np.arctan2(2 * (q2 * q3 + q1 * q4),
                              1 - 2 * (q1 ** 2 + q2 ** 2)))  # eq. 5.12 in [1]
    θ = np.rad2deg(np.arcsin(2 * (q2 * q4 - q1 * q3)))  # eq. 5.13 in [1]
    ψ = np.rad2deg(np.arctan2(2 * (q1 * q2 + q3 * q4),
                              1 - 2 * (q2 ** 2 + q3 ** 2)))  # eq. 5.14 in [1]
    return φ, θ, ψ


def get_rotation_angle(R):
    """ get the amount of rotation, along a fixed axis (i.e.: vector), see also
    [Ku02]_.

    Parameters
    ----------
    R : numpy.ndarray, size=(3,3)
        rotation matrix

    Returns
    -------
    φ : float, unit=degrees
        amount of rotation, along a fixed axis

    See Also
    --------
    get_rotation_vector

    References
    ----------
    .. [Ku02] Kuipers, "Quaternions and rotation sequences", 2002.
    """
    φ = np.rad2deg(np.arccos(np.divide(np.trace(R) - 1,
                                       2)))  # eq. 3.4 in [1], pp.57
    return φ


def get_rotation_vector(R):
    """ get the vector where rotation is applied around, see also [Ku02]_.

    Parameters
    ----------
    R : numpy.ndarray, size=(3,3)
        rotation matrix

    Returns
    -------
    v : numpy.ndarray, size=(3,)
        vector along which is rotated in 3D space

    See Also
    --------
    get_rotation_angle

    References
    ----------
    .. [Ku02] Kuipers, "Quaternions and rotation sequences", 2002.
    """
    v = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 1],
                  R[0, 1] - R[1, 0]])  # eq. 3.12 in [1], pp.66
    return v


def rodrigues_est(XYZ_1, XYZ_2):
    """ estimate the rotation between two coupled coordinate lists, based on
    [Ro40]_.

    Parameters
    ----------
    XYZ_1 : numpy.ndarray, size=(k,3)
        array with points in coordinate frame 1
    XYZ_2 : numpy.ndarray, size=(k,3)
        array with points in coordinate frame 1

    Returns
    -------
    a,b,c : float
        components of the rejection vector

    See Also
    --------
    rodrigues_mat

    References
    ----------
    .. [Ro40] Rodrigues, "Des lois géométriques qui régissent les déplacements
              d'un système solide dans l'espace, et de la variation des
              coordonnées provenant de ces déplacements considérés indépendants
              des causes qui peuvent les produire", Journal de mathématiques
              pures et appliquées. vol.5 pp.380–440, 1840.
    """
    m = XYZ_1.shape[0]
    dXYZ = XYZ_1 - XYZ_2

    A = np.vstack((
        np.moveaxis(
            np.expand_dims(np.hstack(
                (np.zeros(
                    (m, 1)), np.expand_dims(-XYZ_2[:, 2] - XYZ_1[:, 2],
                                            axis=1),
                 np.expand_dims(+XYZ_2[:, 1] + XYZ_1[:, 1], axis=1))),
                axis=2), [0, 1, 2], [2, 1, 0]),
        np.moveaxis(
            np.expand_dims(np.hstack(
                (np.expand_dims(+XYZ_2[:, 2] + XYZ_1[:, 2],
                                axis=1), np.zeros((m, 1)),
                 np.expand_dims(-XYZ_2[:, 0] - XYZ_1[:, 0], axis=1))),
                axis=2), [0, 1, 2], [2, 1, 0]),
        np.moveaxis(
            np.expand_dims(np.hstack(
                (np.expand_dims(-XYZ_2[:, 1] - XYZ_1[:, 1], axis=1),
                 np.expand_dims(-XYZ_2[:, 0] - XYZ_1[:, 0],
                                axis=1), np.zeros((m, 1)))),
                axis=2), [0, 1, 2], [2, 1, 0]),
    ))

    # squeeze back to collumns
    x_hat = np.linalg.lstsq(np.transpose(A, axes=(2, 0, 1)).reshape(
        (3 * m, 3, 1)).squeeze(),
                            dXYZ.reshape(-1)[:, np.newaxis],
                            rcond=None)[0]
    a, b, c = x_hat[0], x_hat[1], x_hat[2]
    return a, b, c


def rodrigues_mat(a, b, c):
    S = np.array([[0, +c / 2, -b / 2], [
        -c / 2,
        0,
        +a / 2,
    ], [+b / 2, -a / 2, 0]])  # skew symmetric
    R = (np.eye(3) - S) * np.linalg.inv(np.eye(3) + S)
    return R


def ecef2ocs(xyz, uvw):
    """ Rotate from an Earth fixed frame towards an orbital coordinate frame,
    see also [Ye20].

    Parameters
    ----------
    xyz : numpy.ndarray, size=(3,), unit=m
        position vector
    uvw : numpy.ndarray, size=(3,), unit=m/s
        velocity vector

    Returns
    -------
    R : numpy.ndarray, size=(3,3)
        rotation matrix

    See Also
    --------
    rot_2_euler

    References
    ----------
    .. [Ye20] Ye et al., "Resolving time-varying attitude jitter of an optical
              remote sensing satellite based on a time-frequency analysis"
              Optics express, vol.28(11) pp.15805--15823, 2020.
    """
    # create rotation matrix towards orbital coordinate system
    Z_o = np.divide(xyz, np.linalg.norm(xyz))
    X_o = np.cross(uvw, xyz)
    X_o = np.divide(X_o, np.linalg.norm(X_o))
    Y_o = np.cross(Z_o, X_o)
    Y_o = np.divide(Y_o, np.linalg.norm(Y_o))
    R = np.column_stack([X_o, Y_o, Z_o])  # (1) in [Ye20]
    return R


def deconvolve_jitter(f, dt):
    N = f.size()
    k = np.arange(N)

    g_F = np.fft.fft(g)
    # (5) in [Ye20]
    f = np.fft.ifft(
        np.divide(g_F,
                  np.exp(np.divide(2 * np.pi * 1j * k * dt, N)) - 1))
    return f


def asd(f_a, f_c, beta, F):
    d_roll = np.divide(-f_c, F)
    d_pitch = np.divide(-f_a * np.cos(beta) ** 2, F)  # (7) in [Ye20]
    return
