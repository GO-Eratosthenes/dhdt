import numpy as np

# forward attitude functions
def rot_mat(theta):
    """ build a 2x2 rotation matrix

    Parameters
    ----------
    theta : float
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
    R = np.array([[+np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [+np.sin(np.radians(theta)), +np.cos(np.radians(theta))]])
    return R

def euler_rot_x(psi):
    """ create three dimensional rotation matrix, along X-axis

    Parameters
    ----------
    psi : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_x : numpy.array
        rotation matrix along the X-axis (1st dimension)
    """

    R_x = np.diag(3)
    R_x[1:,1:] = rot_mat(psi)
    return R_x

def euler_rot_y(theta):
    """ create three dimensional rotation matrix, along Y-axis

    Parameters
    ----------
    theta : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_y : numpy.array
        rotation matrix along the X-axis (2nd dimension)
    """
    R_y = np.diag(3)
    R_2 = rot_mat(theta)
    R_y[0,0],R_y[0,-1],R_y[-1,0],R_y[-1,-1] = R_2[0,0],R_2[0,-1],R_2[-1,0],\
                                              R_2[-1,-1]
    return R_y

def euler_rot_z(phi):
    """ create three dimensional rotation matrix, along Z-axis

    Parameters
    ----------
    phi : float, unit=degrees
        amount of rotation

    Returns
    -------
    R_x : numpy.array
        rotation matrix along the X-axis (3rd dimension)
    """
    R_z = np.diag(3)
    R_z[:-1, :-1] = rot_mat(phi)
    return R_z

# backward attitude methods

def rot_2_euler(R):
    """ convert to euler angels, rotating order is via Z->Y->X

    Parameters
    ----------
    R : numpy.array, size=(3,3)
        rotation matrix, direction cosine matrix

    Returns
    -------
    phi, theta, psi : float, unit=degrees
        rotation angles

    References
    ----------
    .. [1] Kim, "Rigid body dynamics for beginners", 2013.
    """
    phi = np.rad2deg(np.arctan2(R[1,2], R[2,2]))        # p.60 in [1]
    theta = np.rad2deg(np.arcsin(-R[0,2]))              # eq.5.13 in [1]
    psi = np.rad2deg(np.arctan2(R[0,1], R[0,0]))        # eq.5.14 in [1]
    return phi, theta, psi

def lodrigues_est(XYZ_1, XYZ_2):
    m = XYZ_1.shape[0]
    dXYZ = XYZ_1 - XYZ_2

    A = np.vstack((np.moveaxis(np.expand_dims(
        np.hstack((np.zeros((m,1)),
                   np.expand_dims(-XYZ_2[:,2]-XYZ_1[:,2],axis=1),
                   np.expand_dims(+XYZ_2[:,1]+XYZ_1[:,1],axis=1))),
                    axis=2), [0,1,2], [2,1,0]),
                   np.moveaxis(np.expand_dims(
        np.hstack((
                   np.expand_dims(+XYZ_2[:,2]+XYZ_1[:,2], axis=1),
                   np.zeros((m, 1)),
                   np.expand_dims(-XYZ_2[:,0]-XYZ_1[:,0], axis=1))),
                       axis=2), [0,1,2], [2,1,0]),
                   np.moveaxis(np.expand_dims(
        np.hstack((
                np.expand_dims(-XYZ_2[:,1]-XYZ_1[:,1], axis=1),
                np.expand_dims(-XYZ_2[:,0]-XYZ_1[:,0], axis=1),
                np.zeros((m, 1)))),
                       axis=2), [0,1,2], [2,1,0]),))

    # squeeze back to collumns
    x_hat = np.linalg.lstsq(np.transpose(A, axes=(2,0,1)).reshape((3*m,3,1)).squeeze(),
                            dXYZ.reshape(-1)[:,np.newaxis], rcond=None)[0]
    a,b,c = x_hat[0], x_hat[1], x_hat[2]
    return a,b,c

def lodrigues_mat(a,b,c):
    S = np.array([[0, +c/2, -b/2],
                  [-c/2, 0, +a/2,],
                  [+b/2, -a/2, 0]]) # skew symmetric
    R = (np.eye(3) - S)* np.linalg.inv(np.eye(3) + S)
    return R