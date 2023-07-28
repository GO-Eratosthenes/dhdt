# general libraries
import warnings
import numpy as np


# binary transform functions
def affine_binairy_registration(B1, B2):
    # preparation
    pT = np.sum(B1)  # Lebesgue integral
    pO = np.sum(B2)

    Jac = pO / pT  # Jacobian

    x = np.linspace(0, B1.shape[1] - 1, B1.shape[1])
    y = np.linspace(0, B1.shape[0] - 1, B1.shape[0])
    X1, Y1 = np.meshgrid(x, y)
    del x, y

    # calculating moments of the template
    x11 = Jac * np.sum(X1 * B1)
    x12 = Jac * np.sum(X1 ** 2 * B1)
    x13 = Jac * np.sum(X1 ** 3 * B1)

    x21 = Jac * np.sum(Y1 * B1)
    x22 = Jac * np.sum(Y1 ** 2 * B1)
    x23 = Jac * np.sum(Y1 ** 3 * B1)
    del X1, Y1

    x = np.linspace(0, B2.shape[1] - 1, B2.shape[1])
    y = np.linspace(0, B2.shape[0] - 1, B2.shape[0])
    X2, Y2 = np.meshgrid(x, y)
    del x, y

    # calculating moments of the observation
    y1 = np.sum(X2 * B2)
    y12 = np.sum(X2 ** 2 * B2)
    y13 = np.sum(X2 ** 3 * B2)
    y12y2 = np.sum(X2 ** 2 * Y2 * B2)
    y2 = np.sum(Y2 * B2)
    y22 = np.sum(Y2 ** 2 * B2)
    y23 = np.sum(Y2 ** 3 * B2)
    y1y22 = np.sum(X2 * Y2 ** 2 * B2)
    y1y2 = np.sum(X2 * Y2 * B2)
    del X2, Y2

    # estimation
    mu = pO

    def func1(x):
        q11, q12, q13 = x
        return [mu * q11 + y1 * q12 + y2 * q13 - x11,
                mu * q11 ** 2 + y12 * q12 ** 2 + y22 * q13 ** 2 + 2 * y1 * q11 * q12 + \
                2 * y2 * q11 * q13 + 2 * y1y2 * q12 * q13 - x12,
                mu * q11 ** 3 + y13 * q12 ** 3 + y23 * q13 ** 3 + 3 * y1 * q11 ** 2 * q12 + \
                3 * y2 * q11 ** 2 * q13 + 3 * y12 * q12 ** 2 * q11 + 3 * y12y2 * q12 ** 2 * q13 + \
                3 * y22 * q11 * q13 ** 2 + 3 * y1y22 * q12 * q13 ** 2 + \
                6 * y1y2 * q11 * q12 * q13 - x13]

    Q11, Q12, Q13 = fsolve(func1, (1.0, 1.0, 1.0))

    # test for complex solutions, which should be excluded

    def func2(x):
        q21, q22, q23 = x
        return [mu * q21 + y1 * q22 + y2 * q23 - x21,
                mu * q21 ** 2 + y12 * q22 ** 2 + y22 * q23 ** 2 + 2 * y1 * q21 * q22 + \
                2 * y2 * q21 * q23 + 2 * y1y2 * q22 * q23 - x22,
                mu * q21 ** 3 + y13 * q22 ** 3 + y23 * q23 ** 3 + 3 * y1 * q21 ** 2 * q22 + \
                3 * y2 * q21 ** 2 * q23 + 3 * y12 * q22 ** 2 * q21 + 3 * y12y2 * q22 ** 2 * q23 + \
                3 * y22 * q21 * q23 ** 2 + 3 * y1y22 * q22 * q23 ** 2 + \
                6 * y1y2 * q21 * q22 * q23 - x23]

    Q21, Q22, Q23 = fsolve(func2, (1.0, 1.0, 1.0))
    # test for complex solutions, which should be excluded

    Q = np.array([[Q12, Q13, Q11], [Q22, Q23, Q21]])
    return Q


# boundary describtors
def get_relative_group_distances(x, K=5):
    for i in range(1, K + 1):
        if i == 1:
            x_minus = np.expand_dims(np.roll(x, +i), axis=1)
            x_plus = np.expand_dims(np.roll(x, -i), axis=1)
        else:
            x_new = np.expand_dims(np.roll(x, +i), axis=1)
            x_minus = np.concatenate((x_minus, x_new), axis=1)
            x_new = np.expand_dims(np.roll(x, -i), axis=1)
            x_plus = np.concatenate((x_plus, x_new), axis=1)
            del x_new
    dx_minus = x_minus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dx_plus = x_plus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    return dx_minus, dx_plus


def get_relative_distances(x, x_id, K=5):
    # minus
    start_idx = x_id - K
    ending_idx = x_id
    ids = np.arange(start_idx, ending_idx)
    x_min = x[ids % len(x)]
    # plus
    start_idx = x_id
    ending_idx = x_id + K
    ids = np.arange(start_idx, ending_idx)
    x_plu = x[ids % len(x)]

    dx_minus = x_min - np.repeat(x[x_id], K)
    dx_plus = x_plu - np.repeat(x[x_id], K)
    return dx_minus, dx_plus


def beam_angle_statistics(x, y, K=5, xy_id=None):
    """
    implements beam angular statistics (BAS)
    
    input:
        
    output:
        
    see Arica & Vural, 2003
    BAS: a perceptual shape descriptoy based on the beam angle statistics
    Pattern Recognition Letters 24: 1627-1639
    
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
    """

    if xy_id is None:  # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K)
        ax = 1
    else:  # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # dot product instead of argument
    C_minus = np.arctan2(dy_minus, dx_minus)
    C_plus = np.arctan2(dy_plus, dx_plus)

    C = C_minus - C_plus
    C_1, C_2 = np.mean(C, axis=ax), np.std(C, axis=ax)  # estimate moments
    BAS = np.concatenate(
        (np.expand_dims(C_1, axis=ax), np.expand_dims(C_2, axis=ax)), axis=ax)
    return BAS


def cast_angle_neighbours(x, y, sun, K=5, xy_id=None):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''
    if xy_id is None:  # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K)
        ax = 1
    else:  # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # rotate towards the sun
    CAN = np.concatenate((np.arctan2(sun[0] * dx_minus + sun[1] * dy_minus,
                                     -sun[1] * dx_minus + sun[0] * dy_minus),
                          np.arctan2(sun[0] * dx_plus + sun[1] * dy_plus,
                                     -sun[1] * dx_plus + sun[0] * dy_plus)),
                         axis=ax)
    return CAN


def neighbouring_cast_distances(x, y, sun, K=5, xy_id=None):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''
    if xy_id is None:  # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K)
        ax = 1
    else:  # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # rotate towards the sun and take only one axes
    CD = np.concatenate((sun[0] * dx_minus + sun[1] * dy_minus,
                         sun[0] * dx_plus + sun[1] * dy_plus),
                        axis=ax)
    return CD
