import numpy as np


def point_orientation(p, q, r):
    """ define if a point is left or right of a directed line, given by p
    towards q

    Parameters
    ----------
    p,q : numpy.ndarray, size=(2,)
        two dimensional points specifing a line
    r : numpy.ndarray, size=(m,2)
        two dimensional points

    Returns
    -------
    det : numpy.ndarray, size=(m,2), dtype=float
        normalized and oriented separation distance
    """
    det = (p[0] * q[1] - p[1] * q[0]) + \
          (p[1] * r[..., 0] - p[0] * r[..., 1]) + \
          (q[0] * r[..., 1] - q[1] * r[..., 0])
    return det


def point_direction_irt_line(p, q, r):
    """ define if a point is left or right of a directed line, given by p
    towards q

    Parameters
    ----------
    p,q : numpy.ndarray, size=(2,)
        two dimensional points specifing a line
    r : numpy.ndarray, size=(m,2)
        array with two dimensional points

    Returns
    -------
    dir : {-1,0,+1}, numpy.ndarray, size=(m,2)
        specifies the side where a point is situated in respect to the line
    """
    dir = np.sign(point_orientation(p, q, r))
    return dir


def point_distance_irt_line(p, q, r):
    """ calculate the perpendicular distance of a point in respect to a
    directed line, given by 'p' towards 'q'

    Parameters
    ----------
    p,q : numpy.ndarray, size=(2,)
        two dimensional points specifing a line
    r : numpy.ndarray, size=(m,2)
        array with two dimensional points

    Returns
    -------
    dis : numpy.ndarray, size=(m,2), dtype=float
        perpendicular distance of the point in respect to the line
    """
    dis_pq = np.linalg.norm(np.diff(np.column_stack([p, q]), axis=1))
    dis = np.divide(np.abs(point_orientation(p, q, r)), dis_pq)
    return dis


def oriented_area(p, q, r):
    """ calculate the area of a triangle, given by the point p, q & r

    Parameters
    ----------
    p,q,r : numpy.ndarray, size=(2,)
        two dimensional points specifying a triangle
    """
    return .5 * point_orientation(p, q, r)


def segment_intersection(p, q, r, s):
    """ calculate if two segments intersect

    Parameters
    ----------
    p,q : numpy.ndarray, size=(2,)
        two dimensional points specifying a line
    r,s : numpy.ndarray, size=(m,2)
        array with two-dimensional points specifying line-segments

    Returns
    -------
    numpy.ndarray, size=(m,), dtype=bool
        array with verdict if a crossing is present
    """

    t = np.divide((p[0] - r[..., 0]) * (r[..., 1] - s[..., 1]) -
                  (p[1] - r[..., 1]) * (r[..., 0] - s[..., 0]),
                  (p[0] - q[0]) * (r[..., 1] - s[..., 1]) -
                  (p[1] - q[1]) * (r[..., 0] - s[..., 0])
                  )
    u = np.divide((p[0] - r[..., 0]) * (p[1] - q[1]) -
                  (p[1] - r[..., 1]) * (p[0] - q[0]),
                  (p[0] - q[0]) * (r[..., 1] - s[..., 1]) -
                  (p[1] - q[1]) * (r[..., 0] - s[..., 0])
                  )
    x = np.logical_and(0 <= t <= 1, 0 <= u <= 1)
    return x
