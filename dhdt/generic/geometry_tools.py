import numpy as np


def point_orientation(p, q, r):
    """ define if a point is left or right of a directed line, given by p
    towards q

    """
    det = (p[0]*q[1] - p[1]*q[0]) + \
          (p[1]*r[...,0] - p[0]*r[...,1]) + \
          (q[0]*r[...,1] - q[1]*r[...,0])
    return det


def point_direction_irt_line(p, q, r):
    dir = np.sign(point_orientation(p, q, r))
    return dir


def point_distance_irt_line(p, q, r):
    dis_pq = np.linalg.norm(np.diff(np.column_stack([p, q]), axis=1))
    dis = np.divide(np.abs(point_orientation(p, q, r)), dis_pq)
    return dis


def oriented_area(p, q, r):
    """ calculate the area of a triangle, given by the point p, q & r """
    return .5 * point_orientation(p, q, r)
