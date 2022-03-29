import numpy as np
import matplotlib as plt

from scipy import ndimage

def get_d8_dir(V_x, V_y):
    """ get directional data of direct neighbors

    Parameters
    ----------
    V_x : np.array, size=(m,n), float
        array of displacements in horizontal direction
    V_y : np.array, size=(m,n), float
        array of displacements in vertical direction

    Returns
    -------
    d8_dir : np.array, size=(m,n), range=0...7
        compass directions starting from North going in clockwise direction

    Notes
    -----
    It is important to know what type of coordinate systems is used, thus:

        .. code-block:: text

               N             0
            NW | NE       7  |  1
          W----+----E   6----8----2
            SW | SE       5  |  3
               S             4

    References
    ----------
    .. [1] Garbrecht & Martz, “The assignment of drainage direction over flat
       surfaces in raster digital elevation models”, Journal of hydrology,
       vol.193 pp.204-213, 1997.
    """
    d8_dir = np.round(np.divide(4*np.arctan2(V_x,V_y),np.pi))
    d8_dir = np.mod(d8_dir,8,
                    where=np.invert(np.isnan(d8_dir)))

    if np.any(np.isnan(d8_dir)):
        d8_dir[np.isnan(d8_dir)] = 8

    d8_dir = d8_dir.astype(int)
    return d8_dir

def get_d8_offsets(shape, order='C'):
    """ get the offsets of the direct neighbors for linear indexing

    Parameters
    ----------
    shape : list
        dimensions of the array of interest
    order : {'C','F'}
        index order in either C-like or Fortran like order

    Returns
    -------
    d8_offsets : np.array, size=(8)
        relative offsets of neighbors

    Notes
    -----
    Two different orders are used:

         .. code-block:: text

        order 'C'        order 'F'
          0 1 2 3        0 4 8
          4 5 6 7        1 5 9
          8 9 ...        2 6 ...

    """
    m,n = shape[0], shape[1]
    if order=='F':
        d8_offset = np.array([
            -1,     # north
            +m-1,   # northeast
            +m,     # east
            +m+1,   # southeast
            +1,     # south
            -m+1,   # southwest
            -m,     # west
            -m-1,   # northwest
            0])     # no data value - stationary
    else:
        d8_offset = np.array([
            -n,     # north
            -n+1,   # northeast
            +1,     # east
            +n+1,   # southeast
            +n,     # south
            +n-1,   # southwest
            -1,     # west
            -n-1,   # northwest
            0])     # no data value - stationary
    return d8_offset

def d8_flow(Class, V_x, V_y, iter=4, analysis=True): #todo: create colored polygon outputs

    # analysis : boolean
    #    create intermediate results, to see the evolution of the algorithm

    # admin
    (m, n) = Class.shape
    mn = m * n
    idx_offset = get_d8_offsets(Class.shape)

    # prepare grids
    d8_flow = get_d8_dir(V_x, V_y)
    # make the border non-moving, i.e.: 8
    d8_flow[:, 0], d8_flow[:, -1], d8_flow[0, :], d8_flow[-1, :] = 8, 8, 8, 8

    d8_offset = np.take(idx_offset, d8_flow, axis=0)
    idx = np.linspace(0, mn - 1, mn, dtype=int).reshape([m, n], order='C')
    Class_new = ndimage.grey_dilation(Class, size=(5, 5))

    if analysis: Class_stack = np.zeros((m, n, iter))

    for i in range(iter):
        # find the boundary of the polygons
        Bnd = np.sign(np.abs(np.gradient(Class_new)).sum(0)).astype(bool) != 0
        # only get the inner/touching boundaries
        Bnd[Class==0] = False

        sel_idx = idx[Bnd]  # pixel locations of interest
        sel_off = d8_offset[Bnd]  # associated offset to use
        downstream_idx = sel_idx + sel_off

        # update
        Class_new[Bnd] = Class_new.ravel()[downstream_idx]
        if analysis: Class_stack[:,:,i] = Class_new.copy()

    if analysis:
        return Class_stack
    else:
        return Class_new