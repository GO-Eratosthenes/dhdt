import numpy as np

from scipy import ndimage
from ..generic.mapping_tools import map2pix
from ..generic.gis_tools import get_mask_boundary
from ..processing.matching_tools import remove_posts_outside_image


def get_d8_dir(V_x, V_y):
    """ get directional data of direct neighbors, following [GM97]_.

    Parameters
    ----------
    V_x : numpy.ndarray, size=(m,n), float
        array of displacements in horizontal direction
    V_y : numpy.ndarray, size=(m,n), float
        array of displacements in vertical direction

    Returns
    -------
    d8_dir : numpy.ndarray, size=(m,n), range=0...7
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
    .. [GM97] Garbrecht & Martz, “The assignment of drainage direction over
              flat surfaces in raster digital elevation models”, Journal of
              hydrology, vol.193 pp.204-213, 1997.
    """
    d8_dir = np.round(np.divide(4 * np.arctan2(V_x, V_y), np.pi))
    d8_dir = np.mod(d8_dir, 8, where=np.invert(np.isnan(d8_dir)))

    # assign no direction to non-moving velocities and no-data values
    np.putmask(d8_dir, np.isnan(d8_dir), 8)
    np.putmask(d8_dir, np.hypot(V_x, V_y) == 0, 8)

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
    d8_offsets : numpy.ndarray, size=(8)
        relative offsets of neighbors

    Notes
    -----
    Two different orders are used:

     .. code-block:: text

          0 1 2 3        0 4 8
          4 5 6 7        1 5 9
          8 9 ...        2 6 ...

          order 'C'      order 'F'

    """
    m, n = shape[0], shape[1]
    if order == 'F':
        d8_offset = np.array([
            -1,  # north
            +m - 1,  # northeast
            +m,  # east
            +m + 1,  # southeast
            +1,  # south
            -m + 1,  # southwest
            -m,  # west
            -m - 1,  # northwest
            0
        ])  # no data value - stationary
    else:
        d8_offset = np.array([
            -n,  # north
            -n + 1,  # northeast
            +1,  # east
            +n + 1,  # southeast
            +n,  # south
            +n - 1,  # southwest
            -1,  # west
            -n - 1,  # northwest
            0
        ])  # no data value - stationary
    return d8_offset


def get_d8_flow_towards(dir=0):
    d8_to_you = np.mod(np.linspace(0, 8, 9) + 4 + dir, 8)
    d8_to_you[-1] = 8
    return d8_to_you


def d8_flow(Class,
            V_x,
            V_y,
            iter=20,
            analysis=True):  # todo: create colored polygon outputs
    """ re-organize labelled dataset by steering towards convergent zones

    Parameters
    ----------
    Class : numpy.ndarray, size=(m,n)
        labelled array
    V_x, V_y :  numpy.ndarray, size=(m,n)
        grids with horizontal and vertical directions
    iter : integer, {x ∈ ℕ | x ≥ 0}, default=20
        number of iterations to be executed
    analysis : boolean
        keeps intermediate results, to see the evolution of the algorithm

    Returns
    -------
    Class_new : numpy.ndarray, size=(m,n)
        newly evolved labelled array

    See Also
    --------
    get_d8_dir, d8_flow
    """

    # admin
    m, n = Class.shape[0:2]
    mn = np.multiply(m, n)
    idx_offset = get_d8_offsets(Class.shape)

    # prepare grids
    d8_flow = get_d8_dir(V_x, V_y)
    # make the border non-moving, i.e.: 8
    d8_flow[:, 0], d8_flow[:, -1], d8_flow[0, :], d8_flow[-1, :] = 8, 8, 8, 8

    d8_offset = np.take(idx_offset, d8_flow, axis=0)
    idx = np.linspace(0, mn - 1, mn, dtype=int).reshape([m, n], order='C')
    Class_new = ndimage.grey_dilation(Class, size=(5, 5))

    if analysis:
        Class_stack = np.zeros((m, n, iter))

    for i in range(iter):
        # find the boundary of the polygons
        Bnd = np.sign(np.abs(np.gradient(Class_new)).sum(0)).astype(bool) != 0
        # only get the inner/touching boundaries
        Bnd[Class == 0] = False

        sel_idx = idx[Bnd]  # pixel locations of interest
        sel_off = d8_offset[Bnd]  # associated offset to use
        downstream_idx = sel_idx + sel_off

        # update
        Class_new[Bnd] = Class_new.ravel()[downstream_idx]
        if analysis:
            Class_stack[..., i] = Class_new.copy()

    if analysis:
        return Class_stack
    else:
        return Class_new


def d8_catchment(V_x, V_y, geoTransform, x, y, disperse=True, flat=True):
    """ estimate the catchment behind a seed point

    Parameters
    ----------
    V_x, V_y :  numpy.ndarray, size=(m,n)
        grids with horizontal and vertical directions
    geoTransform
    x, y : {float, numpy.ndarray}
        map locations of the seeds

    Returns
    -------
    C :  numpy.ndarray, size=(m,n), dtype=boolean
        grid with the specific catchments

    See Also
    --------
    get_d8_dir, d8_flow
    """
    C = np.zeros_like(V_x, dtype=bool)

    # admin
    m, n = C.shape[0:2]
    mn = np.multiply(m, n)
    idx_offset = get_d8_offsets(C.shape)

    # initiate seeds
    i, j = map2pix(geoTransform, x, y)
    i, j = np.round(i).astype(int).flatten(), np.round(j).astype(int).flatten()
    i, j, _ = remove_posts_outside_image(C, i, j)
    C[i, j] = True

    # prepare grids
    d8_flow = get_d8_dir(V_x, V_y)
    # make the border non-moving, i.e.: 8
    d8_flow[:, 0], d8_flow[:, -1], d8_flow[0, :], d8_flow[-1, :] = 8, 8, 8, 8

    idx = np.linspace(0, mn - 1, mn, dtype=int).reshape([m, n], order='C')

    # create list to see which grids will point to the poi
    d8_2_you = get_d8_flow_towards()

    for i in range(2 * np.ceil(np.hypot(m, n)).astype(int)):
        # get boundary
        Bnd = get_mask_boundary(C)
        Bnd[:, 0] = False
        Bnd[:, -1] = False
        Bnd[0, :] = False
        Bnd[-1, :] = False
        sel_idx = idx[Bnd]  # pixel locations of interest

        # get neighborhood of boundary
        ngh_idx = np.add.outer(sel_idx, idx_offset)
        ngh_d8 = np.take(d8_flow, ngh_idx, mode='clip')
        ngh_2_you = np.repeat(np.atleast_2d(d8_2_you), ngh_d8.shape[0], axis=0)

        # look if it flows to this cell
        in_C = ngh_d8 == ngh_2_you
        if flat:  # , or is flat
            in_C = np.logical_or(in_C, ngh_d8 == 8)
        if disperse:
            ngh_2_you = np.repeat(np.atleast_2d(get_d8_flow_towards(dir=-1)),
                                  ngh_d8.shape[0],
                                  axis=0)
            in_C = np.logical_or(in_C, ngh_d8 == ngh_2_you)
            ngh_2_you = np.repeat(np.atleast_2d(get_d8_flow_towards(dir=+1)),
                                  ngh_d8.shape[0],
                                  axis=0)
            in_C = np.logical_or(in_C, ngh_d8 == ngh_2_you)
        in_C_idx = ngh_idx[in_C]

        already_in_C = np.take(C, in_C_idx, mode='clip')
        # stop iteration if no update is done
        if np.all(already_in_C):
            break
        # update
        np.put(C, in_C_idx, True, mode='clip')

        # stop if all is full or empty
        if np.logical_or(np.all(C), np.all(np.invert(C))):
            break
    return C


# todo: stopping criteria
def d8_streamflow(V_x, V_y, geoTransform, x, y):
    C = np.zeros_like(V_x, dtype=bool)

    # admin
    m, n = C.shape[0:2]
    mn = np.multiply(m, n)
    idx_offset = get_d8_offsets(C.shape)

    # initiate seeds
    i, j = map2pix(geoTransform, x, y)
    i, j = np.round(i).astype(int).flatten(), np.round(j).astype(int).flatten()
    i, j, _ = remove_posts_outside_image(C, i, j)
    C[i, j] = True

    # prepare grids
    d8_flow = get_d8_dir(V_x, V_y)
    # make the border non-moving, i.e.: 8
    d8_flow[:, 0], d8_flow[:, -1], d8_flow[0, :], d8_flow[-1, :] = 8, 8, 8, 8

    d8_offset = np.take(idx_offset, d8_flow, axis=0)
    idx = np.linspace(0, mn - 1, mn, dtype=int).reshape([m, n], order='C')

    for i in range(10):
        # get boundary
        Bnd = get_mask_boundary(C)

        sel_idx = idx[Bnd]  # pixel locations of interest
        sel_off = d8_offset[Bnd]  # associated offset to use
        downstream_idx = sel_idx + sel_off

        # update
        C.ravel()[downstream_idx] = C[Bnd]
    return C
