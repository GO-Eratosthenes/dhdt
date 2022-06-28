import numpy as np

from sklearn.neighbors import NearestNeighbors

from ..input.read_sentinel2 import read_mean_sun_angles_s2

def get_network_indices(n):
    """ Generate a list with all matchable combinations

    Parameters
    ----------
    n : {integer, numpy.array}
        number of images or list with id's

    Returns
    -------
    grid_idxs : numpy.array, size=(2,k)
        list of couples

    See Also
    --------
    get_network_by_sunangle_s2 : same version, but with constrain on sun angle
    get_adjacency_matrix_from_netwrok : construct design matrix from edge list
    """
    if type(n) in (int, np.int64, np.int32, np.int16):
        grids = np.indices((n, n))
    else:
        grids = np.meshgrid(n,n)
    grid_1 = np.triu(grids[0] + 1, +1).flatten()
    grid_1 = grid_1[grid_1 != 0] - 1
    grid_2 = np.triu(grids[1] + 1, +1).flatten()
    grid_2 = grid_2[grid_2 != 0] - 1
    grid_idxs = np.vstack((grid_1, grid_2))
    return grid_idxs

def get_network_indices_constrained(idx,d,d_max,n_max):
    """ Generate a list with all matchable combinations within a certain range

    Parameters
    ----------
    idx : numpy.array, size=(m,_)
        list with indices
    d : numpy.array, size=(m,l)
        property to use for estimating closeness
    d_max : float,
        threshold for the maximum difference to still include
    n_max : integer
        maximum amount of nodes

    Returns
    -------
    grid_idxs : numpy.array, size=(2,k)
        list of couples

    See Also
    --------
    get_network_indices : same version, but more generic
    get_adjacency_matrix_from_netwrok : construct design matrix from edge list
    """
    n_max = min(len(idx) - 1, n_max)

    if d.ndim==1: d = d.reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=n_max+1, algorithm='auto').fit(d)
    distances, indices = nbrs.kneighbors(d)

    IN = distances[:, 1:]<=d_max

    grid2 = indices[:, 1:]
    grid1,_ = np.indices((len(idx), n_max))
    grid_idxs = np.vstack((grid1[IN], grid2[IN]))
    return grid_idxs

def get_network_by_sunangle_s2(datPath, sceneList, n):
    """ construct a network, connecting elements with closest sun angle with
    each other.

    Parameters
    ----------
    datPath : string
        location of the imagery
    scenceList : list
        list with strings of the Sentinel-2 imagery of interest
    n : integer
        amount of connectivity of the network

    Returns
    -------
    grid_idxs : numpy.array, size=(2,k), dtype=integer
        list indices giving couples

    See Also
    --------
    get_network_indices : simple version, without constrains
    get_adjacency_matrix_from_netwrok : construct design matrix from edge list
    """
    # can not be more connected than the amount of entries
    n = min(len(sceneList) - 1, n)

    # get sun-angles from the imagery into an array
    L = np.zeros((len(sceneList), 2), 'float')
    for i in range(len(sceneList)):
        sen2Path = datPath + sceneList[i]
        (sunZn, sunAz) = read_mean_sun_angles_s2(sen2Path)
        L[i, :] = [sunZn, sunAz]
    d_max = np.inf
    grid_idxs = get_network_indices_constrained(np.arange(len(sceneList)),
                                                L, d_max, n)
    return grid_idxs

def get_adjacency_matrix_from_network(GridIdxs, number_of_nodes):
    """ transforms an edge list into an adjacency matrix, this is a general
    co-registration adjustment matrix

    Parameters
    ----------
    grd_ids : numpy.array, size=(2,k), dtype=integer
        array with list of couples
    number_of_nodes integer: integer
        amount of nodes in the network

    Returns
    -------
    A : numpy.array, size=(m,l)
        design matrix

    References
    ----------
    .. [1] Altena & Kääb. "Elevation change and improved velocity retrieval
       using orthorectified optical satellite data from different orbits"
       Remote sensing vol.9(3) pp.300 2017.
    """
    A = np.zeros([GridIdxs.shape[1], number_of_nodes])
    A[np.arange(GridIdxs.shape[1]), GridIdxs[0, :]] = +1
    A[np.arange(GridIdxs.shape[1]), GridIdxs[1, :]] = -1
    return A
