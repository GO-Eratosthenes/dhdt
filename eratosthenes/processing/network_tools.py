import numpy as np

from sklearn.neighbors import NearestNeighbors

from .handler_s2 import read_mean_sun_angles_s2


def getNetworkIndices(n):  # processing
    """
    Generate a list with all matchable combinations
    input:   n              integer           number of images
    output:  GridIdxs       array (2 x k)     list of couples
    """
    Grids = np.indices((n, n))
    Grid1 = np.triu(Grids[0] + 1, +1).flatten()
    Grid1 = Grid1[Grid1 != 0] - 1
    Grid2 = np.triu(Grids[1] + 1, +1).flatten()
    Grid2 = Grid2[Grid2 != 0] - 1
    GridIdxs = np.vstack((Grid1, Grid2))
    return GridIdxs


def getNetworkBySunangles(scene_paths, n):  # processing
    """
    Construct network, connecting elements with closest sun angle with each
    other.
    input:   datPath        string            location of the imagery
             scenceList     list              list with strings of the images
                                              of interest
             n              integer           connectivity
    output:  GridIdxs       array (2 x k)     list of couples
    """
    # can not be more connected than the amount of entries
    n = min(len(scene_paths) - 1, n)

    # get sun-angles from the imagery into an array
    L = np.zeros((len(scene_paths), 2), 'float')
    for i, scene_path in enumerate(scene_paths):
        (sunZn, sunAz) = read_mean_sun_angles_s2(scene_path)
        L[i, :] = [sunZn, sunAz]
    # find nearest
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='auto').fit(L)
    distances, indices = nbrs.kneighbors(L)
    Grid2 = indices[:, 1:]
    Grid1, dummy = np.indices((len(scene_paths), n))
    GridIdxs = np.vstack((Grid1.flatten(), Grid2.flatten()))
    return GridIdxs


def getAdjacencyMatrixFromNetwork(GridIdxs, number_of_nodes):
    """
    Transforms an edge list into an Adjacency matrix
    input:   GridIdxs        array (2 x k)    list of couples
             number_of_nodes integer          amount of nodes in the network
    output:  Astack          array (m x l)    design matrix
    """
    # General coregistration adjustment matrix
    Astack = np.zeros([GridIdxs.shape[1], number_of_nodes])
    Astack[np.arange(GridIdxs.shape[1]), GridIdxs[0, :]] = +1
    Astack[np.arange(GridIdxs.shape[1]), GridIdxs[1, :]] = -1
    return Astack
