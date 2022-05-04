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
    """
    if isinstance(n,int):
        grids = np.indices((n, n))
    else:
        grids = np.meshgrid(n,n)
    grid_1 = np.triu(grids[0] + 1, +1).flatten()
    grid_1 = grid_1[grid_1 != 0] - 1
    grid_2 = np.triu(grids[1] + 1, +1).flatten()
    grid_2 = grid_2[grid_2 != 0] - 1
    grid_idxs = np.vstack((grid_1, grid_2))
    return grid_idxs

def getNetworkBySunangles(datPath, sceneList, n):  # processing
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
    n = min(len(sceneList) - 1, n)

    # get sun-angles from the imagery into an array
    L = np.zeros((len(sceneList), 2), 'float')
    for i in range(len(sceneList)):
        sen2Path = datPath + sceneList[i]
        (sunZn, sunAz) = read_mean_sun_angles_s2(sen2Path)
        L[i, :] = [sunZn, sunAz]
    # find nearest
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='auto').fit(L)
    distances, indices = nbrs.kneighbors(L)
    Grid2 = indices[:, 1:]
    Grid1, dummy = np.indices((len(sceneList), n))
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
