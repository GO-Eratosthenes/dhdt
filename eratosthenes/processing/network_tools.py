import numpy as np

from sklearn.neighbors import NearestNeighbors

from .handler_s2 import read_mean_sun_angles_S2


def getNetworkIndices(n):  # processing
    """
    generate a list with all matchable combinations
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
        (sunZn, sunAz) = read_mean_sun_angles_S2(sen2Path)
        L[i, :] = [sunZn, sunAz]
    # find nearest
    nbrs = NearestNeighbors(n_neighbors=n+1, algorithm='auto').fit(L)
    distances, indices = nbrs.kneighbors(L)
    Grid2 = indices[:, 1:]
    Grid1, dummy = np.indices((len(sceneList), n))
    GridIdxs = np.vstack((Grid1.flatten(), Grid2.flatten()))
    return GridIdxs


def getAjacencyMatrixFromNetwork(GridIdxs, number_of_nodes):
    """

    :param GridIdxs: grid indices
    :return:
    """
    # General coregistration adjustment matrix
    Astack = np.zeros([GridIdxs.shape[1], number_of_nodes])
    Astack[np.arange(GridIdxs.shape[1]), GridIdxs[0, :]] = +1
    Astack[np.arange(GridIdxs.shape[1]), GridIdxs[1, :]] = -1
    return Astack
