import numpy as np
import statsmodels.api as sm

from scipy.linalg import block_diag

from .matching_tools import get_coordinates_of_template_centers, lucas_kanade
from .network_tools import getNetworkIndices, getNetworkBySunangles, \
    getAdjacencyMatrixFromNetwork
from .handler_s2 import read_view_angles_s2
from ..generic.mapping_tools import castOrientation, rotMat
from ..generic.mapping_io import read_geo_image
from ..generic.filtering_statistical import mad_filtering
from ..preprocessing.read_s2 import read_sun_angles_s2
from ..generic.handler_im import get_image_subset


def coregister(scene_paths, coreg_path, connectivity=2, step_size=True,
               temp_size=15, bbox=None, sig_y=10, lstsq_mode='ordinary',
               rgi_mask=None):
    """
    Find the relative position between imagery, using the illuminated direction
    of the image. A network of images is constructed, so a bundle block
    adjustment is possible, with associated error-propagation.

    input:   scene_paths    list              image names
             coreg_path     string or Path    file name where to save
                                              coregistration values
             connectivity   integer           amount of connections for each
                                              node in the network
             step_size      boolean           reduce sampling grid to kernel
                                              resolution, so estimates are
                                              uncorrelated
             temp_size      integer           size of the kernel, in pixels
             bbox           array (4 x 1)     array giving a subset, if this is
                                              used, instead of the full image
             sig_y          float             matching precision, in meters
             lstsq_mode     string            least squares adjustment method
                                              one can choose from:
                                                  - ordinary
                                                  - weighted
                                                  - generalized
             rgi_mask       array (m x n)     glacier/stable terrain mask
    """

    # construct network
    GridIdxs = getNetworkIndices(len(scene_paths))
    GridIdxs = getNetworkBySunangles(scene_paths, connectivity)
    Astack = getAdjacencyMatrixFromNetwork(GridIdxs, len(scene_paths))

    if lstsq_mode in ('weighted', 'generalized'):
        # get observation angle
        scene_path = scene_paths[0]
        (obsZn, obsAz) = read_view_angles_s2(scene_path)
        if bbox is not None:
            obsZn = get_image_subset(obsZn, bbox)
            obsAz = get_image_subset(obsAz, bbox)

    Dstack = np.zeros(GridIdxs.T.shape)
    for i in range(GridIdxs.shape[1]):
        for j in range(GridIdxs.shape[0]):
            scene_path = scene_paths[GridIdxs[j, i]]
            # get sun orientation
            (_, sunAz) = read_sun_angles_s2(scene_path)
            if bbox is not None:
                sunAz = sunAz[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # read shadow image
            M, _, _, _ = read_geo_image(scene_path/"shadows.tif")

            if i == 0 & j == 0:
                if step_size:  # reduce to kernel resolution
                    (sampleI, sampleJ) = get_coordinates_of_template_centers(
                                                M, temp_size)
                else:
                    # sampling grid will be a full resolution, for every pixel
                    (sampleJ, sampleI) = np.meshgrid(np.arange(M.shape[1]),
                                                     np.arangec(M.shape[0]))

            # create ridge image
            Mcan = castOrientation(M, sunAz)
            Mcan[Mcan < 0] = 0
            if j == 0:
                Mstack = Mcan
            else:
                Mstack = np.stack((Mstack, Mcan), axis=2)

        # optical flow following Lukas Kanade

        #####################
        # blur with gaussian?
        #####################

        (y_N, y_E) = lucas_kanade(Mstack[:, :, 0], Mstack[:, :, 1], temp_size,
                                  sampleI, sampleJ)

        # select correct displacements
        if rgi_mask is None:
            Msk = y_N != 0
        else:
            # also get stable ground?
            raise NotImplementedError('Not implemented yet!')

        # robust 3sigma-filtering
        cut_off = 3
        IN = np.logical_and(mad_filtering(y_N[Msk], cut_off),
                            mad_filtering(y_E[Msk], cut_off))

        # keep selection and get their observation angles
        (iIN, jIN) = np.nonzero(Msk)
        iIN = iIN[IN]
        jIN = jIN[IN]

        y_N = y_N[iIN, jIN]
        y_E = y_E[iIN, jIN]

        # construct design matrix and measurement vector
        A = np.tile(np.identity(2, dtype=float), (len(y_N), 1))
        y = np.reshape(np.hstack((y_N[:, np.newaxis], y_E[:, np.newaxis])),
                       (-1, 1))

        if lstsq_mode == 'ordinary':
            xHat = np.linalg.lstsq(A, y, rcond=None)
            xHat = xHat[0]
        elif lstsq_mode == 'generalized':
            # there might be some heteroskedasticity
            y_Zn = abs(obsZn[IN])  # no negative values permitted
            y_Az = obsAz[IN]

            # build 2*2*n covariance matrix
            q_yy = np.zeros((len(y_Zn), 2, 2), dtype=np.float)
            for b in range(len(y_Zn)):
                B = np.array([[sig_y, 0],
                              [0, (1+np.tan(np.radians(y_Zn[b])))*sig_y]])
                q_yy[b] = np.matmul(rotMat(y_Az[b]), B)
            # transform to diagonal matrix
            Q_yy = block_diag(*q_yy)
            del q_yy, y_N, y_E, y_Zn, y_Az
            gls_model = sm.GLS(A, y, sigma=Q_yy)
            xHat = gls_model.fit()
        elif lstsq_mode == 'weighted':
            raise NotImplementedError('Not implemented yet!')

        Dstack[i, :] = np.transpose(xHat)

    xCoreg = np.linalg.lstsq(Astack, Dstack, rcond=None)
    xCoreg = xCoreg[0]

    # write co-registration
    f = open(coreg_path, 'w')
    for i in range(xCoreg.shape[0]):
        line = str(scene_paths[i]) + ' ' + '{:+3.4f}'.format(
            xCoreg[i, 0]) + ' ' + '{:+3.4f}'.format(xCoreg[i, 1])
        f.write(line + '\n')
    f.close()


def get_coregistration(coreg_path, im_list=None):
    """
    The co-registration parameters are written in a specific file. This
    function retrieves these parameters, and finds the corresponding values
    of the images given by im_list

    input:   coreg_path     string or Path    location of coregistration file
             im_list        list (k x 1)      image names
    output:  co_name        list (k x 1)      code of the image or image name
             co_reg         array  (k x 2)    relative coordinates of the
                                              network adjustment
    """
    # read file
    with open(coreg_path) as f:
        lines = f.read().splitlines()
        co_name = [line.split(' ')[0] for line in lines]
        co_reg = np.array([list(map(float, line.split(' ')[1:]))
                           for line in lines])
    del lines

    # make a selection
    if im_list is not None:
        raise NotImplementedError('Not implemented yet!')

    return co_name, co_reg
