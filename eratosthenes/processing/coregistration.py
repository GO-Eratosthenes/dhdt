import numpy as np
import statsmodels.api as sm

from osgeo import gdal
from scipy.linalg import block_diag

from .matching_tools import getGridAtTemplateCenters, LucasKanade
from .network_tools import getNetworkIndices, getNetworkBySunangles, \
    getAjacencyMatrixFromNetwork
from .handler_s2 import read_view_angles_S2
from ..generic.mapping_tools import castOrientation, rotMat
from ..preprocessing.read_s2 import read_sun_angles_S2


def coregistration(sat_path, datPath, connectivity=2, stepSize=True,
                   tempSize=15, bbox=None, sig_y=10, lstsq_mode='simple'):
    """

    :return:
    """

    # construct network
    GridIdxs = getNetworkIndices(len(sat_path))
    GridIdxs = getNetworkBySunangles(datPath, sat_path, connectivity)
    Astack = getAjacencyMatrixFromNetwork(GridIdxs, len(sat_path))

    if lstsq_mode in ('weighted', 'generalized'):
        # get observation angle
        sen2Path = datPath + sat_path[0]
        (obsZn, obsAz) = read_view_angles_S2(sen2Path)
        if bbox is not None:
            obsZn = obsZn[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            obsAz = obsAz[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        if stepSize:  # reduce to kernel resolution
            obsZn = getGridAtTemplateCenters(obsZn, tempSize)
            obsAz = getGridAtTemplateCenters(obsAz, tempSize)

    Dstack = np.zeros(GridIdxs.T.shape)
    for i in range(GridIdxs.shape[1]):
        for j in range(GridIdxs.shape[0]):
            sen2Path = datPath + sat_path[GridIdxs[j, i]]
            # get sun orientation
            (_, sunAz) = read_sun_angles_S2(sen2Path)
            if bbox is not None:
                sunAz = sunAz[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # read shadow image
            img = gdal.Open(sen2Path + "shadows.tif")
            M = np.array(img.GetRasterBand(1).ReadAsArray())
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

        (y_N, y_E) = LucasKanade(Mstack[:, :, 0], Mstack[:, :, 1],
                                 tempSize, stepSize)
        # select correct displacements
        Msk = y_N != 0

        #########################
        # also get stable ground?
        #########################

        # robust selection through group statistics
        med_N = np.median(y_N[Msk])
        mad_N = np.median(np.abs(y_N[Msk] - med_N))

        med_E = np.median(y_E[Msk])
        mad_E = np.median(np.abs(y_E[Msk] - med_E))

        # robust 3sigma-filtering
        alpha = 3
        IN_N = (y_N > med_N - alpha * mad_N) & (y_N < med_N + alpha * mad_N)
        IN_E = (y_E > med_E - alpha * mad_E) & (y_E < med_E + alpha * mad_E)
        IN = IN_N & IN_E
        del IN_N, IN_E, med_N, mad_N, med_E, mad_E

        # keep selection and get their observation angles
        IN = IN & Msk
        y_N = y_N[IN]
        y_E = y_E[IN]

        # construct measurement vector and variance matrix
        A = np.tile(np.identity(2, dtype=float), (len(y_N), 1))
        y = np.reshape(np.hstack((y_N[:, np.newaxis], y_E[:, np.newaxis])),
                       (-1, 1))

        if lstsq_mode == 'simple':
            xHat = np.linalg.lstsq(A, y, rcond=None)
            xHat = xHat[0]
        elif lstsq_mode == 'generalized':
            # Generalized Least Squares adjustment, as there might be some
            # heteroskedasticity
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
    f = open(datPath + 'coreg.txt', 'w')
    for i in range(xCoreg.shape[0]):
        line = sat_path[i] + ' ' + '{:+3.4f}'.format(
            xCoreg[i, 0]) + ' ' + '{:+3.4f}'.format(xCoreg[i, 1])
        f.write(line + '\n')
    f.close()
