import numpy as np
import pandas as pd
from osgeo import osr

from dhdt.generic.mapping_tools import ecef2llh, ll2map, map2ll, pix_centers
from dhdt.generic.unit_check import (are_three_arrays_equal,
                                     are_two_arrays_equal,
                                     correct_geoTransform, is_crs_an_srs,
                                     lat_lon_angle_check)


def wgs84_param():
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    major_axis = wgs84.GetSemiMajor()
    flattening = 1 / wgs84.GetInvFlattening()
    return major_axis, flattening


def earth_eccentricity():
    major_axis, flattening = wgs84_param()
    eccentricity = (2 * flattening) - (flattening**2)
    return eccentricity


def earth_axes():
    major_axis, flattening = wgs84_param()
    eccentricity = earth_eccentricity()
    minor_axis = major_axis * np.sqrt(1 - eccentricity)
    return major_axis, minor_axis


def standard_gravity():
    """
    provide value for the standard gravitational parameter of the Earth

    Returns
    -------
    μ : float, unit: m**3 * s**-2
        standard gravity
    """
    μ = 3.986004418E14
    return μ


def transform_rpy_xyz2ocs(xyz, uvw, roll, pitch, yaw, xyz_time, ang_time):
    x = np.interp(ang_time, xyz_time, xyz[..., 0])
    y = np.interp(ang_time, xyz_time, xyz[..., 1])
    z = np.interp(ang_time, xyz_time, xyz[..., 2])
    u = np.interp(ang_time, xyz_time, uvw[..., 0])
    v = np.interp(ang_time, xyz_time, uvw[..., 1])
    w = np.interp(ang_time, xyz_time, uvw[..., 2])
    xyz, uvw = np.stack((x, y, z), axis=1), np.stack((u, v, w), axis=1)

    m, n = xyz.shape
    Z_ocs = np.divide(xyz, np.tile(np.linalg.norm(xyz, axis=-1), (n, 1)).T)
    X_ocs = np.cross(uvw, xyz, axisa=-1, axisb=-1)
    X_ocs = np.divide(X_ocs, np.tile(np.linalg.norm(X_ocs, axis=-1), (n, 1)).T)
    Y_ocs = np.cross(Z_ocs, X_ocs, axisa=-1, axisb=-1)
    Y_ocs = np.divide(Y_ocs, np.tile(np.linalg.norm(Y_ocs, axis=-1), (n, 1)).T)

    R_stack = np.dstack((X_ocs, Y_ocs, Z_ocs))
    rpy = np.stack((roll, pitch, yaw), axis=1)

    np.einsum('...i,...i->...i', R_stack, rpy)

    print('.')

    return roll, pitch, yaw


def estimate_inclination_via_xyz_uvw(xyz, uvw):
    xyz, uvw = np.atleast_2d(xyz), np.atleast_2d(uvw)
    m, n = xyz.shape
    h = np.cross(xyz, uvw)
    h_mag = np.linalg.norm(h, axis=-1)
    h = np.divide(h, np.tile(h_mag, (n, 1)).T)

    z = np.array([0, 0, 1])
    i = 90 - np.rad2deg(np.arccos(np.dot(h, z)))
    i += 90
    return i


def calculate_correct_mapping(Zn_grd,
                              Az_grd,
                              bnd,
                              det,
                              grdTransform,
                              crs,
                              sat_dict=None):
    """

    Parameters
    ----------
    Zn_grd, Az_grd : {numpy.ndarray, numpy.masked.array}, size=(k,l,h)
        observation angles of the different detectors/bands
    bnd : numpy.ndarray, size=(h,)
        number of the band, corresponding to the third dimension of 'Zn_grd'
    det : numpy.ndarray, size=(h,)
        number of the detector, corresponding to the third dimension of
        'Zn_grd'
    grdTransform : tuple, size=(8,)
        geotransform of the grid of 'Zn_grd' and 'Az_grd'
    crs : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    sat_dict : dictonary
        metadata of the satellite platform, preferably having the following
        elements:
            * 'inclination'
            * 'revolutions_per_day'
            * 'altitude'

    Returns
    -------
    Ltime : numpy.ndarray, size=(p,), unit=seconds
        asd
    lat, lon : float, unit=degrees
        ground location of the satelite at time 0
    radius : float, unit=metre
        distance away from Earths' center
    inclination : unit=degrees
        tilt of the orbital plane in relation to the equator
    period : float, unit=seconds
        time it takes to complete one revolution
    time_para : numpy.ndarray, size=(p,b)
        polynomial fitting parameters for the different bands (b)
    combos : numpy.ndarray, size=(h,2)
        combinations of band and detector pairs,
        corresponding to the third dimension of 'Zn_grd'
    """
    are_two_arrays_equal(Zn_grd, Az_grd)
    are_two_arrays_equal(bnd, det)
    grdTransform = correct_geoTransform(grdTransform)
    if not isinstance(crs, str):
        crs = crs.ExportToWkt()
    depth = 1 if Az_grd.ndim < 3 else Az_grd.shape[2]

    if is_crs_an_srs(crs):
        X_grd, Y_grd = pix_centers(grdTransform, make_grid=True)
        (m, n) = X_grd.shape
        ll_grd = map2ll(np.stack((X_grd.ravel(), Y_grd.ravel()), axis=1), crs)
        Lat_grd = ll_grd[:, 0].reshape((m, n))
        Lon_grd = ll_grd[:, 1].reshape((m, n))
        del ll_grd
    else:
        Lon_grd, Lat_grd = pix_centers(grdTransform, make_grid=True)

    # remove NaN's, and create vectors
    IN = np.invert(np.isnan(Az_grd))
    Lat_grd = np.tile(np.atleast_3d(Lat_grd), (1, 1, depth))
    Lon_grd = np.tile(np.atleast_3d(Lon_grd), (1, 1, depth))
    Lat, Lon, Az, Zn = Lat_grd[IN], Lon_grd[IN], Az_grd[IN], Zn_grd[IN]

    Sat, Gx = line_of_sight(Lat, Lon, Zn, Az)
    del Lat, Lon, Lat_grd, Lon_grd, Zn_grd, Az_grd

    Ltime, lat, lon, radius, inclination, period = orbital_fitting(
        Sat, Gx, sat_dict=sat_dict)

    # vectorize band and detector indicators
    Bnd, Det = np.tile(bnd, (m, n, 1))[IN], np.tile(det, (m, n, 1))[IN]
    X = np.tile(np.atleast_3d(X_grd), (1, 1, depth))[IN]
    Y = np.tile(np.atleast_3d(Y_grd), (1, 1, depth))[IN]
    time_para, combos = time_fitting(Ltime, Az, Zn, Bnd, Det, X, Y,
                                     grdTransform)

    return Ltime, lat, lon, radius, inclination, period, time_para, combos


def remap_observation_angles(Ltime, lat, lon, radius, inclination, period,
                             time_para, combos, X_grd, Y_grd, det_stack,
                             bnd_list, geoTransform, crs):
    if type(bnd_list) in (pd.core.frame.DataFrame, ):
        bnd_list = np.asarray(bnd_list['bandid'])
        bnd_list -= 1  # numbering of python starts at 0
    lat, lon = lat_lon_angle_check(lat, lon)
    are_two_arrays_equal(X_grd, Y_grd)
    geoTransform = correct_geoTransform(geoTransform)

    m, n = X_grd.shape
    # bnd_list = np.asarray(sat_df['bandid'])

    omega_0, lon_0 = _omega_lon_calculation(np.deg2rad(lat), np.deg2rad(lon),
                                            inclination)

    # reconstruct observation angles and sensing time
    b = bnd_list.size
    if type(det_stack) in (np.ma.core.MaskedArray, ):
        T, Zn, Az = np.ma.zeros((m, n, b)), np.ma.zeros((m, n, b)), \
                    np.ma.zeros((m, n, b))
    else:
        T, Zn, Az = np.zeros((m, n, b)), np.zeros((m, n, b)), \
                    np.zeros((m, n, b))
    for idx, bnd in enumerate(bnd_list):
        doi = combos[:, 0] == bnd
        if type(det_stack) in (np.ma.core.MaskedArray, ):
            dT, Zen, Azi = -9999. * np.ma.ones((m, n)), \
                           -9999. * np.ma.ones((m, n)), \
                           -9999. * np.ma.ones((m, n))
        else:
            dT, Zen, Azi = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))

        for cnt, sca in enumerate(combos[doi, 1]):
            IN = det_stack[..., idx] == sca
            if not np.any(IN):
                continue
            dX, dY = X_grd[IN] - geoTransform[0], geoTransform[3] - Y_grd[IN]

            # time stamps
            coef_id = np.where(
                np.logical_and(combos[:, 0] == bnd, combos[:, 1] == sca))[0][0]
            coeffs = time_para[coef_id, :]
            dt = coeffs[
                0] + coeffs[1] * dX + coeffs[2] * dY + coeffs[3] * dX * dY
            dT[IN] = dt
            del dX, dY, coeffs, coef_id
            # acquisition angles
            Px = orbital_calculation(dt, radius, inclination, period, omega_0,
                                     lon_0)  # satellite vector

            ll_pix = map2ll(np.stack((X_grd[IN], Y_grd[IN]), axis=1), crs)
            Gx = np.transpose(ground_vec(ll_pix[:, 0],
                                         ll_pix[:, 1]))  # ground vector
            del ll_pix, dt

            zn, az = acquisition_angles(Px, Gx)
            Zen[IN], Azi[IN] = zn, az
            del Px, Gx
        # put estimates in stack

        if type(det_stack) in (np.ma.core.MaskedArray, ):
            dT, Zen, Azi = np.ma.array(dT, mask=dT == -9999.), \
                           np.ma.array(Zen, mask=Zen == -9999.), \
                           np.ma.array(Azi, mask=Azi == -9999.)
        T[..., idx], Zn[..., idx], Az[..., idx] = dT, Zen, Azi
        del Zen, Azi
    return Zn, Az, T


def get_absolute_timing(lat, lon, sat_dict):
    assert isinstance(sat_dict, dict), 'please provide a dictionary'
    assert ('gps_xyz' in sat_dict.keys()), 'please include GPS metadata'
    assert ('gps_tim' in sat_dict.keys()), 'please include GPS metadata'
    lat, lon = lat_lon_angle_check(lat, lon)

    cen_ll = np.array([lat, lon])
    sat_llh = ecef2llh(sat_dict['gps_xyz'])
    ll_diff = sat_llh[:, :2] - cen_ll[None, :]
    ll_dist = np.linalg.norm(ll_diff, axis=1)

    min_idx = np.argmin(ll_dist)
    # sub-grid interpolation
    dd = np.divide(
        ll_dist[min_idx + 1] - ll_dist[min_idx - 1], 2 *
        ((2 * ll_dist[min_idx]) - ll_dist[min_idx - 1] - ll_dist[min_idx + 1]))

    T = sat_dict['gps_tim']

    dT = np.abs(T[min_idx] - T[int(min_idx + np.sign(dd))])
    T_bias = T[min_idx] + dd * dT
    return T_bias


def acquisition_angles(Px, Gx):
    """ given satellite and ground coordinates, estimate observation angles"""
    are_two_arrays_equal(Px, Gx)

    major_axis, minor_axis = earth_axes()
    Vx = Px - Gx  # observation vector
    del Px
    Vdist = np.linalg.norm(Vx, axis=1)  # make unit length
    Vx = np.einsum('i...,i->i...', Vx, np.divide(1, Vdist))
    del Vdist

    e_Z = np.einsum('...i,i->...i', Gx,
                    1 / np.array([major_axis, major_axis, minor_axis]))
    e_E = np.zeros_like(e_Z)
    e_E[..., 0], e_E[..., 1] = -e_Z[:, 1].copy(), e_Z[:, 0].copy()
    e_plan = np.linalg.norm(e_Z[:, :2], axis=1)
    e_E = np.einsum('i...,i->i...', e_E, np.divide(1, e_plan))
    del e_plan
    e_N = np.array([
        np.multiply(e_Z[:, 1], e_E[:, 2]) - np.multiply(e_Z[:, 2], e_E[:, 1]),
        np.multiply(e_Z[:, 2], e_E[:, 0]) - np.multiply(e_Z[:, 0], e_E[:, 2]),
        np.multiply(e_Z[:, 0], e_E[:, 1]) - np.multiply(e_Z[:, 1], e_E[:, 0])
    ]).T

    LoS = np.zeros_like(e_Z)
    LoS[..., 0] = np.einsum('...i,...i->...', Vx, e_E)
    del e_E
    LoS[..., 1] = np.einsum('...i,...i->...', Vx, e_N)
    del e_N
    LoS[..., 2] = np.einsum('...i,...i->...', Vx, e_Z)
    del e_Z

    az = np.rad2deg(np.arctan2(LoS[..., 0], LoS[..., 1]))
    zn = np.rad2deg(np.arccos(LoS[..., 2]))
    return zn, az


def line_of_sight(Lat, Lon, Zn, Az, eccentricity=None, major_axis=None):
    """

    Parameters
    ----------
    Lat, Lon : numpy.ndarray, size=(m,n), unit=degrees
        location of observation angles
    Zn, Az : numpy.ndarray, size=(m,n), unit=degrees
        polar angles of line of sight to the satellite, also known as,
        delcination and right ascension.
    eccentricity: float
        eccentricity squared, if None default is WGS84 (6.69E-10)
    major_axis: float, unit=meters
        length of the major axis, if None default is WGS84 (6.37E6)

    Returns
    -------
    Sat : numpy.ndarray, size=(m*n,3)
        observation vector from ground location towards the satellite
    Gx : numpy.ndarray, size=(m*n,3)
        ground location in Cartesian coordinates
    """
    Lat, Lon = lat_lon_angle_check(Lat, Lon)
    assert len(
        set({Lat.shape[0], Lon.shape[0], Zn.shape[0], Az.shape[0]})) == 1, \
        ('please provide arrays of the same size')

    if (major_axis is None) or (eccentricity is None):
        major_axis = wgs84_param()[0]
        eccentricity = earth_eccentricity()

    Lat, Lon = np.deg2rad(Lat.flatten()), np.deg2rad(Lon.flatten())
    Zn, Az = np.deg2rad(Zn.flatten()), np.deg2rad(Az.flatten())

    # local tangent coordinate system
    e_E = np.stack((-np.sin(Lon), +np.cos(Lon), np.zeros_like(Lon)), axis=1)
    e_N = np.stack(
        (-np.sin(Lat) * np.cos(Lon), -np.sin(Lat) * np.sin(Lon), +np.cos(Lat)),
        axis=1)
    e_Z = np.stack(
        (np.cos(Lat) * np.cos(Lon), np.cos(Lat) * np.sin(Lon), np.sin(Lat)),
        axis=1)
    LoS = np.stack(
        (np.sin(Zn) * np.sin(Az), np.sin(Zn) * np.cos(Az), np.cos(Zn)), axis=1)

    Sat = np.array([
        LoS[:, 0] * e_E[:, 0] + LoS[:, 1] * e_N[:, 0] + LoS[:, 2] * e_Z[:, 0],
        LoS[:, 0] * e_E[:, 1] + LoS[:, 1] * e_N[:, 1] + LoS[:, 2] * e_Z[:, 1],
        LoS[:, 0] * e_E[:, 2] + LoS[:, 1] * e_N[:, 2] + LoS[:, 2] * e_Z[:, 2]
    ]).T
    Radius = np.divide(major_axis,
                       np.sqrt(1.0 - eccentricity * np.sin(Lat) * np.sin(Lat)))
    Gx = np.array([
        Radius * np.cos(Lat) * np.cos(Lon), Radius * np.cos(Lat) * np.sin(Lon),
        Radius * (1 - eccentricity) * np.sin(Lat)
    ]).T
    return Sat, Gx


def ground_vec(Lat, Lon, eccentricity=None, major_axis=None):
    """ get ground coordinates in Cartesian system

    Parameters
    ----------
    Lat, Lon : numpy.ndarray, size=(m,n), unit=degrees
        location of observation angles
    eccentricity: float
        eccentricity squared, if None default is WGS84 (6.69E-10)
    major_axis: float, unit=meters
        length of the major axis, if None default is WGS84 (6.37E6)

    Returns
    -------
    Gx : numpy.ndarray, size=(m*n,3)
        ground location in Cartesian coordinates

    """
    Lat, Lon = lat_lon_angle_check(Lat, Lon)
    are_two_arrays_equal(Lat, Lon)
    if (major_axis is None) or (eccentricity is None):
        major_axis, flattening = wgs84_param()
        eccentricity = (2 * flattening) - (flattening**2)

    Lat, Lon = np.deg2rad(Lat.flatten()), np.deg2rad(Lon.flatten())
    Radius = np.divide(major_axis,
                       np.sqrt(1.0 - eccentricity * np.sin(Lat) * np.sin(Lat)))
    Gx = np.array([
        Radius * np.cos(Lat) * np.cos(Lon), Radius * np.cos(Lat) * np.sin(Lon),
        Radius * (1 - eccentricity) * np.sin(Lat)
    ])
    return Gx


def _make_timing_system(IN, dX, dY, Ltime):
    are_three_arrays_equal(dX, dY, Ltime)
    DX, DY, LT = dX[IN], dY[IN], Ltime[IN]

    lt = np.array([LT, DX * LT, DY * LT, DX * DY * LT])
    L = np.sum(lt, axis=1)

    dx, dy, n = np.sum(DX), np.sum(DY), np.sum(IN)
    dx2, dxy, dy2 = np.sum(DX**2), np.sum(DX * DY), np.sum(DY**2)
    dx2y, dxy2, dx2y2 = np.sum(DX * DX * DY), np.sum(DX * DY * DY), np.sum(
        DX * DX * DY * DY)

    A = np.array([[n, dx, dy, dxy], [dx, dx2, dxy, dx2y], [dy, dxy, dy2, dxy2],
                  [dxy, dx2y, dxy2, dx2y2]])
    return A, L


def time_fitting(Ltime, Az, Zn, bnd, det, X, Y, geoTransform):
    are_two_arrays_equal(Zn, Az)
    are_two_arrays_equal(bnd, det)
    are_two_arrays_equal(X, Y)
    geoTransform = correct_geoTransform(geoTransform)

    dX, dY = X - geoTransform[0], geoTransform[3] - Y

    # look per band and detector
    combos, idx_inv = np.unique(np.stack((bnd, det), axis=1),
                                axis=0,
                                return_inverse=True)
    X_hat = np.zeros((combos.shape[0], 4))
    for idx, pair in enumerate(combos):
        IN = np.logical_and(bnd == pair[0], det == pair[1])
        A_sca, L_sca = _make_timing_system(IN, dX, dY, Ltime)
        try:
            np.linalg.inv(A_sca)
        except np.linalg.LinAlgError:
            if 'A_bnd' not in locals():
                IN = bnd == pair[0]
                A_bnd, L_bnd = _make_timing_system(IN, dX, dY, Ltime)
            A_sca, L_sca = A_bnd, L_bnd
        x_hat = np.transpose(np.linalg.inv(A_sca) @ L_sca[:, np.newaxis])
        X_hat[idx, ...] = x_hat
        if 'A_bnd' in locals():
            del A_bnd, L_bnd
    return X_hat, combos


def orbital_fitting(Sat,
                    Gx,
                    lat=None,
                    lon=None,
                    radius=None,
                    inclination=None,
                    period=None,
                    sat_dict=None,
                    convtol=0.001,
                    orbtol=1.0,
                    maxiter=20,
                    printing=False):
    """

    Parameters
    ----------
    Sat : numpy.ndarray, size=(m*n,3)
        observation vector from ground location towards the satellite
    Gx : numpy.ndarray, size=(m*n,3)
        ground location in Cartesian coordinates
    lat : float, unit=degrees
        location of satellite within orbital track
    lon : float, unit=degrees
        location of satellite within orbital track
    radius : float, unit=meter
        radius towards the orbiting satellite
    inclination : float, unit=degrees, range=-180...+180
        inclination of the orbital plane with the equator
    period : float, unit=seconds
        time it takes to revolve one time around the Earth
    sat_dict : dictonary

    Returns
    -------

    """
    are_two_arrays_equal(Sat, Gx)
    if (inclination is None) and (sat_dict is not None):
        inclination = np.deg2rad(sat_dict['inclination'])
    if (period is None) and (sat_dict is not None):
        period = (24 * 60 * 60) / sat_dict['revolutions_per_day']
    if radius is None:
        wgs84 = osr.SpatialReference()
        major_axis = wgs84.GetSemiMajor()
        if 'altitude' in sat_dict:
            radius = major_axis + np.mean(sat_dict['altitude'])
        else:
            radius = np.cbrt(
                np.divide(period, 2 * np.pi)**2 * standard_gravity())

    if (lat is None) or (lon is None):
        V_dist = np.sqrt(radius**2 + np.einsum('...i,...i', Sat, Gx)**2 -
                         np.linalg.norm(Gx, axis=1)**2)
        Px = Gx + np.einsum('i,ij->ij', V_dist, Sat)
        if 'gps_xyz' in sat_dict:  # use GPS trajectory
            poi = np.argmin(
                np.linalg.norm(sat_dict['gps_xyz'] - np.mean(Px, axis=0),
                               axis=1))
            Sx = sat_dict['gps_xyz'][poi, :]
            lat_bar = np.arctan2(Sx[2], np.linalg.norm(Sx[0:2]))
            lon_bar = np.arctan2(Sx[1], Sx[0])
            del poi, Sx
        else:  # use center of the scene as approximation
            Lon = np.arctan2(Px[..., 1], Px[..., 0])
            Lat = np.arctan2(Px[..., 2], np.linalg.norm(Px[..., 0:2], axis=1))
            lat_bar, lon_bar = np.mean(Lat), np.mean(Lon)
            del Lat, Lon
        del Px, V_dist

    numobs = Sat.shape[0]
    Ltime = np.zeros((numobs, ))

    rmstime, orbrss = 15.0, 1000.0
    counter = 0
    while (rmstime > convtol) or (orbrss > orbtol) or (counter < maxiter):
        omega_0, lon_0 = _omega_lon_calculation(lat_bar, lon_bar, inclination)

        A_0, L_0 = np.zeros((4, 4, numobs)), np.zeros((4, numobs))

        Vx = observation_calculation(Ltime, Sat, Gx, radius, inclination,
                                     period, omega_0, lon_0)

        # Calculate the partial derivatives w.r.t. the orbit parameters
        P_0 = partial_obs(Ltime, Sat, Gx, lat_bar, lon_bar, radius,
                          inclination, period)
        A_0 += np.einsum('ij...,ik...->jk...', P_0, P_0)
        L_0 += np.einsum('...i,ij...->j...', Vx, P_0)

        P_1 = partial_tim(Ltime, Sat, Gx, lat_bar, lon_bar, radius,
                          inclination, period)
        P_1 = np.squeeze(P_1)
        M_1 = np.einsum('ij...,i...->j...', P_0, P_1)
        A_1 = np.reciprocal(np.einsum('i...,i...->...', P_1, P_1))
        L_1 = np.multiply(np.einsum('i...,...i->...', P_1, Vx), A_1)
        del P_0, P_1, Vx

        A_0 += np.einsum('i...,j...->ij...', A_1 * M_1, M_1)
        L_0 += np.multiply(M_1, L_1)

        A_0, L_0 = np.sum(A_0, axis=2), np.sum(L_0, axis=1)[:, np.newaxis]
        N_1 = np.multiply(M_1, A_1)
        del M_1, A_1

        if counter != 0:
            X_0 = np.einsum('ij,i...->j...', np.linalg.inv(A_0), L_0)
        else:
            X_0 = np.zeros((4, ))

        # back substitute for time corrections
        dtime = L_1 - np.einsum('i...,i...->...', N_1, X_0)
        Ltime -= dtime
        rmstime = np.sum(dtime**2)
        del dtime, N_1, L_1

        # update orbit parameters
        lat_bar -= X_0[0]
        lon_bar -= X_0[1]
        radius -= X_0[2]
        inclination -= X_0[3]

        # evaluate convergence
        rmstime = np.sqrt(rmstime / numobs)
        # orbit convergence, transform to metric units
        X_0[0] *= 6378137.0
        X_0[1] *= 6378137.0
        X_0[3] *= radius
        orbrss = np.linalg.norm(X_0)
        counter += 1
        if printing:
            print('RMS Orbit Fit (meters): ', orbrss)
            print('RMS Time Fit (seconds): ', rmstime)
    lat_bar, lon_bar = np.rad2deg(lat_bar[0]), np.rad2deg(lon_bar[0])
    return Ltime, lat_bar, lon_bar, radius, inclination, period


def _omega_lon_calculation(ϕ, λ, inclination):
    """

    Parameters
    ----------
    ϕ, λ : {float, numpy.array}, unit=radians
        location on the Earth
    inclination : {float, numpy.array}, unit=radians
        angle of the orbital plane i.r.t. the equator
    """
    ω_0 = np.arcsin(np.divide(np.sin(ϕ), np.sin(inclination)))
    λ_0 = λ - np.arcsin(np.divide(np.tan(ϕ), -np.tan(inclination)))
    return ω_0, λ_0


def _gc_calculation(ltime, period, inclination, ω_0, λ_0):
    cta = ω_0 - np.divide(2 * np.pi * ltime, period)
    gclat = np.arcsin(np.sin(cta) * np.sin(inclination))
    gclon = (λ_0 + np.arcsin(np.tan(gclat) / -np.tan(inclination)) -
             2 * np.pi * ltime / (24 * 60 * 60))
    return cta, gclat, gclon


def orbital_calculation(ltime, radius, inclination, period, ω_0, λ_0):
    ltime, ω_0, λ_0 = np.squeeze(ltime), np.squeeze(ω_0), np.squeeze(λ_0)
    cta, gclat, gclon = _gc_calculation(ltime, period, inclination, ω_0, λ_0)
    Px = np.stack((np.multiply(np.cos(gclat), np.cos(gclon)),
                   np.multiply(np.cos(gclat), np.sin(gclon)), np.sin(gclat)),
                  axis=1)
    Px *= radius
    return Px


def observation_calculation(ltime, Sat, Gx, radius, inclination,
                            period, ω_0, λ_0):
    """

    Parameters
    ----------
    ltime : numpy.array, size=(m,1)
    Sat : numpy.ndarray, size=(m,3)
        observation vector from ground location towards the satellite
    Gx : numpy.ndarray, size=(m,3)
        ground location in Cartesian coordinates
    radius : float, unit=meter
        radius towards the orbiting satellite
    inclination : float, unit=degrees, range=-180...+180
        inclination of the orbital plane with the equator
    period : float, unit=seconds
        time it takes to revolve one time around the Earth
    ω_0 : float, unit=degrees
        angle towards ascending node, one of the orbit Euler angles
    λ_0 : float, unit=degrees
        ephemeris longitude

    Returns
    -------
    Vx : numpy.array, size=(m,3)
    """
    are_two_arrays_equal(Sat, Gx)

    cta, gclat, gclon = _gc_calculation(ltime, period, inclination, ω_0, λ_0)
    Vx = np.atleast_2d(np.zeros_like(Sat))
    Gx = np.atleast_2d(Gx)
    Vx[..., 0] = np.squeeze(radius * np.multiply(np.cos(gclat), np.cos(gclon)))
    Vx[..., 1] = np.squeeze(radius * np.multiply(np.cos(gclat), np.sin(gclon)))
    Vx[..., 2] = np.squeeze(radius * np.sin(gclat))
    Vx -= Gx

    Vdist = np.linalg.norm(Vx, axis=1)  # make unit length
    Vx = np.einsum('i...,i->i...', Vx, np.divide(1, Vdist))
    Vx -= Sat
    return Vx


def _pert_param(idx, pert, *args):
    """ perterp one of the arguments

    Parameters
    ----------
    idx : integer, range=0...m-1
        index which of the arguments should be perturbed
    pert : {float, numpy.array}
        amount of pertubation
    args : tuple, size=m
        tuple with different arguments, with datatypes like arrays or floats

    Returns
    -------
    args : tuple, size=m
        tuple with different arguments, with datatypes like arrays or floats
    """
    assert idx < len(args), 'please provide correct index'
    args = list(args)
    args[idx] += pert
    args = tuple(args)
    return args


def partial_obs(ltime, Sat, Gx, ϕ, λ, radius, inclination, period):
    """ numerical differentiation, via pertubation of the observation vector
    """
    are_two_arrays_equal(Sat, Gx)
    P_0 = np.zeros((3, 4, ltime.size))
    ω_0, λ_0 = _omega_lon_calculation(ϕ, λ, inclination)
    Dx = observation_calculation(ltime, Sat, Gx, radius, inclination,
                                 period, ω_0, λ_0)
    pert_var = ['lat', 'lon', 'radius', 'inclination']
    Pert = np.array([1E-5, 1E-5, 1E+1, 1E-4])
    for idx, pert in enumerate(Pert):
        (ϕ, λ, radius, inclination) = _pert_param(idx, +pert, ϕ, λ,
                                                  radius, inclination)

        ω_0, λ_0 = _omega_lon_calculation(ϕ, λ, inclination)
        Dp = observation_calculation(ltime, Sat, Gx, radius,
                                     inclination, period, ω_0, λ_0)
        P_0[0, idx, :] = np.divide(Dp[:, 0] - Dx[:, 0], pert)
        P_0[1, idx, :] = np.divide(Dp[:, 1] - Dx[:, 1], pert)
        P_0[2, idx, :] = np.divide(Dp[:, 2] - Dx[:, 2], pert)
        (ϕ, λ, radius, inclination) = _pert_param(idx, -pert, ϕ, λ,
                                                  radius, inclination)
    return P_0


def partial_tim(ltime, Sat, Gx, ϕ, λ, radius, inclination, period,
                pertubation=.1):
    are_two_arrays_equal(Sat, Gx)
    P_1 = np.zeros((3, 1, ltime.size))
    ω_0, λ_0 = _omega_lon_calculation(ϕ, λ, inclination)
    Dx = observation_calculation(ltime, Sat, Gx, radius, inclination,
                                 period, ω_0, λ_0)

    # pertubation in the time domain
    ltime += pertubation
    Dp = observation_calculation(ltime, Sat, Gx, radius, inclination,
                                 period, ω_0, λ_0)
    ltime -= pertubation

    P_1[0, 0, ...] = np.divide(Dp[..., 0] - Dx[..., 0], pertubation)
    P_1[1, 0, ...] = np.divide(Dp[..., 1] - Dx[..., 1], pertubation)
    P_1[2, 0, ...] = np.divide(Dp[..., 2] - Dx[..., 2], pertubation)
    return P_1
