import os

import numpy as np
from numpy.lib.recfunctions import stack_arrays, merge_arrays
from scipy import sparse

from ..generic.mapping_tools import map2pix, pix2map
from ..generic.handler_im import bilinear_interpolation
from ..processing.coupling_tools import \
    pair_posts, get_elevation_difference, angles2unit
from ..processing.matching_tools import remove_posts_pairs_outside_image
from ..processing.network_tools import get_network_indices

def read_conn_files_to_stack(folder_list, conn_file="conn.txt",
                             folder_path=None):
    """

    Parameters
    ----------
    folder_list : list of strings
        locations where connectivity files are located
    conn_file : string
        name of the connectivity file
    folder_path : string
        root location where all processing folders of the imagery are situated

    Returns
    -------
    dh : numpy.recordarray
        stack of coordinates and times, having the following collumns:
            - timestamp : date stamp
            - caster_x, caster_y : map coordinate of start of shadow
            - casted_x, casted_y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees
    """

    dh_stack = None
    for im_folder in folder_list:
        if folder_path is None:
            conn_path = os.path.join(im_folder, conn_file)
        else:
            conn_path = os.path.join(folder_path, im_folder, conn_file)

        if not os.path.exists(conn_path):
            continue

        im_date = np.datetime64(im_folder[3:])
        C = np.loadtxt(conn_path)
        desc = np.dtype([('timestamp', '<M8[D]'),
                         ('caster_X', np.float64), ('caster_Y', np.float64),
                         ('casted_X', np.float64), ('casted_Y', np.float64),
                         ('azimuth', np.float64), ('zenith', np.float64)])
        dh = np.rec.fromarrays([np.tile(im_date, C.shape[0]),
                                C[:,0], C[:,1], C[:,2], C[:,3], C[:,4], C[:,5] ],
                               dtype=desc)
        del C, im_date
        if dh_stack is None:
            dh_stack = dh.copy()
            continue

        # pair coordinates of the new list to older entries, through
        # a unique counter, given by "id"
        idx_uni = pair_posts(np.stack((dh_stack['caster_X'],
               dh_stack['caster_Y'])).T, np.stack((dh['caster_X'],
               dh['caster_Y'])).T, thres=10)

        id_counter = 1
        if 'id' in dh_stack.dtype.names:
            id_counter = np.max(dh_stack['id'])+1
            id_dh = np.zeros(dh.shape[0], dtype=int)
            # assign to older id's
            NEW = dh_stack['id'][idx_uni[:,0]]==0
            id_dh[idx_uni[np.invert(NEW),1]] = \
                dh_stack['id'][idx_uni[np.invert(NEW),0]]

            # create new ids
            id_dh[idx_uni[NEW, 1]] = id_counter + \
                                      np.arange(0,np.sum(NEW)).astype(int)
            dh_stack['id'][idx_uni[NEW, 0]] = id_counter + \
                                      np.arange(0,np.sum(NEW)).astype(int)
            id_dh = np.rec.fromarrays([id_dh],
                                      dtype=np.dtype([('id', int)]))
            dh = merge_arrays((dh, id_dh), flatten=True, usemask=False)
        else:
            id_stack = np.zeros(dh_stack.shape[0], dtype=int)
            id_dh = np.zeros(dh.shape[0], dtype=int)
            id_stack[idx_uni[:,0]] = \
                id_counter+np.arange(0,idx_uni.shape[0]).astype(int)
            id_dh[idx_uni[:,1]] = \
                id_counter+np.arange(0,idx_uni.shape[0]).astype(int)
            id_stack = np.rec.fromarrays([id_stack],
                                         dtype=np.dtype([('id', int)]))
            id_dh = np.rec.fromarrays([id_dh],
                                      dtype=np.dtype([('id', int)]))
            dh = merge_arrays((dh, id_dh), flatten=True, usemask=False)
            dh_stack = merge_arrays((dh_stack, id_stack),
                                    flatten=True, usemask=False)
        dh_stack = stack_arrays((dh, dh_stack), asrecarray=True, usemask=False)
    return dh_stack

def clean_dh(dh):
    if type(dh) in (np.recarray,):
        dh = dh[dh['id']!=0]
    else:
        dh = dh[dh[:,-1]!=0]
    return dh

def get_casted_elevation_difference(dh):
    """

    Parameters
    ----------
    dh : {numpy.array, numpy.recordarray}
        stack of coordinates and times, having the following collumns:
            - timestamp : unit=days, date stamp
            - caster_x,caster_y : unit=meter, map coordinate of start of shadow
            - casted_x,casted_y : unit=meter, map coordinate of end of shadow
            - azimuth : unit=degrees, sun orientation
            - zenith : unit=degrees, overhead angle of the sun

    Returns
    -------
    dxyt : {numpy.array, numpy.recordarray}
        data array with the following collumns:
            - x_1, y_1 : unit=meter, map coordinate at instance t_1
            - t_1 : unit=days, date stamp
            - x_2, y_2 : unit=meter, map coordinate at instance t_2
            - t_2 : unit=days, date stamp
            - dh_12 : unit=meter, estimated elevation change, between t_1 & t_2

    See Also
    --------
    read_conn_files_to_stack, get_hypsometric_elevation_change
    """
    if type(dh) in (np.recarray,):
        doRecArr = True
        names = ('X_1', 'Y_1', 'T_1', 'X_2', 'Y_2', 'T_2', 'dH_12')
        formats = [np.float64, np.float64, '<M8[D]',
                   np.float64, np.float64, '<M8[D]', np.float64]
        dxyt = np.rec.array(np.zeros((0,7)),
                            names=names, formats=formats)
        ids = np.unique(dh['id']).astype(int)
    else:
        doRecArr = False
        dxyt = np.zeros((0,7))
        ids = np.unique(dh[:,-1]).astype(int)

    for ioi in ids:
        if doRecArr:
            IN = dh['id'] == ioi
            dh_ioi = dh[IN]
        else:
            IN = dh[:,-1] == ioi
            dh_ioi = dh[IN,:]

        n = np.sum(IN).astype(int)
        network_id = get_network_indices(n).T

        if doRecArr:
            sun = angles2unit(dh_ioi['azimuth'])
            xy_1 = np.stack((dh_ioi['casted_X'][network_id[:, 0]],
                             dh_ioi['casted_Y'][network_id[:, 0]])).T
            xy_2 = np.stack((dh_ioi['casted_X'][network_id[:, 1]],
                             dh_ioi['casted_Y'][network_id[:, 1]])).T
            xy_t = np.stack((dh_ioi['caster_X'][network_id[:, 0]],
                             dh_ioi['caster_Y'][network_id[:, 0]])).T
        else:
            sun = dh_ioi[:,-2]
            xy_1 = np.stack((dh_ioi[:,3][network_id[:, 0]],
                             dh_ioi[:,4][network_id[:, 0]])).T
            xy_2 = np.stack((dh_ioi[:,3][network_id[:, 1]],
                             dh_ioi[:,4][network_id[:, 1]])).T
            xy_t = np.stack((dh_ioi[:,1][network_id[:, 0]],
                             dh_ioi[:,2][network_id[:, 0]])).T

        dz_ioi = get_elevation_difference(sun[network_id[:, 0]],
                                          sun[network_id[:, 1]],
                                          xy_1, xy_2, xy_t)
        if doRecArr:
            dxyt_line = np.rec.fromarrays([xy_1[:,0], xy_1[:,1],
                   dh_ioi['timestamp'][network_id[:,0]], xy_2[:,0], xy_2[:,1],
                   dh_ioi['timestamp'][network_id[:,1]], dz_ioi], names=names,
                   formats=formats)
            dxyt = stack_arrays((dxyt, dxyt_line),
                                asrecarray=True, usemask=False)
        else:
            dxyt_line = np.hstack((xy_1, dh_ioi[:,0][network_id[:,0]],
                                   xy_2, dh_ioi[:,0][network_id[:,1]],
                                   dz_ioi))
            dxyt = np.vstack((dxyt, dxyt_line))
    return dxyt

def get_relative_hypsometric_network(dxyt, Z, geoTransform):

    if type(dxyt) in (np.recarray,):
        i_1,j_1 = map2pix(geoTransform, dxyt['X_1'], dxyt['Y_1'])
        i_2,j_2 = map2pix(geoTransform, dxyt['X_2'], dxyt['Y_2'])
    else:
        i_1,j_1 = map2pix(geoTransform, dxyt[:,0], dxyt[:,1])
        i_2,j_2 = map2pix(geoTransform, dxyt[:,3], dxyt[:,4])

    i_1,j_1,i_2,j_2 = np.round(i_1), np.round(j_1), np.round(i_2), np.round(j_2)
    i_1,j_1,i_2,j_2 = remove_posts_pairs_outside_image(Z,i_1,j_1,Z,i_2,j_2)
    i_1,j_1,i_2,j_2 = i_1.astype(int), j_1.astype(int), i_2.astype(int), \
        j_2.astype(int)

    ij_stack = np.vstack((np.hstack((i_1[:,np.newaxis],j_1[:,np.newaxis])),
                          np.hstack((i_2[:,np.newaxis],j_2[:,np.newaxis]))))
    posts,idx_inv = np.unique(ij_stack, axis=0, return_inverse=True)

    m, n = i_1.shape[0], posts.shape[0]
    idx_y = np.concatenate((np.linspace(0,m-1,m).astype(int),
                            np.linspace(0,m-1,m).astype(int)))
    dat_A = np.concatenate((+1*np.ones((m,)), -1*np.ones((m,))))

    # create sparse and signed adjacency matrix
    A = sparse.csr_array((dat_A, (idx_y, idx_inv)), shape=(m, n))
    posts[:,0], posts[:,1] = pix2map(geoTransform, posts[:,0], posts[:,1])
    return posts, A

def get_hypsometric_elevation_change(dxyt, Z, geoTransform):
    """

    Parameters
    ----------
    dxyt : {numpy.array, numpy.recordarray}
        data array with the following collumns:
            - x_1, y_1 : map coordinate at instance t_1
            - t_1 : date stamp
            - x_2, y_2 : map coordinate at instance t_2
            - t_2 : date stamp
            - dh_12 : estimated elevation change, between t_1 & t_2
    Z : numpy.array, size=(m,n)
        array with elevation data
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of the array 'Z'

    Returns
    -------
    dhdt : {numpy.array, numpy.recordarray}
        data array with the following collumns:
            - t_1 : date stamp
            - t_2 : date stamp
            - z_12 : mid-point elevation
            - dz_12 : elevation change, between instance t_1 & t_2

    See Also
    --------
    get_casted_elevation_difference
    """

    if type(dxyt) in (np.recarray,):
        i_1,j_1 = map2pix(geoTransform, dxyt['X_1'], dxyt['Y_1'])
        i_2,j_2 = map2pix(geoTransform, dxyt['X_2'], dxyt['Y_2'])
    else:
        i_1,j_1 = map2pix(geoTransform, dxyt[:,0], dxyt[:,1])
        i_2,j_2 = map2pix(geoTransform, dxyt[:,3], dxyt[:,4])

    Z_1 = bilinear_interpolation(Z, i_1, j_1),
    Z_2 = bilinear_interpolation(Z, i_2, j_2)

    # calculate mid point elevation
    Z_12 = bilinear_interpolation(Z, (i_1+i_2)/2, (j_1+j_2)/2)

    dZ = np.squeeze(Z_1) - np.squeeze(Z_2)
    if type(dxyt) in (np.recarray,):
        dz_12 = dxyt['dH_12'] - dZ
    else:
        dz_12 = dxyt[:,-1] - dZ

    if isinstance(dxyt, np.recarray):
        desc = np.dtype([('T_1', '<M8[D]'), ('T_2', '<M8[D]'),
                         ('Z_12', np.float64), ('dZ_12', np.float64)])
        dhdt = np.rec.fromarrays([dxyt['T_1'], dxyt['T_2'], Z_12, dz_12],
                                 dtype=desc)
        return dhdt
    else:
        dhdt = np.stack((dxyt[:,2], dxyt[:,5], Z_12, dz_12))
        return dhdt