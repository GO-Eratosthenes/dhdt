import os
import re
import warnings

import numpy as np
import pandas as pd
from numpy.lib.recfunctions import stack_arrays, merge_arrays
from scipy import ndimage

from dhdt.generic.debugging import loggg
from dhdt.generic.mapping_tools import map2pix, pix2map
from dhdt.generic.handler_im import simple_nearest_neighbor
from dhdt.processing.coupling_tools import \
    pair_posts, merge_ids, get_elevation_difference, angles2unit
from dhdt.processing.matching_tools import remove_posts_pairs_outside_image
from dhdt.processing.network_tools import get_network_indices
from dhdt.postprocessing.temporal_tools import get_temporal_incidence_matrix, \
    get_temporal_weighting

def get_conn_col_header():
    """ give the collumn name and datatypes

    photohypsometric connectivity textfiles can have different data collumns
    and be of different types. Such are typically given in the header. This
    function specifies its order, so connectivity files can be consistent.

    Returns
    -------
    col_names : list
        list with names of the collumns
    col_dtype : numpy.datatype
        specifies the datatypes of the different collumns
    col_frmt : tuple of strings
        specifies the format for export
    """
    col_names = ['timestamp', 'caster_X', 'caster_Y', 'caster_Z',
                 'casted_X', 'casted_Y', 'casted_X_refine', 'casted_Y_refine',
                 'azimuth', 'zenith', 'zenith_refrac','id','row',
                 'dh', 'dt']
    col_dtype = np.dtype([('timestamp', '<M8[D]'),
                          ('caster_X', np.float64), ('caster_Y', np.float64),
                          ('caster_Z', np.float64),
                          ('casted_X', np.float64), ('casted_Y', np.float64),
                          ('casted_X_refine', np.float64),
                          ('casted_Y_refine', np.float64),
                          ('azimuth', np.float64), ('zenith', np.float64),
                          ('zenith_refrac', np.float64), ('id', np.int32),
                          ('row', np.int32),
                          ('dh', np.float64), ('dt', '<m8[D]')])
    return col_names, col_dtype

def write_df_to_conn_file(df, conn_dir, conn_file="conn.txt", append=False):
    col_names, col_dtype = get_conn_col_header()

    conn_path = os.path.join(conn_dir, conn_file)

    df_sel = df[df.columns.intersection(list(col_names))]
    col_idx = df_sel.columns.get_indexer(list(col_names))
    IN = col_idx!=-1
    col_idx = col_idx[IN]

    if append and os.path.isfile(conn_path):
        f = open(conn_path, 'a+')
    else:
        f = open(conn_path, 'w')
        print('# '+' '.join(tuple(np.array(col_names)[IN])), file=f)

    # write rows of dataframe into text file
    for k in range(df_sel.shape[0]):
        line = ''
        for idx,val in enumerate(col_idx):
            col_oi = df_sel.columns[val]
            if col_oi in ('timestamp', 'date',):
                line += str(df_sel.iloc[k,val])[:10]
            elif col_oi in ('azimuth', 'zenith', 'zenith_refrac'):
                line += '{:+3.4f}'.format(df_sel.iloc[k,val])
            elif col_oi in ('caster_Z', 'casted_Z', 'dh'):
                line += '{:+4.2f}'.format(df_sel.iloc[k,val])
            elif col_oi in ('row', 'id'):
                line += "{:03d}".format(df_sel.iloc[k,val])
            else:
                line += '{:+8.2f}'.format(df_sel.iloc[k,val])
            line += ' '
        print(line[:-1], file=f) # remove last spacer
    f.close()
    return

def write_df_to_belev_file(df, belev_dir, rgi_num=0, rgi_region=0):
    belev_name = 'belev_' + str(rgi_region).zfill(2) + '.' + \
                 str(rgi_num).zfill(5) + '.dat.txt'
    belev_path = os.path.join(belev_dir, belev_name)
    f = open(belev_path, 'w')

    header = 'Elev  ' + "  ".join(list(map(str, df.columns)))
    print(header, file=f)

    func = lambda x: '{:+.2f}'.format(x)
    for index, row in df.iterrows():
        line = index + "  " + "  ".join(list(map(func, row)))
        print(line, file=f)
    f.close()
    return

def get_timestamp_conn_file(conn_path, no_lines=6):
    """ looks inside the comments of a .txt for the time stamp

    Parameters
    ----------
    conn_path : string
        path and filename towards the text file
    no_lines : integer, default=6
        amount of lines to scan in the text file

    Returns
    -------
    timestamp : numpy.datetime64

    See Also
    --------
    get_header_conn_file
    """
    assert isinstance(conn_path, str), ('please provide a string')
    assert os.path.isfile(conn_path), ('please provide a correct location')
    timestamp=None
    with open(conn_path) as file:
        lines = [next(file, '') for x in range(no_lines)]

    for line in lines:
        if line.startswith('#') and bool(re.search('time', line)):
            timing = line.split(':')[-1]
            timestamp = np.datetime64(timing.replace('\n', '').strip())
    return timestamp

def get_header_conn_file(conn_path, no_lines=6):
    """ looks inside the comments of a .txt for the time stamp

    Parameters
    ----------
    conn_path : string
        path and filename towards the text file
    no_lines : integer, default=6
        amount of lines to scan in the text file

    Returns
    -------
    header : list
        a list of strings

    See Also
    --------
    get_timestamp_conn_file
    """
    assert isinstance(conn_path, str), ('please provide a string')
    assert os.path.isfile(conn_path), ('please provide a correct location')
    header=None
    with open(conn_path) as file:
        lines = [next(file, '') for x in range(no_lines)]

    for line in lines:
        if line.startswith('#'):
            header = [names.strip() for names in line.split(' ')[1:]]
    return header

def read_conn_to_df(conn_path, conn_file='conn.txt'):
    if os.path.isdir(conn_path):
        conn_path = os.path.join(conn_path, conn_file)
    col_names = get_header_conn_file(conn_path)
    dh = pd.read_csv(conn_path, delim_whitespace=True,
                     header=None, names=col_names, comment='#')

    if 'timestamp' not in col_names:
        im_date = get_timestamp_conn_file(conn_path)
        if im_date is not None:
            t = np.tile(im_date, dh.shape[0])
            dh.insert(0, 'timestamp', t)
    return dh

def read_conn_files_to_stack(folder_list, conn_file="conn.txt",
                             folder_path=None):
    """ read shadow line text file(s) into numpy.recordarray

    Parameters
    ----------
    folder_list : tuple of strings
        locations where connectivity files are located
    conn_file : {string, tuple of strings}
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
    if isinstance(conn_file, tuple):
        conn_counter = len(conn_file)
    else:
        conn_counter = len(folder_list)

    dh_stack = None
    for i in range(conn_counter):
        if isinstance(conn_file, tuple):
            c_file = conn_file[i]
        else:
            c_file = conn_file

        if folder_path is None:
            conn_path = os.path.join(folder_list[i], c_file)
        elif folder_list is None:
            conn_path = os.path.join(folder_path, c_file)
        else:
            conn_path = os.path.join(folder_path, folder_list[i], c_file)

        if not os.path.isfile(conn_path):
            continue

        if isinstance(conn_file, tuple):
            im_date = np.datetime64('2018-05-16')
            print('OBS: taking arbitrary date')
        else:
            im_date = np.datetime64(folder_list[i][3:])

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

def read_conn_files_to_df(folder_list, conn_file="conn.txt",
                             folder_path=None, dist_thres=10.):
    """ read shadow line text file(s) into pandas.DataFrame

    Parameters
    ----------
    folder_list : {string, tuple of strings}
        locations where connectivity files are located
    conn_file : {string, tuple of strings}
        name of the connectivity file
    folder_path : string
        root location where all processing folders of the imagery are situated
    dist_thres : float, unit=m
        when combining caster posts, what is the maximum distance to tolerate a
        match, and create an id.

    Returns
    -------
    dh : pandas.DataFrame
        DataFrame of coordinates and times, having the following collumns:
            - timestamp : date stamp
            - caster_x, caster_y : map coordinate of start of shadow
            - casted_x, casted_y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees

    Notes
    -----
    The nomenclature is as follows:

        .. code-block:: text

          * sun
           \
            \ <-- sun trace
             \
             |\ <-- caster
             | \
             |  \                    ^ Z
             |   \                   |
         ----┴----+  <-- casted      └-> {X,Y}

    The angles related to the sun are as follows:

        .. code-block:: text

          surface normal              * sun
          ^                     ^    /
          |                     |   /
          |-- zenith angle      |  /
          | /                   | /|
          |/                    |/ | elevation angle
          └----                 └------ {X,Y}

        The azimuth angle declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & Y
                 |
            - <--|--> +
                 |
                 +----> East & X

    The text file has the following columns:

        .. code-block:: text

            * conn.txt
            ├ caster_X      <- map location of the start of the shadow trace
            ├ caster_Y
            ├ casted_X
            ├ casted_Y      <- map location of the end of the shadow trace
            ├ azimuth       <- argument of the illumination direction
            └ zenith        <- off-normal angle of the sun without atmosphere
    """
    assert isinstance(folder_list, tuple)^isinstance(conn_file, tuple), \
        ('one should be a tuple, while the other should be a string')
    if isinstance(conn_file, tuple):
        conn_counter = len(conn_file)
    else:
        conn_counter = len(folder_list)

    dh_df = None
    for i in range(conn_counter):
        c_file = conn_file[i] if isinstance(conn_file, tuple) else conn_file

        if folder_path is None:
            conn_path = os.path.join(folder_list[i], c_file)
        elif folder_list is None:
            conn_path = os.path.join(folder_path, c_file)
        elif isinstance(folder_list, tuple):
            conn_path = os.path.join(folder_path, folder_list[i], c_file)
        else:
            conn_path = os.path.join(folder_path, folder_list, c_file)

        if not os.path.isfile(conn_path):
            continue

        dh = read_conn_to_df(conn_path)

        if dh_df is None:
            dh_df = dh.copy()
            continue

        # pair coordinates of the new list to older entries, through
        # a unique counter, given by "id"
        idx_uni = pair_posts(dh_df[['caster_X','caster_Y']].to_numpy(),
                             dh[['caster_X', 'caster_Y']].to_numpy(),
                             thres=dist_thres)
        dh_df = merge_ids(dh_df, dh, idx_uni)
    return dh_df

def clean_dh(dh):
    """ multi-temporal photohypsometric data can have common caster locations.
    An identification number can be given to them. If this is not done, these
    data are redunant, and can be removed. This function does that.

    Parameters
    ----------
    dh : {numpy.ndarray, numpy.recordarray, pandas.DataFrame}
        array with photohypsometric information

    Returns
    -------
    dh : {numpy.ndarray, numpy.recordarray, pandas.DataFrame}
        reduced array

    See Also
    --------
    read_conn_files_to_stack, read_conn_files_to_df
    """
    if type(dh) in (np.recarray,):
        dh = clean_dh_rec(dh)
    elif type(dh) in (np.ndarray,):
        dh = clean_dh_np(dh)
    else:
        dh = clean_dh_pd(dh)
    return dh

def clean_dh_rec(dh):
    return dh[dh['id']!=0]

def clean_dh_np(dh):
    return dh[dh[:,-1]!=0]

@loggg
def clean_dh_pd(dh):
    if 'id' in dh.columns:
        dh.drop(dh[dh['id'] == 0].index, inplace=True)
    return dh

@loggg
def keep_refined_locations(dh):
    if np.all([header in dh.columns for header in
               ('casted_X_refine', 'casted_Y_refine')]):
        # only keep data points that we able to refine their position
        dh.drop(dh[dh['casted_X_refine'] == dh['casted_X']].index,
                inplace=True)
    return dh

def update_caster_elevation(dh, Z, geoTransform):
    """ include the elevation of the caster location (the start of the shadow
    trace) as a collumn. Which has the name 'caster_z'

    Parameters
    ----------
    dh : {numpy.ndarray, numpy.recordarray, pandas.DataFrame}
        array with photohypsometric information
    Z : numpy.ndarray, size=(m,n), unit=meters
        grid with elevation values
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients

    Returns
    -------
    dh : pandas.DataFrame
        DataFrame of coordinates and times, having at least the following
        collumns:
            - caster_z : elevation of the start of shadow
                (the newly created collumn)
            - timestamp : date stamp
            - caster_x, caster_y : map coordinate of start of shadow
            - casted_x, casted_y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees

    See Also
    --------
    update_casted_elevation

    Notes
    -----
    The nomenclature is as follows:

        .. code-block:: text

          * sun
           \
            \ <-- sun trace
             \
             |\ <-- caster
             | \
             |  \                    ^ Z
             |   \                   |
         ----┴----+  <-- casted      └-> {X,Y}
    """
    if type(dh) in (pd.core.frame.DataFrame,):
        update_caster_elevation_pd(dh, Z, geoTransform)
    #todo : make the same function for recordarray
    return dh

@loggg
def update_caster_elevation_pd(dh, Z, geoTransform):
    assert np.all([header in dh.columns for header in
                   ('caster_X', 'caster_Y')])
    # get elevation of the caster locations
    i_im, j_im = map2pix(geoTransform,
                         dh['caster_X'].to_numpy(), dh['caster_Y'].to_numpy())
    dh_Z = ndimage.map_coordinates(Z, [i_im, i_im], order=1, mode='mirror')
    if not 'caster_Z' in dh.columns: dh['caster_Z'] = None
    dh.loc['caster_Z'] = dh_Z
    return dh

def update_casted_elevation(dxyt, Z, geoTransform):
    """ include the elevation of the casted location (the end of the shadow
    trace) as a collumn. Which has the name 'casted_z'

    Parameters
    ----------
    dh : {numpy.ndarray, numpy.recordarray, pandas.DataFrame}
        array with photohypsometric information
    Z : numpy.ndarray, size=(m,n), unit=meters
        grid with elevation values
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients

    Returns
    -------
    dh : pandas.DataFrame
        DataFrame of coordinates and times, having at least the following
        collumns:
            - casted_z : elevation of the end of shadow
                (the newly created collumn)
            - timestamp : date stamp
            - caster_x, caster_y : map coordinate of start of shadow
            - casted_x, casted_y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees

    See Also
    --------
    update_caster_elevation

    Notes
    -----
    The nomenclature is as follows:

        .. code-block:: text

          * sun
           \
            \ <-- sun trace
             \
             |\ <-- caster
             | \
             |  \                    ^ Z
             |   \                   |
         ----┴----+  <-- casted      └-> {X,Y}
    """
    if type(dxyt) in (pd.core.frame.DataFrame,):
        update_casted_elevation_pd(dxyt, Z, geoTransform)
    #todo : make the same function for recordarray
    return dxyt

@loggg
def update_casted_elevation_pd(dxyt, Z, geoTransform):
    assert np.all([header in dxyt.columns for header in
                   ('X_1', 'Y_1', 'X_2', 'Y_2')])
    # get elevation of the caster locations
    for i in range(2):
        x_str, y_str, z_str = 'X_'+str(i+1), 'Y_'+str(i+1), 'Z_'+str(i+1)
        i_im, j_im = map2pix(geoTransform,
                             dxyt[x_str].to_numpy(), dxyt[y_str].to_numpy())
        dh_Z = ndimage.map_coordinates(Z, [i_im, j_im], order=1, mode='mirror')
        if not z_str in dxyt.columns: dxyt[z_str] = None
        dxyt[z_str] = dh_Z
    return dxyt

def update_glacier_id(dh, R, geoTransform):
    """ include the glacier id (as given by R), where the shadow is casted on.
    This collumn has the name 'glacier_id'.

    Parameters
    ----------
    dh : {numpy.ndarray, numpy.recordarray, pandas.DataFrame}
        array with photohypsometric information
    R : numpy.ndarray, size=(m,n), dtype=int
        grid with glacier labels
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients

    Returns
    -------
    dh : pandas.DataFrame
        DataFrame of coordinates and times, having at least the following
        collumns:
            - glacier_id : id of the glacier where the shadow is casted on
                (the newly created collumn)
            - timestamp : date stamp
            - caster_x, caster_y : map coordinate of start of shadow
            - casted_x, casted_y : map coordinate of end of shadow
            - azimuth : sun orientation, unit=degrees
            - zenith : overhead angle of the sun, unit=degrees
    """
    if type(dh) in (pd.core.frame.DataFrame,):
        update_glacier_id_pd(dh, R, geoTransform)
    #todo : make the same function for recordarray
    return dh

@loggg
def update_glacier_id_pd(dh, R, geoTransform):
    assert np.all([header in dh.columns for header in
                   ('casted_X', 'casted_Y')])
    # get elevation of the caster locations
    i_im, j_im = map2pix(geoTransform,
                         dh['caster_X'].to_numpy(), dh['caster_Y'].to_numpy())
    rgi_id = simple_nearest_neighbor(R, i_im, j_im).astype(int)
    if not 'glacier_id' in dh.columns: dh['glacier_id'] = None
    dh.loc['glacier_id'] = rgi_id
    return dh

def get_casted_elevation_difference(dh):
    """ transform angles and locations, to spatial temporal elevation changes

    Parameters
    ----------
    dh : {pandas.DataFrame, numpy.array, numpy.recordarray}
        stack of coordinates and times, having at least the following collumns:
            - timestamp : unit=days, date stamp
            - caster_x,caster_y : unit=meter, map coordinate of start of shadow
            - casted_x,casted_y : unit=meter, map coordinate of end of shadow
            - casted_x_refine,casted_y_refine : unit=meter, map coordinate of
                end of shadow, though optimized.
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
            optional:
            - g_1, g_2 : integer, glacier identification

    See Also
    --------
    read_conn_files_to_stack, get_hypsometric_elevation_change

    Notes
    -----
    The nomenclature is as follows:

        .. code-block:: text

          * sun
           \
            \ <-- sun trace
             \
             |\ <-- caster
             | \
             |  \                    ^ Z
             |   \                   |
         ----┴----+  <-- casted      └-> {X,Y}
    """
    if type(dh) in (np.recarray,):
        dxyt = get_casted_elevation_difference_rec(dh)
    elif type(dh) in (np.ndarray,):
        dxyt = get_casted_elevation_difference_np(dh)
    else:
        dxyt = get_casted_elevation_difference_pd(dh)
    return dxyt

def get_casted_elevation_difference_pd(dh):
    assert 'id' in dh.columns, 'please couple the dataframe to other timestamps'
    # is refraction present
    if 'zenith_refrac' not in dh.columns:
        warnings.warn('OBS: refractured zenith does not seem to be calculated')
        zn_name = 'zenith'
    else:
        zn_name = 'zenith_refrac'

    if 'casted_X_refine' not in dh.columns:
        warnings.warn('OBS: refined cast locations not present')
        x_name, y_name = 'casted_X', 'casted_Y'
    else:
        x_name, y_name = 'casted_X_refine', 'casted_Y_refine'

    if 'glacier_id' not in dh.columns:
        names = ('X_1', 'Y_1', 'T_1', 'X_2', 'Y_2', 'T_2', 'dH_12')
        formats = [np.float64, np.float64, '<M8[D]',
                   np.float64, np.float64, '<M8[D]', np.float64]
    else:
        names = ('X_1', 'Y_1', 'T_1', 'G_1',
                 'X_2', 'Y_2', 'T_2', 'G_2', 'dH_12')
        formats = [np.float64, np.float64, '<M8[D]', np.int16,
                   np.float64, np.float64, '<M8[D]', np.int16, np.float64]

    dxyt = None
    ids = np.unique(dh['id']).astype(int)
    for ioi in ids:
        dh_ioi = dh[dh['id'] == ioi] # get

        n = dh_ioi.shape[0]
        if n<2: continue

        # sort in temporal progression
        dh_ioi = dh_ioi.iloc[np.argsort(dh_ioi['timestamp']), :]

        network_id = get_network_indices(n).T

        xy_1 = dh_ioi[['casted_X','casted_Y']].iloc[network_id[:,0]].to_numpy()
        xy_2 = dh_ioi[['casted_X','casted_Y']].iloc[network_id[:,1]].to_numpy()
        xy_t = dh_ioi[[x_name, y_name]].iloc[network_id[:,0]].to_numpy()
        zn_1 = dh_ioi[zn_name].iloc[network_id[:,0]].to_numpy()
        zn_2 = dh_ioi[zn_name].iloc[network_id[:,1]].to_numpy()

        dz_ioi = get_elevation_difference(zn_1, zn_2, xy_1, xy_2, xy_t)
        t_1 = dh_ioi['timestamp'].iloc[network_id[:,0]].to_numpy()
        t_2 = dh_ioi['timestamp'].iloc[network_id[:,1]].to_numpy()
        if 'glacier_id' not in dh.columns:
            dxyt_line = np.rec.fromarrays([xy_1[:,0], xy_1[:,1], t_1,
                                           xy_2[:,0], xy_2[:,1], t_2, dz_ioi],
                                          names=names, formats=formats)
        else:
            g_1 = dh_ioi['glacier_id'].iloc[network_id[:, 0]].to_numpy()
            g_2 = dh_ioi['glacier_id'].iloc[network_id[:, 1]].to_numpy()
            dxyt_line = np.rec.fromarrays([xy_1[:, 0], xy_1[:, 1], t_1, g_1,
                                           xy_2[:, 0], xy_2[:, 1], t_2, g_2,
                                           dz_ioi],
                                          names=names, formats=formats)
        dxyt = pd.concat([dxyt, pd.DataFrame.from_records(dxyt_line)], axis=0)
    return dxyt

def get_casted_elevation_difference_rec(dh):
    names = ('X_1', 'Y_1', 'T_1', 'X_2', 'Y_2', 'T_2', 'dH_12')
    formats = [np.float64, np.float64, '<M8[D]',
               np.float64, np.float64, '<M8[D]', np.float64]
    dxyt = np.rec.array(np.zeros((0,7)),
                        names=names, formats=formats)
    ids = np.unique(dh['id']).astype(int)
    for ioi in ids:
        IN = dh['id'] == ioi # get id of interest
        dh_ioi = dh[IN]

        n = np.sum(IN).astype(int)
        network_id = get_network_indices(n).T

        sun = angles2unit(dh_ioi['azimuth'])
        xy_1 = np.stack((dh_ioi['casted_X'][network_id[:, 0]],
                         dh_ioi['casted_Y'][network_id[:, 0]])).T
        xy_2 = np.stack((dh_ioi['casted_X'][network_id[:, 1]],
                         dh_ioi['casted_Y'][network_id[:, 1]])).T
        xy_t = np.stack((dh_ioi['caster_X'][network_id[:, 0]],
                         dh_ioi['caster_Y'][network_id[:, 0]])).T

        dz_ioi = get_elevation_difference(sun[network_id[:, 0]],
                                          sun[network_id[:, 1]],
                                          xy_1, xy_2, xy_t)
        dxyt_line = np.rec.fromarrays([xy_1[:,0], xy_1[:,1],
               dh_ioi['timestamp'][network_id[:,0]], xy_2[:,0], xy_2[:,1],
               dh_ioi['timestamp'][network_id[:,1]],
               np.atleast_1d(np.squeeze(dz_ioi))], names=names, formats=formats)
        dxyt = stack_arrays((dxyt, dxyt_line),
                            asrecarray=True, usemask=False)
    return dxyt

def get_casted_elevation_difference_np(dh):
    dxyt = np.zeros((0,7))
    ids = np.unique(dh[:,-1]).astype(int)

    for ioi in ids:
        IN = dh[:,-1] == ioi
        dh_ioi = dh[IN,:]

        n = np.sum(IN).astype(int)
        network_id = get_network_indices(n).T

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
        dxyt_line = np.hstack((xy_1, dh_ioi[:,0][network_id[:,0]],
                               xy_2, dh_ioi[:,0][network_id[:,1]],
                               dz_ioi))
        dxyt = np.vstack((dxyt, dxyt_line))
    return dxyt

@loggg
def clean_dxyt(dxyt):
    """ since dhdt is glacier centric, connections between shadow casts that are
    siturated on different glaciers, are remove by this function

    Parameters
    ----------
    dxyt : {numpy.array, numpy.recordarray}, size=(m,n)
        data array with the following collumns:
            - X_1, Y_1 : map coordinate at instance t_1
            - T_1 : date stamp
            - X_2, Y_2 : map coordinate at instance t_2
            - T_2 : date stamp
            - dH_12 : estimated elevation change, between t_1 & t_2
        optional:
            - Z_1, Z_2 : elevation at instance/location _1 and _2
            - G_1, G_2 : glacier id at instance/location _1 and _2

    Returns
    -------
    dxyt : {numpy.array, numpy.recordarray}, size=(k,n)
    """
    if np.all([header in dxyt.columns for header in
               ('G_1', 'G_2')]):
        # only keep data points that are on the same glacier
        dxyt = dxyt[dxyt['G_1']==dxyt['G_2']]
    return dxyt

def get_relative_hypsometric_network(dxyt, Z, geoTransform):
    """ create design matrix from photohypsometric data collection

    Parameters
    ----------
    dxyt : {numpy.array, numpy.recordarray}, size=(m,n)
        data array with the following collumns:
            - X_1, Y_1 : map coordinate at instance t_1
            - T_1 : date stamp
            - X_2, Y_2 : map coordinate at instance t_2
            - T_2 : date stamp
            - dH_12 : estimated elevation change, between t_1 & t_2
        optional:
            - Z_1, Z_2 : elevation at instance/location _1 and _2
    Z : numpy.array, size=(m,n)
        array with elevation data
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of the array 'Z'

    Returns
    -------
    posts : numpy.array, size=m,2
        measurement array with map locations
    A : numpy.array, size=m,k
        design matrix with elevation differences
    """

    if type(dxyt) in (np.recarray,):
        i_1,j_1 = map2pix(geoTransform,
                          dxyt['X_1'].to_numpy(), dxyt['Y_1'].to_numpy())
        i_2,j_2 = map2pix(geoTransform,
                          dxyt['X_2'].to_numpy(), dxyt['Y_2'].to_numpy())
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

def get_hypsometric_elevation_change(dxyt, Z=None, geoTransform=None):
    """

    Parameters
    ----------
    dxyt : {numpy.array, numpy.recordarray}
        data array with the following collumns:
            - X_1, Y_1 : map coordinate at instance t_1
            - T_1 : date stamp
            - X_2, Y_2 : map coordinate at instance t_2
            - T_2 : date stamp
            - dH_12 : estimated elevation change, between t_1 & t_2
        optional:
            - Z_1, Z_2 : elevation at instance/location _1 and _2
            - G_1, G_2 : glacier identification
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
            optional:
            - g_1 & g_2 : glacier identification

    See Also
    --------
    get_casted_elevation_difference
    """
    if type(dxyt) in (np.recarray,):
        dhdt = get_hypsometric_elevation_change_rec(dxyt, Z=Z,
                                                    geoTransform=geoTransform)
    elif type(dxyt) in (np.ndarray,):
        dhdt = get_hypsometric_elevation_change_np(dxyt, Z=Z,
                                                   geoTransform=geoTransform)
    elif type(dxyt) in (pd.core.frame.DataFrame,):
        assert np.all([header in dxyt.columns for header in
                       ('X_1','Y_1','X_2','Y_2','dH_12')])
        dhdt = get_hypsometric_elevation_change_pd(dxyt, Z=Z,
                                                    geoTransform=geoTransform)
    return dhdt

def get_hypsometric_elevation_change_rec(dxyt, Z=None, geoTransform=None):
    i_1,j_1 = map2pix(geoTransform,
                      dxyt['X_1'].to_numpy(), dxyt['Y_1'].to_numpy())
    i_2,j_2 = map2pix(geoTransform,
                      dxyt['X_2'].to_numpy(), dxyt['Y_2'].to_numpy())

    Z_1 = ndimage.map_coordinates(Z, [i_1, j_1], order=1, mode='mirror')
    Z_2 = ndimage.map_coordinates(Z, [i_2, j_2], order=1, mode='mirror')

    # calculate mid point elevation
    Z_12 = ndimage.map_coordinates(Z, [(i_1+i_2)/2, (j_1+j_2)/2],
                                   order=1, mode='mirror')

    dZ = np.squeeze(Z_1) - np.squeeze(Z_2)
    dz_12 = dxyt['dH_12'] - dZ

    desc = np.dtype([('T_1', '<M8[D]'), ('T_2', '<M8[D]'), ('Z_12', np.float64),
                     ('dZ_12', np.float64)])
    dhdt = np.rec.fromarrays([dxyt['T_1'], dxyt['T_2'], Z_12, dz_12],
                             dtype=desc)
    return dhdt

def get_hypsometric_elevation_change_np(dxyt, Z=None, geoTransform=None):
    i_1,j_1 = map2pix(geoTransform, dxyt[:,0], dxyt[:,1])
    i_2,j_2 = map2pix(geoTransform, dxyt[:,3], dxyt[:,4])

    Z_1 = ndimage.map_coordinates(Z, [i_1, j_1], order=1, mode='mirror')
    Z_2 = ndimage.map_coordinates(Z, [i_2, j_2], order=1, mode='mirror')

    # calculate mid point elevation
    Z_12 = ndimage.map_coordinates(Z, [(i_1+i_2)/2, (j_1+j_2)/2],
                                   order=1, mode='mirror')

    dZ = np.squeeze(Z_1) - np.squeeze(Z_2)
    dz_12 = dxyt[:,-1] - dZ

    dhdt = np.stack((dxyt[:,2], dxyt[:,5], Z_12, dz_12))
    return dhdt

def get_hypsometric_elevation_change_pd(dxyt, Z=None, geoTransform=None):

    simple=True
    if 'Z_1' not in dxyt.columns:
        i_1,j_1 = map2pix(geoTransform,
                          dxyt['X_1'].to_numpy(), dxyt['Y_1'].to_numpy())
        Z_1 = ndimage.map_coordinates(Z, [i_1, j_1], order=1, mode='mirror')
    else:
        Z_1 = dxyt['Z_1'].to_numpy()
    if 'Z_2' not in dxyt.columns:
        i_2,j_2 = map2pix(geoTransform,
                          dxyt['X_2'].to_numpy(), dxyt['Y_2'].to_numpy())
        Z_2 = ndimage.map_coordinates(Z, [i_2, j_2], order=1, mode='mirror')
    else:
        Z_2 = dxyt['Z_2'].to_numpy()

    if simple:
        Z_12 = (Z_1+Z_2)/2
    else:
        # calculate mid point elevation
        Z_12 = ndimage.map_coordinates(Z, [(i_1 + i_2) / 2, (j_1 + j_2) / 2],
                                       order=1, mode='mirror')

    dZ = np.squeeze(Z_1) - np.squeeze(Z_2)
    dz_12 = dxyt['dH_12'] - dZ

    desc = [('T_1', '<M8[D]'), ('T_2', '<M8[D]'), ('Z_12', np.float64),
            ('dZ_12', np.float64)]
    arrs = [dxyt['T_1'], dxyt['T_2'], Z_12, dz_12]

    if 'G_1' in dxyt.columns:
        arrs.append(dxyt['G_1'].to_numpy()), desc.append(('G_1', np.int16))
    if 'G_2' in dxyt.columns:
        arrs.append(dxyt['G_2'].to_numpy()), desc.append(('G_2', np.int16))

    dhdt = np.rec.fromarrays(arrs, dtype=np.dtype(desc))
    dhdt = pd.DataFrame.from_records(dhdt)
    return dhdt

def get_mass_balance_per_elev(dhdt, spac=100.):
    L = (dhdt['Z_12'].to_numpy() // spac).astype(int)
    labels, indices, counts = np.unique(L,
                                        return_counts=True,
                                        return_inverse=True)
    dZ_split = np.split(dhdt.iloc[indices.argsort(),
                                   dhdt.columns.get_loc('dZ_12')],
                         counts.cumsum()[:-1])

    # using the fixed-date system of a mass-balance year,
    # see also "Mass-balance year" at pp.66 in [COG11]_
    yr_min = np.min(dhdt['T_1'].to_numpy())
    yr_mnd = yr_min.astype('datetime64[M]').astype(int) % 12
    yr_min = yr_min.astype('datetime64[Y]').astype(int) + 1970
    if yr_mnd < 9: yr_min -= 1

    yr_max = np.max(dhdt['T_2'].to_numpy())
    yr_mnd = yr_max.astype('datetime64[M]').astype(int) % 12
    yr_max = yr_max.astype('datetime64[Y]').astype(int) + 1970
    if yr_mnd >= 9: yr_max += 1

    T_step = pd.date_range(start=str(yr_min), end=str(yr_max+1),
                           freq='A-SEP').to_numpy()

    time_range = np.arange(yr_min+1,yr_max+1)
    belev = np.zeros((labels.size, time_range.size))
    for idx,arr in enumerate(dZ_split):
        df_sub = dhdt.iloc[arr.index,:]

        T_1,T_2 = df_sub['T_1'].values.astype('datetime64[D]'),\
                  df_sub['T_2'].values.astype('datetime64[D]')

        A = get_temporal_incidence_matrix(T_1, T_2, T_step, open_network=True)
        W = get_temporal_weighting(T_1, T_2, covariance=True)

        x_hat = np.linalg.lstsq(np.dot(W,A),
                                np.dot(df_sub['dZ_12'].to_numpy(), W),
                                rcond=None)[0]
        belev[idx,:] = x_hat

    elev = labels * spac
    belev = pd.DataFrame(belev,
                         columns=list(map(str, time_range.tolist())),
                         index=list(map(str, elev.astype(int).tolist())))
    return belev