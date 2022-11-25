import os

import numpy as np
import pandas

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from skimage.filters import threshold_otsu

from ..generic.debugging import loggg
from ..generic.mapping_tools import get_max_pixel_spacing, map2pix
from ..generic.handler_im import bilinear_interpolation, simple_nearest_neighbor
from ..generic.data_tools import \
    four_pl_curve, logit, hough_transf, logit_weighting
from ..generic.unit_check import are_two_arrays_equal
from .matching_tools import get_data_and_mask

def transform_4_logit(X,Y,method='exact', **kwargs):
    """
    Parameters
    ----------
    X,Y : numpy.array, size=(m,)
        coordinates, where Y has a logisitc curve
    method : string, {'exact', 'quantile'}
        using either exact minimum and maximum, or also quantile entities

    Returns
    -------
    X_tilde,Y_tilde : numpy.array, size=(m,)
        transformed data, see also [1]

    References
    ----------
    .. [1] Asuero, et al. "Fitting straight lines with replicated observations
       by linear regression. IV. Transforming data." Critical reviews in
       analytical chemistry, vol.41(1) pp. 36-69, 2011.
    """

    if method in ('exact',):
        a, d = np.nanmin(Y), np.nanmax(Y)
    else:
        qsep = 0.05 if kwargs.get('quantile_dist') == None else \
            kwargs.get('quantile_dist')
        a, d = np.quantile(Y, qsep), np.quantile(Y, 1-qsep)
    X_tilde, Y_tilde = np.log10(X), logit(np.divide(Y-d, a-d))
    return X_tilde, Y_tilde, a, d

def refine_cast_location(I, M, geoTransform, x_casted, y_casted, azimuth,
                         x_caster, y_caster, extent=120, method='spline',
                         **kwargs):
    """ search along the episolar line to find the cast ridge

    Parameters
    ----------
    I : {numpy.array, numpy.masked.array}, size=(m,n), ndim=2
        grid with intensity values
    M : numpy.array, size=(m,n), ndim=2, dtype={bool,int}
        grid with mask
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    x_casted, y_casted :  float, unit=meter
        rough map coordinate of the cast shadow edge
    azimuth : float, unit=degrees
        illumination direction of the sun
    x_caster, y_caster :  float, unit=meter
        map coordinate of the start of the shadow line
    extent : {float,int}, unit=meter
        search range along the episolar line
    method : {'curve_fit','hough'}
        estimation of the curve fitting, either via non-linear least squares
        fitting, or via a more robust Hough transform.

    Returns
    -------
    x_refine, y_refine : float, unit=meter
        refined map coordinate of the cast shadow edge

    """
    I,M = get_data_and_mask(I,M)
    are_two_arrays_equal(I, M)

    cast_length = np.hypot(x_casted - x_caster, y_casted - y_caster)
    steps = get_max_pixel_spacing(geoTransform)
    spread = np.minimum(extent, cast_length-steps)
    rng = np.arange(-spread, +spread + steps, steps)
    samples, az_rad = rng.size, np.deg2rad(azimuth+180)

    # get coordinates of episolar line
    line_x = np.sin(az_rad)*rng + x_casted
    line_y = np.cos(az_rad)*rng + y_casted

    line_i, line_j = map2pix(geoTransform, line_x, line_y)
    line_I = bilinear_interpolation(I, line_i, line_j)
    line_M = simple_nearest_neighbor(M.astype(int), line_i, line_j)
    local_x = np.arange(samples)

    # search for double shadows
    prio = np.where(line_M[:samples//2-1]==0)[0] # prior
    post = np.where(line_M[samples//2+2:]==1)[0] # post

    if prio.size>0: # bright pixels are present...
        prio_idx = np.max(prio)+1
        local_x, rng, line_I = local_x[prio_idx:], rng[prio_idx:], \
                               line_I[prio_idx:]
    if post.size>0:
        post_idx = np.min(post)-1
        local_x, rng, line_I = local_x[:post_idx], rng[:post_idx], \
                               line_I[:post_idx]

    # estimate infliction point
    if line_I.shape[0]<4: # shadow cast line is too small
        return x_casted, y_casted

    if method in ('curve fitting','curve_fit'):
        try:
            a_hat, a_cov = curve_fit(four_pl_curve, local_x, line_I,
                                     p0=[np.min(line_I), 1., .1, np.max(line_I)])
            y_lsq = four_pl_curve(local_x, *a_hat)
            infl_point = np.interp(a_hat[2], local_x, rng)
            point_cov = a_cov[2,2]*steps
        except:
            infl_point, point_cov = 0,0
    elif method in ('spline', 'cubic_spline'):
#        qnt = 0.5 if kwargs.get('quantile') is None else kwargs.get('quantile')
#        spli = UnivariateSpline(rng, line_I-np.quantile(line_I,qnt), k=3)
        spli = CubicSpline(rng, line_I-threshold_otsu(line_I))
        crossings = spli.roots()
        # only select upwards roots
        UP = np.sign(spli(crossings, 1))==1
        if np.any(UP):
            crossings = crossings[UP]
        # select closest to
        infl_point = crossings[np.argmin(np.abs(crossings))]
    else: # estimate via Hough transform
        x_tilde, y_tilde, a, d = transform_4_logit(local_x, line_I,
           method='exact', quantile_dist=0.05)
        w = logit_weighting(np.divide(line_I - d, a - d))
        c, b, point_cov = hough_transf(x_tilde, y_tilde, w=w,
                                       bnds=[-1, +1.5, -1.5, +1.5])
        y_hgh = four_pl_curve(local_x, a, b[0], -c[0], d)
        infl_point = np.interp(-c[0], local_x, rng)

    x_refine = x_casted + np.sin(az_rad)*infl_point
    y_refine = y_casted + np.cos(az_rad)*infl_point
    return x_refine, y_refine

@loggg
def update_casted_location(dh, S, M, geoTransform, extent=120.):
    """ update the cast location to sub-pixel level, by looking at the intensity
    profile along the sun trace.

    Parameters
    ----------
    dh : pandas.DataFrame
        array with photohypsometric information
    S : numpy.array, size=(m,n), dtype={integer,float}
        grid with intensities, mostly related to illumination/shadowing
    M : numpy.array, size=(m,n), dtype={integer,boolean}
        mask of the rough shadow polygon, where True denotes a shade
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients
    extent : {float, integer}, unit=meters
        amount of data used along the trace line.

    Returns
    -------
    dh : pandas.DataFrame
        array with photohypsometric information

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
    assert type(dh) in (pandas.core.frame.DataFrame, ), \
        ('please provide a pandas dataframe')
    assert np.all([header in dh.columns for header in
                  ('caster_X', 'caster_Y', 'casted_X', 'casted_Y')])

    if not 'casted_X_refine' in dh.columns: dh['casted_X_refine'] = None
    if not 'casted_Y_refine' in dh.columns: dh['casted_Y_refine'] = None

    for cnt in range(dh.shape[0]):
        x, y = dh.iloc[cnt]['casted_X'], dh.iloc[cnt]['casted_Y']
        x_t, y_t = dh.iloc[cnt]['caster_X'], dh.iloc[cnt]['caster_Y']
        az = dh.iloc[cnt]['azimuth']

        x_new, y_new = refine_cast_location(S, M, geoTransform, x, y, az,
                                            x_t, y_t, extent=extent)
        dh.loc[cnt, 'casted_X_refine'] = x_new
        dh.loc[cnt, 'casted_Y_refine'] = y_new
    return dh

def update_list_with_refined_casted(dh, file_dir, file_name='conn.txt'):
    """ write

    Parameters
    ----------
    dh : pandas.DataFrame
        array with photohypsometric information
    file_dir : string
        directory where the file should be situated
    file_name : string, default='conn.txt'
        name of the file to export the data columns to

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

    The text file has at least the following columns:

        .. code-block:: text

            * conn.txt
            ├ caster_X        <- map location of the start of the shadow trace
            ├ caster_Y
            ├ casted_X
            ├ casted_Y        <- map location of the end of the shadow trace
            ├ casted_X_refine
            ├ casted_Y_rifine <- refined location of the end of the shadow trace
            ├ azimuth         <- argument of the illumination direction
            └ zenith          <- off-normal angle of the sun without atmosphere

    See Also
    --------
    update_casted_location
    """
    col_order = ('caster_X', 'caster_Y', 'casted_X', 'casted_Y',
                 'casted_X_refine', 'casted_Y_refine',
                 'azimuth', 'zenith')
    col_frmt = ('{:+8.2f}', '{:+8.2f}', '{:+8.2f}', '{:+8.2f}',
                '{:+8.2f}', '{:+8.2f}',
                '{:+3.4f}', '{:+3.4f}')
    assert type(dh) in (pandas.core.frame.DataFrame, ), \
        ('please provide a pandas dataframe')
    assert np.all([header in dh.columns for header in col_order])

    # write to file
    f = open(os.path.join(file_dir, file_name), 'w')

    # if timestamp is give, add this to the
    timestamps = dh['timestamp'].unique()
    if len(timestamps)==1:
        print('# time: '+np.datetime_as_string(timestamps[0], unit='D'),
              file=f)
    else:
        col_order = ('timestamp',) + col_order
        col_frmt = ('place_holder',) + col_order
    col_order += ('zenith_refrac',)
    col_frmt = col_frmt + ('{:+3.4f}',)
    dh_sel = dh[dh.columns.intersection(list(col_order))]

    # add header
    col_idx = dh_sel.columns.get_indexer(list(col_order))
    IN = col_idx!=-1
    print('# '+' '.join(tuple(np.array(col_order)[IN])), file=f)
    col_idx = col_idx[IN]

    # write rows of dataframe into text file
    for k in range(dh_sel.shape[0]):
        line = ''
        for idx,val in enumerate(col_idx):
            if col_order[val] == 'timestamp':
                line += np.datetime_as_string(dh_sel.iloc[k,val], unit='D')
            else:
                line += col_frmt[idx].format(dh_sel.iloc[k,val])
            line += ' '
        print(line[:-1], file=f) # remove last spacer
    f.close()
    return
