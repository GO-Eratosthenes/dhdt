import os

import numpy as np
import pandas
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
from tqdm import tqdm

from ..generic.data_tools import (four_pl_curve, hough_transf, logit,
                                  logit_weighting)
from ..generic.debugging import loggg
from ..generic.handler_im import simple_nearest_neighbor
from ..generic.mapping_tools import get_max_pixel_spacing, map2pix
from ..generic.unit_check import are_two_arrays_equal
from .matching_tools import get_data_and_mask


def transform_4_logit(X, Y, method='exact', **kwargs):
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
    .. [As1] Asuero, et al. "Fitting straight lines with replicated
             observations by linear regression. IV. Transforming data."
             Critical reviews in analytical chemistry, vol.41(1) pp. 36-69,
             2011.
    """

    if method in ('exact', ):
        a, d = np.nanmin(Y), np.nanmax(Y)
    else:
        qsep = 0.05 if kwargs.get('quantile_dist') is None else \
            kwargs.get('quantile_dist')
        a, d = np.quantile(Y, qsep), np.quantile(Y, 1 - qsep)
    X_tilde, Y_tilde = np.log10(X), logit(np.divide(Y - d, a - d))
    return X_tilde, Y_tilde, a, d


def refine_cast_location(Z,
                         M,
                         geoTransform,
                         x_casted,
                         y_casted,
                         azimuth,
                         x_caster,
                         y_caster,
                         extent=120,
                         method='spline',
                         **kwargs):
    """ search along the episolar line to find the cast ridge

    Parameters
    ----------
    Z : {numpy.array, numpy.masked.array}, size=(m,n), ndim=2
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
    if np.any((np.isnan(x_casted), np.isnan(y_casted), np.isnan(x_caster),
               np.isnan(y_caster), np.isnan(azimuth))):
        return x_casted, y_casted

    Z, M = get_data_and_mask(Z, M)
    are_two_arrays_equal(Z, M)

    cast_length = np.hypot(x_casted - x_caster, y_casted - y_caster)
    steps = get_max_pixel_spacing(geoTransform)
    spread = np.minimum(extent, cast_length - steps)
    rng = np.arange(-spread, +spread + steps, steps)
    samples, az_rad = rng.size, np.deg2rad(azimuth + 180)

    # get coordinates of episolar line
    line_x = np.sin(az_rad) * rng + x_casted
    line_y = np.cos(az_rad) * rng + y_casted

    line_i, line_j = map2pix(geoTransform, line_x, line_y)
    line_Z = ndimage.map_coordinates(Z, [line_i, line_j],
                                     order=1,
                                     mode='mirror')
    line_M = simple_nearest_neighbor(M.astype(int), line_i, line_j)
    local_x = np.arange(samples)

    # search for double shadows
    prio = np.where(line_M[:samples // 2 - 1] == 0)[0]  # prior
    post = np.where(line_M[samples // 2 + 2:] == 1)[0]  # post

    if prio.size > 0:  # bright pixels are present...
        prio_idx = np.max(prio) + 1
        local_x = local_x[prio_idx:]
        rng = rng[prio_idx:]
        line_Z = line_Z[prio_idx:]
    if post.size > 0:
        post_idx = np.min(post) - 1
        local_x = local_x[:post_idx]
        rng = rng[:post_idx]
        line_Z = line_Z[:post_idx]

    # estimate infliction point
    if line_Z.shape[0] < 4:  # shadow cast line is too small
        return x_casted, y_casted

    if method in ('curve fitting', 'curve_fit'):
        try:
            a_hat, a_cov = curve_fit(
                four_pl_curve,
                local_x,
                line_Z,
                p0=[np.min(line_Z), 1., .1,
                    np.max(line_Z)])
            y_lsq = four_pl_curve(local_x, *a_hat)
            infl_point = np.interp(a_hat[2], local_x, rng)
            point_cov = a_cov[2, 2] * steps
        except RuntimeError:
            infl_point, point_cov = 0, 0
    elif method in ('spline', 'cubic_spline'):
        # qnt = 0.5 if kwargs.get('quantile') is None else \
        #     kwargs.get('quantile')
        # spli = UnivariateSpline(rng, line_Z-np.quantile(line_Z,qnt), k=3)
        spli = CubicSpline(rng, line_Z - threshold_otsu(line_Z))
        crossings = spli.roots()
        # only select upwards roots
        UP = np.sign(spli(crossings, 1)) == 1
        if np.any(UP):
            crossings = crossings[UP]
        # select closest to
        infl_point = crossings[np.argmin(np.abs(crossings))]
    else:  # estimate via Hough transform
        x_tilde, y_tilde, a, d = transform_4_logit(local_x,
                                                   line_Z,
                                                   method='exact',
                                                   quantile_dist=0.05)
        w = logit_weighting(np.divide(line_Z - d, a - d))
        c, b, point_cov = hough_transf(x_tilde,
                                       y_tilde,
                                       w=w,
                                       bnds=[-1, +1.5, -1.5, +1.5])
        y_hgh = four_pl_curve(local_x, a, b[0], -c[0], d)
        infl_point = np.interp(-c[0], local_x, rng)

    if np.isnan(az_rad) or np.isnan(infl_point):
        return x_casted, y_casted
    else:
        x_refine = x_casted + np.sin(az_rad) * infl_point
        y_refine = y_casted + np.cos(az_rad) * infl_point
        return x_refine, y_refine


@loggg
def update_casted_location(dh, S, M, geoTransform, extent=120.):
    """ update the cast location to sub-pixel level, by looking at the 
    intensity profile along the sun trace.

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
    """  # noqa: W605
    assert type(dh) in (pandas.core.frame.DataFrame,), \
        'please provide a pandas dataframe'
    assert np.all([
        header in dh.columns
        for header in ('caster_X', 'caster_Y', 'casted_X', 'casted_Y')
    ])

    if 'casted_X_refine' not in dh.columns:
        dh['casted_X_refine'] = None
    if 'casted_Y_refine' not in dh.columns:
        dh['casted_Y_refine'] = None

    # create pointers for iloc
    idx_caster_x = dh.columns.get_loc("caster_X")
    idx_caster_y = dh.columns.get_loc("caster_Y")
    idx_casted_x = dh.columns.get_loc("casted_X")
    idx_casted_y = dh.columns.get_loc("casted_Y")
    idx_refine_x = dh.columns.get_loc("casted_X_refine")
    idx_refine_y = dh.columns.get_loc("casted_Y_refine")

    idx_azimuth = dh.columns.get_loc("azimuth")
    for cnt in tqdm(range(dh.shape[0])):
        x, y = dh.iloc[cnt, idx_casted_x], dh.iloc[cnt, idx_casted_y]
        x_t, y_t = dh.iloc[cnt, idx_caster_x], dh.iloc[cnt, idx_caster_y]
        az = dh.iloc[cnt, idx_azimuth]
        if np.any((np.isnan(x), np.isnan(y), np.isnan(x_t), np.isnan(y_t),
                   np.isnan(az))):
            continue
        x_new, y_new = refine_cast_location(S,
                                            M,
                                            geoTransform,
                                            x,
                                            y,
                                            az,
                                            x_t,
                                            y_t,
                                            extent=extent)
        dh.iloc[cnt, idx_refine_x], dh.iloc[cnt, idx_refine_y] = x_new, y_new
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
            ├ casted_Y_refine <- refined location of the end of the shadow trace
            ├ azimuth         <- argument of the illumination direction
            └ zenith          <- off-normal angle of the sun without atmosphere

    See Also
    --------
    update_casted_location
    """  # noqa: W605,E501
    col_order = ('caster_X', 'caster_Y', 'casted_X', 'casted_Y',
                 'casted_X_refine', 'casted_Y_refine', 'azimuth', 'zenith')
    col_frmt = ('{:+8.2f}', '{:+8.2f}', '{:+8.2f}', '{:+8.2f}', '{:+8.2f}',
                '{:+8.2f}', '{:+3.4f}', '{:+3.4f}')
    assert type(dh) in (pandas.core.frame.DataFrame,), \
        ('please provide a pandas dataframe')
    assert np.all([header in dh.columns for header in col_order])

    # write to file
    f = open(os.path.join(file_dir, file_name), 'w')

    # if timestamp is give, add this to the
    timestamps = dh['timestamp'].unique()
    if len(timestamps) == 1:
        print('# time: ' + np.datetime_as_string(timestamps[0], unit='D'),
              file=f)
    else:
        col_order = ('timestamp', ) + col_order
        col_frmt = ('place_holder', ) + col_order
    col_order += ('zenith_refrac', )
    col_frmt = col_frmt + ('{:+3.4f}', )
    dh_sel = dh[dh.columns.intersection(list(col_order))]

    # add header
    col_idx = dh_sel.columns.get_indexer(list(col_order))
    IN = col_idx != -1
    print('# ' + ' '.join(tuple(np.array(col_order)[IN])), file=f)
    col_idx = col_idx[IN]

    # write rows of dataframe into text file
    for k in range(dh_sel.shape[0]):
        line = ''
        for idx, val in enumerate(col_idx):
            if col_order[val] == 'timestamp':
                line += np.datetime_as_string(dh_sel.iloc[k, val], unit='D')
            else:
                elem = dh_sel.iloc[k, val]
                if elem is None:
                    # refinement or refraction seem to have an error
                    # find
                    ori_name = '_'.join(dh_sel.columns[val].split('_')[0:2])
                else:
                    line += col_frmt[idx].format(elem)
            line += ' '
        print(line[:-1], file=f)  # remove last spacer
    f.close()
    return


# def filter_unresolved_refinements(dh):
