import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

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
        return x_casted, y_casted, 0

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
        qnt = 0.5 if kwargs.get('quantile') is None else kwargs.get('quantile')
#        spli = UnivariateSpline(rng, line_I-np.quantile(line_I,qnt), k=3)
        spli = CubicSpline(rng, line_I-np.quantile(line_I,qnt))
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
    return x_refine, y_refine, point_cov

