import os

import numpy as np
from osgeo import osr
from scipy.ndimage import label, binary_dilation, binary_erosion

from dhdt.generic.data_tools import gompertz_curve
from dhdt.generic.mapping_io import make_geo_im
from dhdt.generic.mapping_tools import pix2map, get_max_pixel_spacing, \
    pix_centers
from dhdt.generic.terrain_tools import terrain_slope
from dhdt.generic.unit_check import zenit_angle_check
from dhdt.generic.unit_conversion import deg2arg
from dhdt.postprocessing.solar_tools import make_shadowing
from dhdt.postprocessing.terrain_tools import d8_catchment
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.preprocessing.shadow_filters import fade_shadow_cast
from dhdt.preprocessing.shadow_geometry import shadow_image_to_list
from dhdt.presentation.velocity_tools import make_seeds
from dhdt.processing.matching_tools import remove_posts_outside_image
from dhdt.postprocessing.solar_tools import make_shadowing, make_shading
from dhdt.testing.mapping_tools import \
    create_local_crs, create_artificial_geoTransform


def create_artificial_terrain(m, n, step_size=.01, multi_res=(2, 4)):
    """ create artificail terrain, based upon Perlin noise, with a flavour of
    multi-resolution within

    Parameters
    ----------
    m,n : integer, {x ∈ ℕ | x ≥ 0}
        dimension of the elevation model to be created
    step_size : float, default=0.01
        parameter to be used to scale the frequency of the topography

    Returns
    -------
    Z : numpy.ndarray, size=(m,n), ndim=2
        data array with elevation values
    geoTransform : tuple, size=(8,1)
        affine transformation coefficients, and dimensions of the array.

    See Also
    --------
    dhdt.testing.mapping_tools.create_local_crs
    """
    if m is None: m = 1E3
    if n is None: n = 1E3

    def _linreg(a, b, x):
        return a + x * (b - a)

    def _fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def _gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    def _perlin(x, y):
        # permutation table
        rng = np.random.default_rng()
        p = np.arange(256, dtype=int)
        rng.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        x_i, y_i = x.astype(int), y.astype(int)
        # internal coordinates
        x_f, y_f = x - x_i, y - y_i
        # fade factors
        u, v = _fade(x_f), _fade(y_f)
        # noise components
        n00 = _gradient(p[p[x_i] + y_i], x_f, y_f)
        n01 = _gradient(p[p[x_i] + y_i + 1], x_f, y_f - 1)
        n11 = _gradient(p[p[x_i + 1] + y_i + 1], x_f - 1, y_f - 1)
        n10 = _gradient(p[p[x_i + 1] + y_i], x_f - 1, y_f)
        # combine noises
        x1, x2 = _linreg(n00, n10, u), _linreg(n01, n11, u)
        return _linreg(x1, x2, v)

    lin_m = np.arange(0, step_size * m, step_size)
    lin_n = np.arange(0, step_size * n, step_size)
    x, y = np.meshgrid(lin_n, lin_m)
    Z = _perlin(x, y)
    for f in multi_res:
        Z += mat_to_gray(Z) * (1 / f) * _perlin(f * x, f * y)
    Z += 1
    Z *= 1E3
    geoTransform = create_artificial_geoTransform(Z, spac=10.)
    return Z, geoTransform


def create_artificial_glacier_mask(Z, geoTransform, seeds=42, labeling=True):
    """ create from seeds and an elevation model, some regions that might
    represent a glacier.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        array with elevation
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    seeds : int
        amount of seeds to use to initiate the glacier growth
    labeling : bool
        label the mask with unique numbers

    Returns
    -------
    R : numpy.ndarray, size=(m,n), dtype={boolean,int)
        labeled array or mask
    """
    if Z is None: Z, geoTransform = create_artificial_terrain(None, None)
    i, j = make_seeds(np.ones_like(Z, dtype=bool), n=seeds)
    i, j, _ = remove_posts_outside_image(Z, i, j)
    x, y = pix2map(geoTransform, i, j)
    V_x, V_y = terrain_slope(Z, get_max_pixel_spacing(geoTransform))
    R = d8_catchment(V_x, V_y, geoTransform, x, y)
    if labeling:
        R = label(R)[0]
    return R


def create_artifical_sun_angles(n,
                                az_min=160.,
                                zn_min=35.,
                                az_max=200.,
                                zn_max=85.):
    zn_min, zn_max = zenit_angle_check(zn_min), zenit_angle_check(zn_max)
    zn = np.random.uniform(low=zn_min, high=zn_max, size=n)
    az = deg2arg(np.random.uniform(low=az_min, high=az_max, size=n))
    if n == 1: return az[0], zn[0]
    return az, zn


def create_shadow_caster_casted(Z,
                                geoTransform,
                                az,
                                zn,
                                out_path,
                                out_name="conn-test.txt",
                                incl_image=False,
                                incl_wght=False,
                                **kwargs):
    """
    Create artificial shadowing, and a connection file, for testing purposes

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n)
        array with elevation
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    az, zn : float, unit=degrees
        sun angles
    out_path : string
        location where to write the connectivity file
    out_name : string, default="conn-test.txt"
        name of the file
    """
    spac = get_max_pixel_spacing(geoTransform)
    Shw = make_shadowing(Z, az, zn, spac=spac)

    if np.logical_or(incl_image, incl_wght): crs = create_local_crs()

    if incl_image:
        im_name = out_name.split('.')[0] + ".tif"
        make_geo_im(Shw, geoTransform, crs.ExportToWkt(),
                    os.path.join(out_path, im_name))

    if incl_wght:
        t_size = 13 if kwargs.get('t_size') == None else kwargs.get('t_size')
        wg_name = out_name.split('.')[0] + "wgt.tif"
        # create weighting function
        F = fade_shadow_cast(Shw, az, t_size=t_size)
        W = np.abs(F - np.round(F)) * 2
        make_geo_im(W, geoTransform, crs.ExportToWkt(),
                    os.path.join(out_path, wg_name))

    shadow_image_to_list(Shw,
                         geoTransform,
                         out_path,
                         timestamp='0001-01-01',
                         Zn=zn,
                         Az=az,
                         out_name=out_name,
                         **kwargs)
    return


def create_artificial_glacier_change(Z, R):
    """ glacier change anomalies have a strong relation with elevation. This
    is simulated by this function, where elevations are adjusted, though this is
    only done for regions that are specified by the glacier mask.

    Parameters
    ----------
    Z : numpy.ndarray, size=(m,n), dtype=float
        grid with elevations
    R : numpy.ndarray, size=(m,n), dtype={boolean, integer}
        grid indicating the glacier regions

    Returns
    -------
    Z_new : numpy.ndarray, size=(m,n), dtype=float
        adjusted elevation grid
    a,b : floats
        regression parameters, used for the hypsometric elevation adjustment
    """
    if isinstance(R.flat[0], np.integer):  # if labelled array, convert to mask
        R = R != 0
    Z_glac = Z[R]

    a = np.random.uniform(low=10, high=30) / 5E2
    b = np.random.uniform(low=np.quantile(Z_glac, 0.4),
                          high=np.quantile(Z_glac, 0.6))

    Z_dh = np.zeros_like(Z)
    Z_dh[R] = a * (Z_glac - b)

    # make sure the edges are not too strong
    R_edt = distance_transform_edt(R)
    R_brd = gompertz_curve(R_edt, 10., .5)
    Z_dh *= R_brd

    Z_new = Z_dh + Z
    return Z_new, a, b


def create_artificial_morphsnake_data(m, n, delta=10):
    """ create artificial data for testing the shadow finding function

    Parameters
    ----------
    m,n : integer
        size of the constructed arrays
    delta : integer
        maximum amount of dilation and erosion

    Returns
    -------
    Shw : numpy.ndarray, size=(m,n), dtype=bool
        shadowing array
    I : numpy.ndarray, size=(m,n,2), dtype=float
        multi-spectral dummy data, based on shadowing and shading
    M : numpy.ndarray, size=(m,n), dtype=bool
        initial boolean array, roughly specifying where the shadow is located
    """
    Z, geoTransform = create_artificial_terrain(m, n)
    az, zn = create_artifical_sun_angles(1)

    Shw = make_shadowing(Z, az, zn, geoTransform)
    Shd = make_shading(Z, az, zn, geoTransform)

    # create initial boolean array, that should not align with this shadowing
    X, Y = pix_centers(geoTransform)
    R = np.round(delta * np.sin(X / m) * np.sin(Y / n)).astype(int)
    M = np.zeros((m, n), dtype=bool)
    # crude calculation
    for d in np.arange(-delta, delta + 1):
        if np.sign(d) == 1:
            dummy = binary_dilation(Shw, np.ones((d, d)))
        elif np.sign(d) == -1:
            dummy = binary_erosion(Shw, np.ones((np.abs(d), np.abs(d))))
        else:  # when zero
            dummy = Shw
        IN = R == d
        M[IN] = dummy[IN]

    # create multispectral dummy
    alpha = 0.3

    Shd = mat_to_gray(Shd)  # to a range of 0...1
    I = np.dstack((1 - (1 - alpha) * Shw + alpha * Shd,
                   (1 - alpha) * Shd + 1 - alpha * Shw))

    return Shw, I, M
