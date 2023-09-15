import numpy as np

from scipy import ndimage

from dhdt.generic.handler_im import get_grad_filters
from dhdt.postprocessing.displacement_filters import nan_resistant_filter


def test_nan_resistant_filter(m=40, n=30, nan_frac=0.1, tsize=3):
    slopes = np.random.uniform(low=-10., high=+10., size=(2, ))
    grd_1, grd_2 = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
    Z = grd_1 * slopes[0] + grd_2 * slopes[1]

    pad = tsize // 2
    fx, fy = get_grad_filters(ftype='kroon', tsize=tsize, order=1)
    Z_dx = ndimage.convolve(Z, fx, mode='reflect')
    Z_dy = ndimage.convolve(Z, fy, mode='reflect')
    Z_dx = Z_dx[pad:-pad, pad:-pad]
    Z_dy = Z_dy[pad:-pad, pad:-pad]

    # corrupt the data
    mn = m * n
    nan_i, nan_j = np.unravel_index(np.random.choice(mn, int(mn * nan_frac)),
                                    (m, n),
                                    order='C')
    Z[nan_i, nan_j] = np.nan

    Z_dx_nan = nan_resistant_filter(Z, fx)
    Z_dy_nan = nan_resistant_filter(Z, fy)
    Z_dx_nan = Z_dx_nan[pad:-pad, pad:-pad]
    Z_dy_nan = Z_dy_nan[pad:-pad, pad:-pad]

    assert np.isclose(0, np.nanmedian(np.abs(Z_dx_nan - Z_dx)))
    assert np.isclose(0, np.nanmedian(np.abs(Z_dy_nan - Z_dy)))
