import numpy as np
from scipy import ndimage

from dhdt.generic.filtering_statistical import make_2D_Gaussian


def ssim(I1, I2):
    """ estimate the structural similarity between two images

    References
    ----------
    .. [Wa04] Wang et al., "Image quality assessment: from error visibility to
              structural similarity", IEEE transactions on image processing,
              vol.13(4) pp.600-612, 2004.
    """

    ɛ_1 = (0.01 * 255)**2
    ɛ_2 = (0.03 * 255)**2

    window = make_2D_Gaussian([11, 11], fwhm=3)

    μ_1, μ_2 = ndimage.convolve(I1, window), ndimage.convolve(I2, window)
    μ_1, μ_2 = μ_1**2, μ_2**2
    μ_12 = μ_1 * μ_2

    σ_1, σ_2 = ndimage.convolve(I1**2, window), ndimage.convolve(I2**2, window)
    σ_12 = ndimage.convolve(I1 * I2, window) - μ_12
    σ_1 -= μ_1
    σ_2 -= μ_2

    struc_sim = np.divide((2 * μ_12 + ɛ_1) * (2 * σ_12 + ɛ_2),
                          (μ_1 + μ_2 + ɛ_1) * (σ_1 + σ_2 + ɛ_2))
    return struc_sim


def moments(x, y, k, stride):

    def _mom_int(Z, k, stride):
        mu = Z[:-k:stride, :-k:stride] - \
             Z[:-k:stride, +k::stride] - \
             Z[+k::stride, :-k:stride] + \
             Z[+k::stride, +k::stride]
        return mu

    def _integral_image(Z):
        m, n = Z.shape
        C = np.zeros((m + 1, n + 1))
        C[1:, 1:] = np.cumsum(np.cumsum(Z, 0), 1)
        return C

    k_norm = k**2

    x_pad = np.pad(x, int((k - stride) / 2), mode='reflect')
    y_pad = np.pad(y, int((k - stride) / 2), mode='reflect')

    int_1_x = _integral_image(x_pad)
    int_1_y = _integral_image(y_pad)
    int_2_x = _integral_image(x_pad * x_pad)
    int_2_y = _integral_image(y_pad * y_pad)

    int_xy = _integral_image(x_pad * y_pad)

    μ_x = _mom_int(int_1_x, k, stride) / k_norm
    μ_y = _mom_int(int_1_y, k, stride) / k_norm

    σ_x = _mom_int(int_2_x, k, stride) / k_norm
    σ_y = _mom_int(int_2_y, k, stride) / k_norm
    σ_xy = _mom_int(int_xy, k, stride) / k_norm

    σ_x -= μ_x**2
    σ_y -= μ_y**2
    σ_xy -= μ_x * μ_y

    OUT_x, OUT_y = σ_x < 0, σ_y < 0
    np.putmask(σ_x, OUT_x, 0)
    np.putmask(σ_y, OUT_y, 0)
    np.putmask(σ_xy, np.logical_or(OUT_x, OUT_y), 0)
    return μ_x, μ_y, σ_x, σ_y, σ_xy


def vif(I1, I2, k=11):
    """ Visual Information Fidelity

    References
    ----------
    .. [Sh06] Sheikh et al., "Image information and visual quality", IEEE
              transactions on image processing,
              vol.13(4) pp.600-612, 2004.
    """
    σ_nsq = 0.1
    _, _, σ_1, σ_2, σ_12 = moments(I1, I2, k)

    g = np.divide(σ_12, σ_1, where=σ_1 != 0)
    g = np.maximum(g, 0)
    sv_sq = σ_2 - g * σ_12

    vi_fidel = np.divide(
        np.log(1 + g**2 * σ_1 / (sv_sq + σ_nsq)) + 1e-4,
        np.log(1 + σ_1 / σ_nsq) + 1e-4)
    return vi_fidel


def psnr(I1, I2):
    mse = np.mean((I1 - I2)**2)
    mse_isq = np.sqrt(mse)
    peak_snr = 20 * np.log10(np.divide(255.0, mse_isq, where=mse_isq != 0))
    return peak_snr
