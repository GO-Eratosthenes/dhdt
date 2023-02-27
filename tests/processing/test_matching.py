import numpy as np

from dhdt.processing.matching_tools import get_integer_peak_location
from dhdt.processing.matching_tools_frequency_filters import normalize_power_spectrum

# artificial creation functions

# testing functions
def test_phase_plane_localization(Q, di, dj, tolerance=1.):
    """ test if an estimate is within bounds

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-power spectrum
    di,dj : {float, integer}, unit=pixels
        displacement estimation
    tolerance : float
        absolute tolerance that is allowed
    """
    C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    di_hat,dj_hat,_,_ = get_integer_peak_location(C)

    assert np.isclose(np.round(di), di_hat, tolerance)
    assert np.isclose(np.round(dj), dj_hat, tolerance)
    return

def test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=.1):
    assert np.isclose(di, di_hat, tolerance)
    assert np.isclose(dj, dj_hat, tolerance)
    return

def test_phase_direction(θ, di, dj, tolerance=5):
    tolerance = np.deg2radrad(tolerance)
    θ = np.deg2rad(θ)
    θ_tilde = np.arctan2(di, dj)

    # convert to complex domain, so angular difference can be done
    a,b = 1j*np.sin(θ), 1j*np.sin(θ_tilde)
    a += np.cos(θ)
    b += np.cos(θ_tilde)
    assert np.isclose(a,b, tolerance)
    return

def test_normalize_power_spectrum(Q, tolerance=.001):
    # trigonometric version
    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    np.putmask(Qn, Q==0, 0)

    Qd = normalize_power_spectrum(Q)
    assert np.all(np.isclose(Qd,Qn, tolerance))
    return