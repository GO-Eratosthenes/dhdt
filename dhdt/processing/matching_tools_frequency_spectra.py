import numpy as np

from .matching_tools import get_peak_indices
from .matching_tools_frequency_filters import perdecomp

def directional_spectrum(I, num_estimates=1): #todo: bring into correct framework
    """ look at the directional spectra, to get dominant frequency

    Parameters
    ----------
    I : numpy.ndarray, size=(m,n)
        satellite image (prefferably with sun-glitter)
    num_estimates : integer
        amount of spectral peaks to be estimated
    Returns
    -------
    di,dj : floats
        off-angle
    val : floats
        associated scores, of similar size as "di,dj"

    Notes
    -----
    .. [KuXX] Kudryavtsev et al. "Sun glitter imagery of surface waves. Part 1:
              Directional spectrum retrieval and validation" Journal of
              geophysical research: oceans. vol.122 pp.1369-1383
    """
    # spectral pre-processing
    per,_ = perdecomp(I)

    # spectral transform
    spec = np.fft.fft2(per)
    spec = np.log10(np.abs(np.fft.fftshift(spec)))

    # find peaks
    idx,val = get_peak_indices(spec, num_estimates=num_estimates)

    di, dj = idx[:,0], idx[:,1]
    di -= spec.shape[0] // 2
    dj -= spec.shape[1] // 2
    return di, dj, val