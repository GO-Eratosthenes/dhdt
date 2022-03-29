import numpy as np

from ..generic.test_tools import construct_phase_values,\
    cross_spectrum_to_coordinate_list
from ..generic.filtering_statistical import mad_filtering

from .matching_tools_frequency_filters import normalize_power_spectrum

def phase_fitness(Q, di, dj, norm=2):
    assert type(Q) == np.ndarray, ('please provide an array')
    # admin
    data = cross_spectrum_to_coordinate_list(Q)
    C = construct_phase_values(data[:,0:2], di, dj)

    # calculation
    QC = (data[:,-1] - C) ** norm
    dXY = np.abs(QC)
    fitness = (1-np.divide(np.sum(dXY) , 2*norm*dXY.size))**norm
    return fitness

def phase_support(Q, di, dj, thres=1.4826):
    assert type(Q) == np.ndarray, ('please provide an array')

    data = cross_spectrum_to_coordinate_list(Q)
    C = construct_phase_values(data[:,0:2], di, dj)

    # calculation
    QC = (data[:,-1] - C) ** 2
    dXY = np.abs(QC)
    IN = mad_filtering(dXY, thres=thres)

    support = np.divide(np.sum(IN), np.prod(IN.shape))
    return support

def signal_to_noise(Q, C, norm=2):
    """ calculate the signal to noise from a theoretical and an experimental
    cross-spectrum

    Parameters
    ----------
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    C : np.array, size=(m,n), dtype=complex
        phase plane
    norm : integer
        norm for the difference

    Returns
    -------
    snr : float, range=0...1
        signal to noise ratio

    See Also
    --------
    phase_fitness
    """
    assert type(Q) == np.ndarray, ('please provide an array')
    assert type(C) == np.ndarray, ('please provide an array')

    Qn = normalize_power_spectrum(Q)
    Q_diff = np.abs(Qn-C)**norm
    snr = 1 - (np.sum(Q_diff) / (2*norm*np.prod(C.shape)))
    return snr

def local_coherence(Q, ds=1):
    """ estimate the local coherence of a spectrum

    Parameters
    ----------
    Q : np.array, size=(m,n), dtype=complex
        array with cross-spectrum, with centered coordinate frame
    ds : integer, default=1
        kernel radius to describe the neighborhood

    Returns
    -------
    M : np.array, size=(m,n), dtype=float
        vector coherence from no to ideal, i.e.: 0...1

    See Also
    --------
    thresh_masking

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from ..generic.test_tools import create_sample_image_pair

    >>> # create cross-spectrum with random displacement
    >>> im1,im2,_,_,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> spec1,spec2 = np.fft.fft2(im1), np.fft.fft2(im2)
    >>> Q = spec1 * np.conjugate(spec2)
    >>> Q = normalize_spectrum(Q)
    >>> Q = np.fft.fftshift(Q) # transform to centered grid

    >>> C = local_coherence(Q)

    >>> plt.imshow(C), cmap='OrRd'), plt.colorbar(), plt.show()
    >>> plt.imshow(Q), cmap='twilight'), plt.colorbar(), plt.show()
    """
    assert type(Q) == np.ndarray, ("please provide an array")

    diam = 2 * ds + 1
    C = np.zeros_like(Q)
    (isteps, jsteps) = np.meshgrid(np.linspace(-ds, +ds, 2 * ds + 1, dtype=int), \
                                   np.linspace(-ds, +ds, 2 * ds + 1, dtype=int))
    IN = np.ones(diam ** 2, dtype=bool)
    IN[diam ** 2 // 2] = False
    isteps, jsteps = isteps.flatten()[IN], jsteps.flatten()[IN]

    for idx, istep in enumerate(isteps):
        jstep = jsteps[idx]
        Q_step = np.roll(Q, (istep, jstep))
        # if the spectrum is normalized, then no division is needed
        C += Q * np.conj(Q_step)
    C = np.abs(C) / np.sum(IN)
    return C