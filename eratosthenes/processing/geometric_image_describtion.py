import numpy as np

# image libraries
from skimage.transform import radon
from scipy.signal import find_peaks

# local functions

def get_radon_angle(I, num_dir=1, fitting='polynomial'):
    """ get the major directional tendencies within an image.

    Parameters
    ----------
    I : np.array, size=(m,n)
        image with intensities
    num_dir : positive integer
        amount of directions to estimate
    fitting : {'polynomial', 'median'}
        fitting function used for the localization

    Returns
    -------
    theta : np.array, float, unit=degree
        predominant direction(s) of the image
    score : np.array, float
        relative strength of the directional signal

    References
    ----------
    .. [1] Gong et al. "Simulating the roles of crevasse routing of surface
       water and basal friction on the surge evolution of Basin 3, Austfonna
       ice-cap" The cryosphere, vol.12(5) pp.1563-1577, 2018.
    .. [2] https://www.runmycode.org/companion/view/2711
    """

    theta = np.linspace(-90, +90, 36, edpoint=False)
    sinogram = radon(I, theta)

    # entropy
    sino_std = np.std(sinogram, axis=1)
    sino_med = np.median(np.stack(np.roll(sino_std, -1),
                                  sino_std,
                                  np.roll(sino_std, +1)), axis=1)
    if fitting in ['polynomial']:
        theta_x = np.linspace(-90, +90, 360, edpoint=False)
        poly = np.polyfit(theta, sino_med, 12)
        
    else:
        # local peaks
        idx, properties = find_peaks(sino_std,distance=3)
        num_dir
        score = []
    return theta[idx], score

# get_orientation_of_two_subsets(I1_sub, I2_sub)