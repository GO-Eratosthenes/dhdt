import numpy as np

from tqdm import tqdm

# image libraries
from skimage.transform import radon
from scipy.signal import find_peaks

# local functions
from ..generic.mapping_tools import map2pix
from .matching_tools import pad_radius
from .matching_tools_frequency_filters import low_pass_circle
from .coupling_tools import create_template_at_center

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
    theta = np.linspace(-90, +90, 36, endpoint=False)
    circle = low_pass_circle(I)
    sinogram = radon(I, theta, circle=False)

    if len(np.unique(I))==I.size:
        breakpoint

    # entropy
    sino_std = np.std(sinogram, axis=0)
    sino_med = np.median(np.stack((np.roll(sino_std, -1),
                                   sino_std,
                                   np.roll(sino_std, +1))), axis=0)
    if fitting in ['polynomial']:
        theta_x = np.linspace(-90, +90, 360, endpoint=False)
        poly = np.poly1d(np.polyfit(theta, sino_med, 12))
        breakpoint
    else:
        # local peaks
        idx, properties = find_peaks(sino_std,distance=3)
        idx = idx[:num_dir]
        theta_hat, score = theta[idx], sino_std[idx]
    return theta_hat, score

def radon_orientation(I, geoTransform, X_grd, Y_grd,
                      temp_radius=7, num_dir=1, fitting='polynomial'):
    Theta, Score = np.zeros_like(X_grd), np.zeros_like(X_grd)

    # todo
#    if num_dir>1:
#        np.repeat()
    (m,n) = X_grd.shape
    assert(X_grd.shape == Y_grd.shape) # should be of the same size
    # preparation
    I = pad_radius(I,temp_radius)
    I_grd,J_grd = map2pix(geoTransform, X_grd, Y_grd)
    I_grd,J_grd = I_grd.flatten(), J_grd.flatten()
    I_grd,J_grd = np.round(I_grd).astype(int), np.round(J_grd).astype(int)
    I_grd += temp_radius
    J_grd += temp_radius

    for counter in tqdm(range(len(I_grd))):
        I_sub = create_template_at_center(I, I_grd[counter], J_grd[counter],
                                          temp_radius)
        idx_grd = np.unravel_index(counter, (m, n), 'C')
        if np.ptp(I_sub)==0:
            Theta[idx_grd[0],idx_grd[1]], Score[idx_grd[0],idx_grd[1]] = 0, 0
            continue

        theta, score = get_radon_angle(I_sub, num_dir=num_dir, fitting=fitting)
        # write results
        Theta[idx_grd[0], idx_grd[1]] = theta
        Score[idx_grd[0], idx_grd[1]] = score
    return Theta, Score

# get_orientation_of_two_subsets(I1_sub, I2_sub)