import numpy as np

from scipy import ndimage  # for image filtering
from skimage.feature import match_template


# image matching functions
def LucasKanade(I1, I2, window_size, stepSize=False, tau=1e-2):  # processing
    """
    displacement estimation through optical flow
    following Lucas & Kanade 1981
    input:   I1             array (n x m)     image with intensities
             I2             array (n x m)     image with intensities
             window_size    integer           kernel size of the neighborhood
             stepSize       boolean
             tau            float             smoothness parameter
    output:  u              array (n x m)     displacement estimate
             v              array (n x m)     displacement estimate
    """
    kernel_x = np.array(
        [[-1., 1.],
         [-1., 1.]]
    )
    kernel_t = np.array(
        [[1., 1.],
         [1., 1.]]
    ) * .25
    radius = np.floor(window_size / 2).astype(
        'int')  # window_size should be odd

    fx = ndimage.convolve(I1, kernel_x)
    fy = ndimage.convolve(I1, np.flip(np.transpose(kernel_x), axis=0))
    ft = ndimage.convolve(I2, kernel_t) + ndimage.convolve(I1, -kernel_t)

    # double loop to visit all pixels
    if stepSize:
        stepsI = np.arange(radius, I1.shape[0] - radius, window_size)
        stepsJ = np.arange(radius, I1.shape[1] - radius, window_size)
        u = np.zeros((len(stepsI), len(stepsJ)))
        v = np.zeros((len(stepsI), len(stepsJ)))
    else:
        stepsI = range(radius, I1.shape[0] - radius)
        stepsJ = range(radius, I1.shape[1] - radius)
        u = np.zeros(I1.shape)
        v = np.zeros(I1.shape)

    for iIdx in range(len(stepsI)):
        i = stepsI[iIdx]
        for jIdx in range(len(stepsJ)):
            j = stepsI[jIdx]
            # get templates
            Ix = fx[i - radius:i + radius + 1,
                    j - radius:j + radius + 1].flatten()
            Iy = fy[i - radius:i + radius + 1,
                    j - radius:j + radius + 1].flatten()
            It = ft[i - radius:i + radius + 1,
                    j - radius:j + radius + 1].flatten()

            # look if variation is present
            if np.std(It) != 0:
                b = np.reshape(It, (It.shape[0], 1))  # get b here
                A = np.vstack((Ix, Iy)).T  # get A here
                # threshold tau should be larger
                # than the smallest eigenvalue of A'A
                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                    nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                    u[iIdx, jIdx] = nu[0]
                    v[iIdx, jIdx] = nu[1]

    return (u, v)


def NormalizedCrossCorr(I1, I2):  # processing
    """
    Simple normalized cross correlation
    input:   I1             array (n x m)     template with intensities
             I2             array (n x m)     search space with intensities
    output:  x              integer           location of maximum
             y              integer           location of maximum
    """
    result = match_template(I1, I2)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    return (x, y)
