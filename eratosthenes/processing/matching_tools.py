import numpy as np

from scipy import ndimage  # for image filtering
from skimage.feature import match_template

from eratosthenes.generic.filtering_statistical import make_2D_Gaussian

# image matching functions
def lucas_kanade(I1, I2, window_size, sampleI, sampleJ, tau=1e-2):  # processing
    """
    displacement estimation through optical flow
    following Lucas & Kanade 1981
    input:   I1             array (n x m)     image with intensities
             I2             array (n x m)     image with intensities
             window_size    integer           kernel size of the neighborhood
             sampleI        array (k x l)     grid with image coordinates
             sampleJ        array (k x l)     grid with image coordinates
             tau            float             smoothness parameter
    output:  Ugrd           array (k x l)     displacement estimate
             Vgrd           array (k x l)     displacement estimate
    """
    kernel_x = np.array(
        [[-1., 1.],
         [-1., 1.]]
    )
    kernel_t = np.array(
        [[1., 1.],
         [1., 1.]]
    ) * .25

    fx = ndimage.convolve(I1, kernel_x)
    fy = ndimage.convolve(I1, np.flip(np.transpose(kernel_x), axis=0))
    ft = ndimage.convolve(I2, kernel_t) + ndimage.convolve(I1, -kernel_t)

    # grid or single estimation
    Ugrd = np.zeros((len(sampleI), len(sampleJ)))
    Vgrd = np.zeros((len(sampleI), len(sampleJ)))

    radius = np.floor(window_size / 2).astype(
            'int')  # window_size should be odd
    for iIdx in range(sampleI.size):
        iIm = sampleI.flat[iIdx]
        jIm = sampleJ.flat[iIdx]
        
        (iGrd, jGrd) = np.unravel_index(iIdx, sampleI.shape) 
        
        # get templates
        Ix = fx[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()
        Iy = fy[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()
        It = ft[iIm - radius:iIm + radius + 1,
                jIm - radius:jIm + radius + 1].flatten()

        # look if variation is present
        if np.std(It) != 0:
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here
            # threshold tau should be larger
            # than the smallest eigenvalue of A'A
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                Ugrd[iGrd, jGrd] = nu[0]
                Vgrd[iGrd, jGrd] = nu[1]

    return (Ugrd, Vgrd)

def lucas_kanade_single(I1, I2, weighting=True, tau=1e-2):  # processing
    """
    displacement estimation through optical flow
    following Lucas & Kanade 1981
    input:   I1             array (n x m)     image with intensities
             I2             array (n x m)     image with intensities
             tau            float             smoothness parameter
    output:  u              float             displacement estimate
             v              float             displacement estimate
    """
    kernel_x = np.array(
        [[-1., 1.],
         [-1., 1.]]
    )
    kernel_t = np.array(
        [[1., 1.],
         [1., 1.]]
    ) * .25

    Ix = ndimage.convolve(I1, kernel_x)
    Iy = ndimage.convolve(I1, np.flip(np.transpose(kernel_x), axis=0))
    It = ndimage.convolve(I2, kernel_t) + ndimage.convolve(I1, -kernel_t)

    # look if variation is present
    if np.std(It) != 0:
        b = np.reshape(It, (It.size, 1))  # get b here
        A = np.hstack((np.reshape(Ix, (Ix.size, -1)), 
                       np.reshape(Iy, (Iy.size, -1))
                       ))  # get A here
        
        # incorparate weighting, with emphasis on the center
        if weighting is True:
            W = make_2D_Gaussian(I1.shape, np.min(I1.shape))
            W = np.reshape(W, (W.size,-1))
            A = A * np.sqrt(np.diagonal(W))
            b = b * np.sqrt(np.diagonal(W))
        
        # threshold tau should be larger
        # than the smallest eigenvalue of A'A
        if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
            nu = np.matmul(np.linalg.pinv(A), b)  # get displacement here
            u = nu[0]
            v = nu[1]
        else:
            u = 0
            v = 0

    return (u, v)


def normalized_cross_corr(I1, I2):  # processing
    """
    Simple normalized cross correlation
    input:   I1             array (n x m)     template with intensities
             I2             array (n x m)     search space with intensities
    output:  x              integer           location of maximum
             y              integer           location of maximum
    """
    result = match_template(I2, I1)
    max_corr = np.amax(result)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    return (x, y, max_corr)

def cumulative_cross_corr(I1, I2):  # processing
    """
    Simple normalized cross correlation
    input:   I1             array (n x m)     template with maks
             I2             array (n x m)     search space with logical values
    output:  x              integer           location of maximum
             y              integer           location of maximum
    """
    
    # get cut-off value
    # cu = I1[int(np.floor(np.shape(I1)[0]/2)),int(np.floor(np.shape(I1)[1]/2))]
    # cu = np.quantile(I1, 0.5)
    # I1new = I1<cu
    
    I1new = ndimage.distance_transform_edt(I1)
    I2new = ndimage.distance_transform_edt(I2)
    
    result = match_template(I2new, I1new)
    max_corr = np.amax(result)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    return (x, y, max_corr)

# boundary describtors
def beam_angle_statistics(x, y, K=5):
    """
    implements beam angular statistics (BAS)
    
    input:
        
    output:
        
    see Arica & Vural, 2003
    BAS: a perceptual shape descriptoy based on the beam angle statistics
    Pattern Recognition Letters 24: 1627-1639
    
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
    """
    for i in range(1,K+1):
        if i ==1:
            x_minus =  np.expand_dims(np.roll(x, +i), axis= 1)
            y_minus = np.expand_dims(np.roll(y, +i), axis= 1)
            x_plus = np.expand_dims(np.roll(x, -i), axis= 1)
            y_plus =  np.expand_dims(np.roll(y, -i), axis= 1)
        else:
            x_new = np.expand_dims(np.roll(x, +i), axis=1)
            x_minus = np.concatenate((x_minus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, +i), axis=1)
            y_minus = np.concatenate((y_minus, y_new), axis=1)
            x_new = np.expand_dims(np.roll(x, -i), axis=1)
            x_plus = np.concatenate((x_plus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, -i), axis=1)
            y_plus = np.concatenate((y_plus, y_new), axis=1)
            del x_new, y_new
    dx_minus = x_minus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_minus = y_minus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)
    dx_plus = x_plus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_plus = y_plus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)
    
    # dot product instead of argument
    C_minus = np.arctan2(dy_minus, dx_minus)
    C_plus = np.arctan2(dy_plus, dx_plus)
    
    C = C_minus-C_plus
    C_1 = np.mean(C, axis=1)
    C_2 = np.std(C, axis=1)
    BAS = np.concatenate((np.expand_dims(C_1, axis=1),
                          np.expand_dims(C_2, axis=1)), axis=1);
    return BAS

def cast_angle_neighbours(x, y, sun, K=5):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''
    for i in range(1,K+1):
        if i ==1:
            x_minus =  np.expand_dims(np.roll(x, +i), axis= 1)
            y_minus = np.expand_dims(np.roll(y, +i), axis= 1)
            x_plus = np.expand_dims(np.roll(x, -i), axis= 1)
            y_plus =  np.expand_dims(np.roll(y, -i), axis= 1)
        else:
            x_new = np.expand_dims(np.roll(x, +i), axis=1)
            x_minus = np.concatenate((x_minus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, +i), axis=1)
            y_minus = np.concatenate((y_minus, y_new), axis=1)
            x_new = np.expand_dims(np.roll(x, -i), axis=1)
            x_plus = np.concatenate((x_plus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, -i), axis=1)
            y_plus = np.concatenate((y_plus, y_new), axis=1)
            del x_new, y_new
    
    dx_minus = x_minus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_minus = y_minus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)
    dx_plus = x_plus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_plus = y_plus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)    
    
    # rotate towards the sun
    CAN = np.concatenate((np.arctan2(sun[0]*dx_minus + sun[1]*dy_minus, 
                      -sun[1]*dx_minus + sun[0]*dy_minus), 
           np.arctan2(sun[0]*dx_plus + sun[1]*dy_plus, 
                      -sun[1]*dx_plus + sun[0]*dy_plus)), axis=1)
    return CAN

def neighbouring_cast_distances(x, y, sun, K=5):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''    
    for i in range(1,K+1):
        if i ==1:
            x_minus =  np.expand_dims(np.roll(x, +i), axis= 1)
            y_minus = np.expand_dims(np.roll(y, +i), axis= 1)
            x_plus = np.expand_dims(np.roll(x, -i), axis= 1)
            y_plus =  np.expand_dims(np.roll(y, -i), axis= 1)
        else:
            x_new = np.expand_dims(np.roll(x, +i), axis=1)
            x_minus = np.concatenate((x_minus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, +i), axis=1)
            y_minus = np.concatenate((y_minus, y_new), axis=1)
            x_new = np.expand_dims(np.roll(x, -i), axis=1)
            x_plus = np.concatenate((x_plus, x_new), axis=1)
            y_new = np.expand_dims(np.roll(y, -i), axis=1)
            y_plus = np.concatenate((y_plus, y_new), axis=1)
            del x_new, y_new
    
    dx_minus = x_minus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_minus = y_minus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)
    dx_plus = x_plus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dy_plus = y_plus - np.repeat(np.expand_dims(y, axis=1), K, axis=1)    
    # rotate towards the sun and take only one axes
    CD = np.concatenate((sun[0]*dx_minus + sun[1]*dy_minus, 
          sun[0]*dx_plus + sun[1]*dy_plus), axis=1)
    return CD    

def get_coordinates_of_template_centers(grid, temp_size):
    """
    When tiling an array into small templates, this function
    gives the locations of the centers.
    input:   grid           array (n x m)     array with data values
             temp_size       integer           size of the kernel in pixels
    output:  Iidx           array (k x l)     array with row coordinates
             Jidx           array (k x l)     array with collumn coordinates
    """
    radius = np.floor(temp_size / 2).astype('int')
    Iidx = np.arange(radius, grid.shape[0] - radius, temp_size)
    Jidx = np.arange(radius, grid.shape[1] - radius, temp_size)
    # FN ###################################
    # are the following lines equivalent to:
    # Iidx, Jidx = np.meshgrid(Iidx, Jidx)
    # It looks like, but Iidx and Jidx are switched!
    IidxNew = np.repeat(np.transpose([Iidx]), len(Jidx), axis=1)
    Jidx = np.repeat([Jidx], len(Iidx), axis=0)
    Iidx = IidxNew

    return Iidx, Jidx

def get_grid_at_template_centers(grid, temp_size):
    """
    When tiling an array into small templates, this function
    gives the value of the pixel in its center.
    input:   grid           array (n x m)     array with data values
             temp_size       integer           size of the kernel in pixels
    output:  gridnew        array (k x l)     data value of the pixel in the
                                              kernels center
    """
    (Iidx, Jidx) = get_coordinates_of_template_centers(grid, temp_size)

    return grid[Iidx, Jidx]