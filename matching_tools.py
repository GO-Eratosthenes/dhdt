# general libraries
import warnings
import numpy as np

# image processing libraries
from scipy import ndimage, interpolate, fft, signal
from scipy.optimize import fsolve
from skimage.feature import match_template
from skimage.transform import radon
from skimage.measure import ransac
from sklearn.cluster import KMeans

from eratosthenes.generic.filtering_statistical import make_2D_Gaussian, \
    mad_filtering
from eratosthenes.generic.handler_im import get_grad_filters
from eratosthenes.preprocessing.shadow_transforms import pca



# spatial sub-pixel allignment functions
def simple_optical_flow(I1, I2, window_size, sampleI, sampleJ, tau=1e-2):  # processing
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

def affine_optical_flow(I1, I2, model='Affine', iteration=15):
    """
    displacement estimation through optical flow
    following Lucas & Kanade 1981 with an affine model
    
    reference:
              
    :param I1:        NP.ARRAY (_,_)
        image with intensities
    :param I2:        NP.ARRAY (_,_)
        image with intensities
    :param model:     STRING
           :options :  Affine - affine and translation
                       Rotation - TODO
    :param iteration: INTEGER
        number of iterations used
                     
    :return u:        FLOAT
        displacement estimate
    :return v:        FLOAT     
        displacement estimate                    
    """

    (kernel_j,_) = get_grad_filters('kroon')

    kernel_t = np.array(
        [[1., 1., 1.],
         [1., 2., 1.],
         [1., 1., 1.]]
    ) / 10
    
    # smooth to not have very sharp derivatives
    I1 = ndimage.convolve(I1, make_2D_Gaussian((3,3),fwhm=3))
    I2 = ndimage.convolve(I2, make_2D_Gaussian((3,3),fwhm=3))
    
    # calculate spatial and temporal derivatives
    I_dj = ndimage.convolve(I1, kernel_j)
    I_di = ndimage.convolve(I1, np.flip(np.transpose(kernel_j), axis=0))

    # create local coordinate grid
    (mI,nI) = I1.shape
    mnI = I1.size
    (grd_i,grd_j) = np.meshgrid(np.linspace(-(mI-1)/2, +(mI-1)/2, mI), \
                                np.linspace(-(nI-1)/2, +(nI-1)/2, nI), \
                                indexing='ij')
    grd_j = np.flipud(grd_j)
    stk_ij = np.vstack( (grd_i.flatten(), grd_j.flatten()) ).T
    p = np.zeros((1,6), dtype=float)
    p_stack = np.zeros((iteration,6), dtype=float)
    res = np.zeros((iteration,1), dtype=float) # look at iteration evolution
    for i in np.arange(iteration):
        # affine transform
        Aff = np.array([[1, 0, 0], [0, 1, 0]]) + p.reshape(3,2).T
        grd_new = np.matmul(Aff,
                            np.vstack((stk_ij.T, 
                                       np.ones(mnI))))
        new_i = np.reshape(grd_new[0,:], (mI, nI))
        new_j = np.reshape(grd_new[1,:], (mI, nI))
        
        # construct new templates
        try:
            I2_new = interpolate.griddata(stk_ij, I2.flatten().T,
                                          (new_i,new_j), method='cubic')
        except:
            print('different number of values and points')
        I_di_new = interpolate.griddata(stk_ij, I_di.flatten().T,
                                        (new_i,new_j), method='cubic')
        I_dj_new = interpolate.griddata(stk_ij, I_dj.flatten().T,
                                        (new_i,new_j), method='cubic')

        # I_dt = ndimage.convolve(I2_new, kernel_t) + 
        #    ndimage.convolve(I1, -kernel_t)
        I_dt_new = I2_new - I1

        # compose Jacobian and Hessian
        dWdp = np.array([ \
                  I_di_new.flatten()*grd_i.flatten(),
                  I_dj_new.flatten()*grd_i.flatten(),
                  I_di_new.flatten()*grd_j.flatten(),
                  I_dj_new.flatten()*grd_j.flatten(),
                  I_di_new.flatten(),
                  I_dj_new.flatten()])
        # dWdp = np.array([ \
        #           I_di.flatten()*grd_i.flatten(),
        #           I_dj.flatten()*grd_i.flatten(),
        #           I_di.flatten()*grd_j.flatten(),
        #           I_dj.flatten()*grd_j.flatten(),
        #           I_di.flatten(),
        #           I_dj.flatten()])
        
        # remove data outside the template
        A, y = dWdp.T, I_dt_new.flatten()
        IN = ~(np.any(np.isnan(A), axis=1) | np.isnan(y))
        
        A = A[IN,:] 
        y = y[~np.isnan(y)]
        
        #(dp,res[i]) = least_squares(A, y, mode='andrews', iterations=3) 
        if y.size>=6: # structure should not become ill-posed
            try:
                (dp,res[i],_,_) = np.linalg.lstsq(A, y, rcond=None)#[0]
            except ValueError:
                pass #print('something wrong?')
        else:
            break
        p += dp
        p_stack[i,:] = p
    
    # only convergence is allowed 
    (up_idx,_) = np.where(np.sign(res-np.vstack(([1e3],res[:-1])))==1)
    if up_idx.size != 0:
        res = res[:up_idx[0]]
    
    if res.size == 0: # sometimes divergence occurs
        A = np.array([[1, 0], [0, 1]])
        u, v, snr = 0, 0, 0
    else:
        Aff = np.array([[1, 0, 0], [0, 1, 0]]) + \
            p_stack[np.argmin(res),:].reshape(3,2).T
        u, v = Aff[0,-1], Aff[1,-1]   
        A = np.linalg.inv(Aff[:,0:2]).T
        snr = np.min(res)
    return (u, v, A, snr)   

# spatial pattern matching functions
def normalized_cross_corr(I1, I2):
    """
    Simple normalized cross correlation

    :param I1:        NP.ARRAY (_,_)
        image with intensities
    :param I2:        NP.ARRAY (_,_)
        image with intensities
                     
    :return result:   NP.ARRAY (_,_)
        similarity surface           
    """
    result = match_template(I2, I1)
    return result

def cumulative_cross_corr(I1, I2):
    """
    doing normalized cross correlation on distance imagery

    :param I1:        NP.ARRAY (_,_)
        binary array
    :param I2:        NP.ARRAY (_,_)
        binary array
                     
    :return result:   NP.ARRAY (_,_)
        similarity surface           
    """   
    if isinstance(I1, np.floating):        
        # get cut-off value
        cu = np.quantile(I1, 0.5)
        I1 = I1<cu
    if isinstance(I1, np.floating):        
        # get cut-off value
        cu = np.quantile(I2, 0.5)
        I2 = I2<cu
        
    I1new = ndimage.distance_transform_edt(I1)
    I2new = ndimage.distance_transform_edt(I2)
    
    result = match_template(I2new, I1new)
    return result

def sum_sq_diff(I1, I2):
    """
    Simple normalized cross correlation

    :param I1:        NP.ARRAY (_,_)
        image with intensities
    :param I2:        NP.ARRAY (_,_)
        image with intensities
                     
    :return ssd:      NP.ARRAY (_,_)
        dissimilarity surface           
    """    
    
    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                    shape=(I2.shape[0] - t_size[0] + 1,
                           I2.shape[1] - t_size[1] + 1,) +
                          t_size,
                    strides=I2.strides * 2)
    ssd = np.einsum('ijkl,kl->ij', y, I1)
    ssd *= - 2
    ssd += np.einsum('ijkl, ijkl->ij', y, y)
    ssd += np.einsum('ij, ij', I1, I1)
    return ssd

def get_integer_peak_location(C):
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    
    ij = np.unravel_index(np.argmax(C), C.shape, order='F') # 'C'
    di, dj = ij[::-1]
    di -= C.shape[0] // 2
    dj -= C.shape[1] // 2
    return di, dj, snr, max_corr

# sub-pixel localization of the correlation peak    
def get_top_moment(C, ds=1, top=np.array([])):
    """ find location of highest score through bicubic fitting
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    ds : integer, default=1
        size of the radius to use neighboring information    
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----
    [1] Feng et al. "A subpixel registration algorithm for low PSNR images"
    IEEE international conference on advanced computational intelligence,
    pp. 626-630, 2012.    
    [2] Messerli & Grinstad, "Image georectification and feature tracking 
    toolbox: ImGRAFT" Geoscientific instrumentation, methods and data systems, 
    vol. 4(1) pp. 23-34, 2015. 
    """
    
    (subJ,subI) = np.meshgrid(np.linspace(-ds,+ds, 2*ds+1), np.linspace(-ds,+ds, 2*ds+1))    
    subI = subI.ravel()
    subJ = subJ.ravel()
    
    if top.size==0:
        # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
        
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else:# estimate sub-pixel top
        idx_mid = np.int(np.floor((2.*ds+1)**2/2))
        
        Csub = C[i-ds:i+ds+1,j-ds:j+ds+1].ravel()
        Csub = Csub - np.mean(np.hstack((Csub[0:idx_mid],Csub[idx_mid+1:])))
        
        IN = Csub>0
        
        m = np.array([ np.divide(np.sum(subI[IN]*Csub[IN]), np.sum(Csub[IN])) , 
                       np.divide(np.sum(subJ[IN]*Csub[IN]), np.sum(Csub[IN]))])
        ddi, ddj = m[0], m[1]

    return (ddi, ddj)

def get_top_blue(C, ds=1): # wip
    (subJ,subI) = np.meshgrid(np.linspace(-ds,+ds, 2*ds+1), np.linspace(-ds,+ds, 2*ds+1))    
    subI = subI.ravel()
    subJ = subJ.ravel()
    
    # find highest score
    y,x,max_corr,snr = get_integer_peak_location(C)
    
    # estimate Jacobian
    H_x = np.array([[-17., 0., 17.],
                   [-61., 0., 61.],
                   [-17., 0., 17.]]) / 95
    # estimate Hessian
    H_xx = 8 / np.array([[105, -46, 105],
                        [  50, -23,  50],
                        [ 105, -46, 105]] )
    H_xy = 11 / np.array([[-114, np.inf, +114],
                         [np.inf, np.inf, np.inf],
                         [+114, np.inf, -114]] )
    
    # estimate sub-pixel top   
    Csub = C[y-ds:y+ds+1,x-ds:x+ds+1]
    
    Jac = np.array([[Csub*H_x], [Csub*H_x.T]])
    Hes = Jac = np.array([[Csub*H_xx  , Csub*H_xy], 
                          [Csub*H_xy.T, Csub*H_xx.T]])
    
    m0 = np.array([[x], [y]]) - np.linalg.inv(Hes) * Jac
    return (m0[0], m0[1])

def get_top_gaussian(C, top=np.array([])):
    """ find location of highest score through 1D gaussian fit
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----    
    [1] Argyriou & Vlachos, "A Study of sub-pixel motion estimation using 
    phase correlation" Proceeding of the British machine vision conference, 
    pp. 387-396), 2006.   
    """
    if top.size==0: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else: # estimate sub-pixel along each axis   
        ddi = (np.log(C[i+1,j]) - np.log(C[i-1,j])) / \
            2*( (2*np.log(C[i,j])) -np.log(C[i-1,j]) -np.log(C[i+1,j]))
        ddj = (np.log(C[i,j+1]) - np.log(C[i,j-1])) / \
            2*( (2*np.log(C[i,j])) -np.log(C[i,j-1]) -np.log(C[i,j+1]))       
    return (ddi, ddj)

def get_top_parabolic(C, top=np.array([])):
    """ find location of highest score through 1D parabolic fit
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----    
    [1] Argyriou & Vlachos, "A Study of sub-pixel motion estimation using 
    phase correlation" Proceeding of the British machine vision conference, 
    pp. 387-396), 2006.   
    """
    
    if top.size==0: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else: # estimate sub-pixel along each axis   
        ddi = (C[i+1,j] - C[i-1,j]) / 2*( (2*C[i,j]) -C[i-1,j] -C[i+1,j])
        ddj = (C[i,j+1] - C[i,j-1]) / 2*( (2*C[i,j]) -C[i,j-1] -C[i,j+1])       
    return (ddi, ddj)

def get_top_birchfield(C, top=np.array([])):
    """ find location of highest score along each axis
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----    
    [1] Birchfield & Tomasi. "Depth discontinuities by pixel-to-pixel stereo" 
    International journal of computer vision, vol. 35(3)3 pp. 269-293, 1999.    
    """
    
    if top.size==0: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else:
        # estimate sub-pixel along each axis   
        I_m,I_p = .5*(C[i-1,j] + C[i,j]), .5*(C[i+1,j] + C[i,j])
        I_min,I_max = np.amin([I_m, I_p, C[i,j]]), np.amax([I_m, I_p, C[i,j]])  
        # swapped, since Birchfield uses dissimilarity
        ddi = np.amax([0, I_max-C[i,j], C[i,j]-I_min]) 
    
        I_m,I_p = .5*(C[i,j-1] + C[i,j]), .5*(C[i,j+1] + C[i,j])
        I_min,I_max = np.amin([I_m, I_p, C[i,j]]), np.amax([I_m, I_p, C[i,j]])  
        ddj = np.amax([0, I_max-C[i,j], C[i,j]-I_min])
    return (ddi, ddj)

def get_top_ren(C, top=None):
    """ find location of highest score
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----    
    [1] Ren et al. "High-accuracy sub-pixel motion estimation from noisy 
    images in Fourier domain." IEEE transactions on image processing,
    vol. 19(5) pp. 1379-1384, 2010.    
    """
    if top.size is None: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else:
        # estimate sub-pixel along each axis   
        D_i = C[i+1,j] - C[i-1,j]
        ddi = np.sign(D_i)/(1 + ( C[i,j] / np.abs(D_i) )) 
    
        D_j = C[i,j+1] - C[i,j-1]
        ddj = np.sign(D_j)/(1 + ( C[i,j] / np.abs(D_j) )) 
    return (ddi, ddj)

def get_top_triangular(C, top=None):
    """ find location of highest score through triangular fit
    
    Parameters
    ----------    
    C : np.array, size=(_,_)
        similarity surface
    top : np.array, size=(1,2)
        location of the maximum score
    
    Returns
    -------
    ddi : float
        estimated subpixel location on the vertical axis of the peak
    ddj : float
        estimated subpixel location on the horizontal axis of the peak
    
    Notes
    -----    
    [1] Olsen & Coombs, "Real-time vergence control for binocular robots" 
    International journal of computer vision, vol. 7(1), pp. 67-89, 1991.  
    """
    if top.size is None: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        ddi, ddj = 0, 0
    else:
        # estimate sub-pixel along each axis   
        I_m,I_p = C[i-1,j], C[i+1,j]
        I_min,I_max = np.amin([I_m, I_p]), np.amax([I_m, I_p])
        I_sign = 2*(I_p>I_m)-1
        ddi = I_sign * (1- (I_max-I_min)/(C[i,j]-I_min) )
    
        I_m,I_p = C[i,j-1], C[i,j+1]
        I_min,I_max = np.amin([I_m, I_p, C[i,j]]), np.amax([I_m, I_p, C[i,j]])  
        I_sign = 2*(I_p>I_m)-1
        ddj = I_sign * (1- (I_max-I_min)/(C[i,j]-I_min) )
    return (ddi, ddj)

def get_top_esinc(C, ds=1, top=None):
    '''
    find location of highest score using exponential esinc function
    following Argyriou & Vlachos 2006
        "A study of sub-pixel motion estimation using phase correlation"
    
    :param C:         NP.ARRAY (_,_)
        similarity surface
    :param top:       NP.ARRAY (1,2)
        location of the maximum score

    :return iC:       FLOAT  
        estimated subpixel location on the vertical axis of the peak
    :return jC:       FLOAT     
        estimated subpixel location on the horizontal axis of the peak
    '''    
    if top is None: # find highest score
        i,j,max_corr,snr = get_integer_peak_location(C)
    else:
        i, j = top[0], top[1]
    
    if (i==0) | (i!=C.shape[0]) | (j==0) | (j!=C.shape[1]): # top at the border
        iC, jC = 0, 0
    else:
        # estimate sub-pixel per axis
        Cj = C[i,j-ds:j+ds+1].ravel()
        def funcJ(x):
            a, b, c = x
            return [(Cj[0] - a*np.exp(-(b*(-1-c))**2)* \
                     ( np.sin(np.pi*(-1-c))/ np.pi*(-1-c)) )**2,
                    (Cj[1] - a*np.exp(-(b*(+0-c))**2)* \
                     ( np.sin(np.pi*(+0-c))/ np.pi*(+0-c)) )**2,
                    (Cj[2] - a*np.exp(-(b*(+1-c))**2)* \
                     ( np.sin(np.pi*(+1-c))/ np.pi*(+1-c)) )**2]
        
        jA, jB, jC = fsolve(funcJ, (1.0, 1.0, 0.1))    
        
        Ci = C[i-ds:i+ds+1,j].ravel()
        def funcI(x):
            a, b, c = x
            return [(Ci[0] - a*np.exp(-(b*(-1-c))**2)* \
                     ( np.sin(np.pi*(-1-c))/ np.pi*(-1-c)) )**2,
                    (Ci[1] - a*np.exp(-(b*(+0-c))**2)* \
                     ( np.sin(np.pi*(+0-c))/ np.pi*(+0-c)) )**2,
                    (Ci[2] - a*np.exp(-(b*(+1-c))**2)* \
                     ( np.sin(np.pi*(+1-c))/ np.pi*(+1-c)) )**2]
        
        iA, iB, iC = fsolve(funcI, (1.0, 1.0, 0.1)) 

    return (iC, jC)

#todo: Nobach_05, 
# paraboloid

# phase plane functions
def phase_tpss(Q, W, m, p=1e-4, l=4, j=5, n=3):
    """get phase plane of cross-spectrum through two point step size iteration
    
    find slope of the phase plane through 
    two point step size for phase correlation minimization
    
    Parameters
    ---------- 
    following Leprince et al. 2007
        
    
    :param Q:         NP.ARRAY (_,_)
        cross spectrum
    :param m:         NP.ARRAY (1,2)
        similarity surface
    :param p:         FLOAT
        closing error threshold
    :param l:         INTEGER
        number of refinements in iteration
    :param j:         INTEGER
        number of sub routines during an estimation
    :param n:         INTEGER
        mask convergence factor     

    :return m:        NP.ARRAY (2,1)
        displacement estimate
    :return snr:      FLOAT     
        signal-to-noise ratio
   
    Q : np.array, size=(_,_), dtype=complex
        cross spectrum
    m0 : np.array, size=(2,1)
        initial displacement estimate
    p : float, default=1e4
        closing error threshold
    l : integer, default=4
        number of refinements in iteration
    j : integer, default=5
        number of sub routines during an estimation
    n : integer, default=3
        mask convergence factor        
    Returns
    -------
    m : np.array, size=(2,1)   
        sub-pixel displacement
    snr: float 
        signal-to-noise ratio    
    
    See Also
    --------
    phase_svd, phase_radon, phase_difference   
    
    Notes
    -----    
    [1] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """
    
    (m_Q, n_Q) = Q.shape
    fy = 2*np.pi*(np.arange(0,m_Q)-(m_Q/2)) /m_Q
    fx = 2*np.pi*(np.arange(0,n_Q)-(n_Q/2)) /n_Q
        
    Fx = np.repeat(fx[np.newaxis,:],m_Q,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n_Q,axis=1)
    Fx = np.fft.fftshift(Fx)
    Fy = np.fft.fftshift(Fy)

    # initialize    
    m_min = m + np.array([+.1, +.1])

    C_min = 1j*-np.sin(Fx*m_min[1] + Fy*m_min[0])
    C_min += np.cos(Fx*m_min[1] + Fy*m_min[0])
    
    QC_min = Q-C_min # np.abs(Q-C_min) #Q-C_min np.abs(Q-C_min)
    dXY_min = np.multiply(2*W, (QC_min * np.conjugate(QC_min)) )
    
    g_min = np.real(np.array([np.nansum(Fy*dXY_min), \
                              np.nansum(Fx*dXY_min)]))
    
    print(m)
    for i in range(l):
        k = 1
        while True:
            C = 1j*-np.sin(Fx*m[1] + Fy*m[0])
            C += np.cos(Fx*m[1] + Fy*m[0])
            QC = Q-C # np.abs(Q-C)#np.abs(Q-C)
            dXY = 2*W*(QC*np.conjugate(QC))
            g = np.real(np.array([np.nansum(np.multiply(Fy,dXY)), \
                                  np.nansum(np.multiply(Fx,dXY))]))
            
            # difference
            dm = m - m_min
            dg = g - g_min
            
            alpha = np.dot(dm,dg)/np.dot(dg,dg)
            #alpha = np.dot(dm,dm)/np.dot(dm,dg)
            
            if (np.all(np.abs(m - m_min)<=p)) or (k>=j):
                break
            
            # update
            m_min, g_min, dXY_min = np.copy(m), np.copy(g), np.copy(dXY)
            
            m -= alpha*dg
            print(m)
            k += 1
            
        # optimize weighting matrix
        phi = np.abs(QC*np.conjugate(QC))/2
        W = W*(1-(dXY/8))**n
    snr = 1 - (np.sum(phi)/(4*np.sum(W)))
    return (m, snr)

def phase_svd(Q, W, rad=0.1):
    """get phase plane of cross-spectrum through single value decomposition
    
    find slope of the phase plane through 
    single value decomposition
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        cross spectrum
    W : np.array, size=(m,n), dtype=float
        weigthing matrix
    
    Returns
    -------
    di,dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_radon, phase_difference    
    
    Notes
    -----    
    [1] Hoge, W.S. "A subspace identification extension to the phase 
    correlation method", IEEE transactions on medical imaging, vol. 22.2 
    pp. 277-280, 2003.    
    """
    # filtering through magnitude
    # W: M = thresh_masking(S1, m=th, s=ker) th=0.001, ker=10
    
    (m,n) = Q.shape
    Q,W = np.fft.fftshift(Q), np.fft.fftshift(W)
    
    # decompose axis
    n_elements = 1
    u,s,v = np.linalg.svd(W*Q) # singular-value decomposition
    sig = np.zeros((m,n))
    sig[:m,:m] = np.diag(s)
    sig = sig[:,:n_elements] # select first element only
    # v = v[:n_elements,:]
    # reconstruct
    # b = u.dot(sig.dot(v))
    t_m = np.transpose(v).dot(sig)
    t_n = u.dot(sig)# transform
    
    idx_sub = np.arange(np.ceil((0.5-rad)*len(t_n)), \
                        np.ceil((0.5+rad)*len(t_n))+1).astype(int)
    y_ang = np.unwrap(np.angle(t_n[idx_sub]),axis=0)
    A = np.vstack([np.transpose(idx_sub-1), np.ones((len(idx_sub)))]).T
    (dx,_,_,_) = np.linalg.lstsq(A, y_ang, rcond=None)
    
    idx_sub = np.arange(np.ceil((0.5-rad)*len(t_m)), \
                        np.ceil((0.5+rad)*len(t_m))+1).astype(int)
    y_ang = np.unwrap(np.angle(t_m[idx_sub]), axis=0)
    (dy,_,_,_) = np.linalg.lstsq(A, y_ang, rcond=None)
    
    dj = dx[0]*n / (np.pi)
    di = dy[0]*m / (np.pi)
    return di, dj

def phase_difference_1d(Q, axis=0):
    """get displacement from phase plane along one axis through differencing
    
    find slope of the phase plane through 
    local difference of the pahse angles
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        normalized cross spectrum
    
    Returns
    -------
    dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_svd, phase_difference   
    
    Notes
    -----    
    [1] Kay, S. "A fast and accurate frequency estimator", IEEE transactions on 
    acoustics, speech and signal processing, vol.37(12) pp. 1987-1990, 1989.    
    """
    
    if axis==0:
        Q = np.transpose(Q)
    m,n = Q.shape
    
    # find coherent data
    C = local_coherence(Q, ds=1)
    C = np.minimum(C, np.roll(C, (0,1)))
    
    #estimate period
    Q_dj = np.roll(Q, (0,1))
    
    Q_diff = np.multiply(np.conj(Q),Q_dj)
    
    Delta_dj = np.angle(Q_diff)/np.pi
    IN = C>.9
    dj = np.median(Delta_dj[IN])*(m//2)
    return dj

def phase_difference(Q):
    """get displacement from phase plane through neighbouring vector difference
    
    find slope of the phase plane through 
    local difference of the pahse angles
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        normalized cross spectrum
    
    Returns
    -------
    di,dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_svd, phase_difference   
    
    Notes
    -----    
    [1] Kay, S. "A fast and accurate frequency estimator", IEEE transactions on 
    acoustics, speech and signal processing, vol.37(12) pp. 1987-1990, 1989.    
    """
    di = phase_difference_1d(Q, axis=0)
    dj = phase_difference_1d(Q, axis=1)
    
    return di,dj

def phase_lsq(I, J, Q):
    """get phase plane of cross-spectrum through least squares plane fitting
    
    find slope of the phase plane through 
    principle component analysis
    
    Parameters
    ----------    
    I : np.array, size=(mn,1), dtype=float
        vertical coordinate list
    J : np.array, size=(mn,1), dtype=float
        horizontal coordinate list
    Q : np.array, size=(mn,1), dtype=complex
        list with cross-spectrum complex values
    
    Returns
    -------
    di,dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_radon, phase_svd, phase_difference, phase_pca    
       
    """
    A = np.array([I, J, np.ones_like(I)]).T
    
    M = A.transpose().dot(A)
    V = A.transpose().dot(np.angle(Q) / (2*np.pi))

    # pseudoinverse:
    Mp = np.linalg.inv(M.transpose().dot(M)).dot(M.transpose())

    #Least-squares Solution
    plane_normal = Mp.dot(V) 
    
    di = plane_normal[0]
    dj = plane_normal[1]
    return di, dj

class BaseModel(object):
    def __init__(self):
        self.params = None

class PlaneModel(BaseModel):
    """Least squares estimator for phase plane.
    
    Vectors/lines are parameterized using polar coordinates as functional model::
        z = x * dx + y * dy
    This estimator minimizes the squared distances from all points to the
    line, independent of distance::
        min{ sum((dist - x_i * cos(theta) + y_i * sin(theta))**2) }
    A minimum number of 2 points is required to solve for the parameters.
    Attributes
    ----------
    params : tuple
        Plane model parameters in the following order `dist`, `theta`.
    """

    def estimate(self, data):
        """Estimate plane from data using least squares.
        Parameters
        ----------
        data : (N, 3) array
            N points with ``(x, y)`` coordinates of vector, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        if data.shape[0] >= 2:  # well determined
            x_hat = np.linalg.lstsq(data[:,0:2], data[:,-1], rcond=None)[0]
        else:  # under-determined
            raise ValueError('At least two vectors needed.')
        self.params = (x_hat[0], x_hat[1])
        return True

    def residuals(self, data):
        """Determine residuals of data to model
        For each point the shortest distance to the plane is returned.
        
        Parameters
        ----------
        data : (N, 3) array
            N points with x, y coordinates and z values, respectively.
        
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        x_hat = self.params
        
        Q_hat = data[:,0]*x_hat[0] + data[:,1]*x_hat[1]
        
        residuals = np.abs(data[:,-1] - Q_hat)
        return residuals

    def predict_xy(self, xy, params=None):
        """Predict vector using the estimated heading.
        Parameters
        ----------
        xy : array
            x,y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        Q_hat : array
            Predicted plane height at x,y-coordinates.
        """

        if params is None:
            params = self.params
        x_hat = params
        if xy.ndim<2:
            Q_hat = xy[0]*x_hat[0] + xy[1]*x_hat[1]
        else:
            Q_hat = xy[:,0]*x_hat[0] + xy[:,1]*x_hat[1]
        return Q_hat

def phase_ransac(Q, precision_threshold=.05):
    """robustly fit plane using RANSAC algorithm
    
    find slope of the phase plane through 
    random sampling and consensus
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        normalized cross spectrum
    
    Returns
    -------
    dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_svd, phase_difference   
    
    Notes
    -----    
    [1] Fischler & Bolles. "Random sample consensus: a paradigm for model 
    fitting with applications to image analysis and automated cartography" 
    Communications of the ACM vol.24(6) pp.381-395, 1981.   
    [2] Tong et al. "A novel subpixel phase correlation method using singular 
    value decomposition and unified random sample consensus" IEEE transactions 
    on geoscience and remote sensing vol.53(8) pp.4143-4156, 2015.
    """
    (m,n) = Q.shape
    fy = np.flip((np.arange(0,m)-(m/2)) /m)
    fx = np.flip((np.arange(0,n)-(n/2)) /n)
        
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1) 
    
    data = np.vstack((Fx.flatten(), 
                      Fy.flatten(), 
                      np.angle(Q).flatten())).T    
    ransac_model, inliers = ransac(data, PlaneModel, 
                                   min_samples=int(2), 
                                   residual_threshold=precision_threshold, 
                                   max_trials=int(1e3))
    IN = np.reshape(inliers, (m,n)) # what data is within error bounds
    
    di = ransac_model.params[0]/(2.*np.pi) 
    dj = ransac_model.params[1]/(2.*np.pi)
    return di, dj

def phase_pca(I, J, Q): # wip
    """get phase plane of cross-spectrum through principle component analysis
    
    find slope of the phase plane through 
    principle component analysis
    
    Parameters
    ----------    
    I : np.array, size=(mn,1), dtype=float
        vertical coordinate list
    J : np.array, size=(mn,1), dtype=float
        horizontal coordinate list
    Q : np.array, size=(mn,1), dtype=complex
        list with cross-spectrum complex values
    
    Returns
    -------
    di,dj : float     
        sub-pixel displacement
    
    See Also
    --------
    phase_tpss, phase_radon, phase_svd, phase_difference    
       
    """
    eigen_vecs, eigen_vals = pca(np.array([I, J, np.angle(Q)/np.pi]).T)
    e3 = eigen_vecs[:,np.argmin(eigen_vals)] # normal vector
    e3 = eigen_vecs[:,np.argmin(np.abs(eigen_vecs[-1,:]))]
    di = np.sign(e3[-1])*e3[1]
    dj = np.sign(e3[-1])*e3[0]
    return di, dj

def phase_radon(Q): # wip
    """get direction and magnitude from phase plane through Radon transform
    
    find slope of the phase plane through 
    single value decomposition
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        cross spectrum
    
    Returns
    -------
    rho,theta : float     
        magnitude and direction of displacement
    
    See Also
    --------
    phase_tpss, phase_svd, phase_difference   
    
    Notes
    -----    
    [1] Balci & Foroosh. "Subpixel registration directly from the phase 
    difference" EURASIP journal on advances in signal processing, pp.1-11, 2006.
    """
    (m, n) = Q.shape  
    half = m // 2
    Q = np.fft.fftshift(Q)
    
    # estimate direction, through the radon transform
    W = np.fft.fftshift(raised_cosine(Q, beta=1e-5)).astype(bool)
    Q[~W] = 0 # make circular domain
    
    theta = np.linspace(0., 180., max(m,n), endpoint=False)
    R = radon(np.angle(Q), theta) # sinogram
        
    #plt.imshow(R[:half,:]), plt.show()
    #plt.imshow(np.flipud(R[half:,:])), plt.show()
    
    R_fold = np.abs(np.multiply(R[:half,:], R[half:,:]))
    radon_score = np.sum(R_fold, axis=0)
    score_idx = np.argmax(radon_score)
    theta = theta[score_idx]
    del R_fold, radon_score, score_idx

    # rotating coordinate frame, angle difference

    # estimate magnitude
    # peaks can also be seen
    # plt.plot(R[:,score_idx]), plt.show()
    
    return theta
    
#todo: Gonzalez_10:RANSAC->LDA, Konstantinidis_ PAC, Fienup_82: Gradient-descend, Yousef_05:

# frequency preparation
def perdecomp(img):
    """calculate the periodic and smooth components of an image
       
    Parameters
    ----------    
    img : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    per : np.array, size=(m,n)
        periodic component
    cor : np.array, size=(m,n)
        smooth component   
    
    Notes
    -----    
    [1] Moisan, L. "Periodic plus smooth image decomposition", Journal of 
    mathematical imaging and vision vol. 39.2 pp. 161-179, 2011.    
    """
    img = img.astype(float)
    if img.ndim==2:
        (m, n) = img.shape
        per = np.zeros((m, n), dtype=float)
        
        per[+0,:] = +img[0,:] -img[-1,:]
        per[-1,:] = -per[0,:]
        
        per[:,+0] = per[:,+0] +img[:,+0] -img[:,-1]
        per[:,-1] = per[:,-1] -img[:,+0] +img[:,-1]
    
    elif img.ndim==3:
        (m, n, b) = img.shape
        per = np.zeros((m, n, b), dtype=float)
        
        per[+0,:,:] = +img[0,:,:] -img[-1,:,:]
        per[-1,:,:] = -per[0,:,:]
        
        per[:,+0,:] = per[:,+0,:] +img[:,+0,:] -img[:,-1,:]
        per[:,-1,:] = per[:,-1,:] -img[:,+0,:] +img[:,-1,:]
    
    fy = np.cos( 2*np.pi*( np.arange(0,m) )/m )
    fx = np.cos( 2*np.pi*( np.arange(0,n) )/n )

    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx[0,0] = 0
    if img.ndim==3:
        Fx = np.repeat(Fx[:,:,np.newaxis], b, axis=2)
        Fy = np.repeat(Fy[:,:,np.newaxis], b, axis=2)
        cor = np.real( np.fft.ifftn( np.fft.fft2(per) *.5/ (2-Fx-Fy)))
    else:
        cor = np.real( np.fft.ifft2( np.fft.fft2(per) *.5/ (2-Fx-Fy)))
    per = img-cor
    return (per, cor)    

def normalize_power_spectrum(Q):
    Qn = np.divide(Q, abs(Q), out=np.zeros_like(Q), where=Q!=0)
    return Qn
    
# frequency matching filters
def raised_cosine(I, beta=0.35):
    """ raised cosine filter
    
    Parameters
    ----------    
    img : np.array, size=(m,n)
        array with intensities
    beta : float, default=0.35
        roll-off factor
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    tpss   
    
    Notes
    -----    
    [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
    registration of images." IEEE Transactions on geoscience and remote sensing 
    vol. 39(10) pp. 2235-2243, 2001.
    [2] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    Hamm = np.cos( (np.pi/(2*beta)) * (R - (.5-beta)))**2
    selec = np.logical_and((.5 - beta) <= R , R<=.5)
    
    # compose filter
    W = np.zeros((m,n))
    W[(.5 - beta) > R] = 1
    W[selec] = Hamm[selec]
    return W 

# def hanning_window

def low_pass_rectancle(I, r=0.50):
    """ create hard low-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_circle, low_pass_pyramid, low_pass_bell
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.
    """
    
    (m, n) = I.shape
    fy = 2*np.mod(.5 + np.arange(0,m)/m , 1) -1 # fft shifted coordinate frame
    fx = 2*np.mod(.5 + np.arange(0,n)/n , 1) -1
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    # filter formulation 
    W = np.logical_and(np.abs(Fx)<=r, np.abs(Fy)<=r) 
    return W 

def low_pass_pyramid(I, r=0.50):
    """ create low-pass filter with pyramid shape
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_bell
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.    
    """
    R = low_pass_rectancle(I, r)
    W = signal.convolve2d(R.astype(float), R.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_bell(I, r=0.50):
    """ create low-pass filter with a bell shape
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the mother rectangle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    low_pass_rectancle, low_pass_circle, low_pass_pyramid
    
    Notes
    -----
    [1] Takita et al. "High-accuracy subpixel image registration based on 
    phase-only correlation" IEICE transactions on fundamentals of electronics, 
    communications and computer sciences, vol.86(8) pp.1925-1934, 2003.
    """
    R1 = low_pass_rectancle(I, r)
    R2 = low_pass_pyramid(I, r)
    W = signal.convolve2d(R1.astype(float), R2.astype(float), \
                          mode='same', boundary='wrap')
    W = np.fft.fftshift(W/np.max(W))
    return W

def low_pass_circle(I, r=0.50):
    """ create hard low-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, high_pass_circle
        
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    W = R<=r
    return W 

def high_pass_circle(I, r=0.50):
    """ create hard high-pass filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    r : float, default=0.5
        radius of the circle, r=.5 is same as its width
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=bool
        weighting mask
    
    See Also
    --------
    raised_cosine, cosine_bell, low_pass_circle
        
    """
    
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    # filter formulation 
    W = R>=r
    return W

def cosine_bell(I):
    """ cosine bell filter
    
    Parameters
    ----------    
    I : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    W : np.array, size=(m,n), dtype=float
        weighting mask
    
    See Also
    --------
    raised_cosine     
    """
    (m, n) = I.shape
    fy = np.mod(.5 + np.arange(0,m)/m , 1) -.5 # fft shifted coordinate frame
    fx = np.mod(.5 + np.arange(0,n)/n , 1) -.5
    
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    R = np.sqrt(Fx**2 + Fy**2) # radius
    
    # filter formulation 
    W = .5*np.cos(2*R*np.pi) + .5
    W[R>.5] = 0
    return W

def cross_shading_filter(Q): #, az_1, az_2): # wip
    (m,n) = Q.shape    
    Coh = local_coherence(np.fft.fftshift(Q))
    R = np.fft.fftshift(low_pass_circle(Q, r=0.50))
    Coh[R==0] = 0
    
    theta = np.linspace(0., 180., max(m,n), endpoint=False)
    S = radon(Coh, theta)/m # sinogram
    
    # classify
    s = S[m//2,:]
    min_idx,max_idx = np.argmin(s), np.argmax(s)
    # create circle
    x,y = np.sin(np.radians(2*theta)), np.cos(np.radians(2*theta))
    coh_circle = np.vstack((x,y,(s+.1)**2)).T
    kmeans = KMeans(n_clusters=2, \
                    init=np.array([coh_circle[min_idx,:], 
                                  coh_circle[max_idx,:]]),
                    n_init=1
                    ).fit(coh_circle)
    grouping = kmeans.labels_ #.astype(np.float)
    OUT = grouping==grouping[min_idx]     
       
    # construct filter
    fy = (np.arange(0,m)-(m/2)) /m
    fx = (np.arange(0,n)-(n/2)) /n
        
    Fx = np.flip(np.repeat(fx[np.newaxis,:],m,axis=0), axis=1)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    
    Theta = np.round(np.degrees(np.arctan2(Fx,Fy) % np.pi)/360 *m) *360 /m 
    W = np.isin(Theta, theta[~OUT])
    return W

# cross-spectral and frequency signal metrics for filtering
def thresh_masking(S, m=1e-4, s=10):
    """ mask significant intensities in spectrum
    
    Parameters
    ----------    
    S : np.array, size=(m,n), dtype=complex
        array with spectrum, i.e.: S = np.fft.fft2(I)
    m : float, default=1e-3
        cut-off intensity in respect to maximum
    s : integer, default=10
        kernel size of the median filter
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=bool
        frequency mask
    
    See Also
    --------
    tpss   
    
    Notes
    ----- 
    [1] Stone et al. "A fast direct Fourier-based algorithm for subpixel 
    registration of images." IEEE Transactions on geoscience and remote sensing 
    vol. 39(10) pp. 2235-2243, 2001.
    [2] Leprince, et.al. "Automatic and precise orthorectification, 
    coregistration, and subpixel correlation of satellite images, application 
    to ground deformation measurements", IEEE Transactions on Geoscience and 
    Remote Sensing vol. 45.6 pp. 1529-1558, 2007.    
    """    
    Sbar = np.abs(S)
    th = np.max(Sbar)*m
    
    # compose filter
    M = Sbar>th
    M = ndimage.median_filter(M, size=(s,s))
    return M

def local_coherence(Q, ds=1): 
    """ estimate the local coherence of a spectrum
    
    Parameters
    ----------    
    Q : np.array, size=(m,n), dtype=complex
        array with cross-spectrum
    ds : integer, default=10
        kernel radius to describe the neighborhood
        
    Returns
    -------
    M : np.array, size=(m,n), dtype=float
        vector coherence from no to ideal, i.e.: 0...1
    
    See Also
    --------
    thresh_masking   
      
    """
    diam = 2*ds+1
    C = np.zeros_like(Q)
    (isteps,jsteps) = np.meshgrid(np.linspace(-ds,+ds,2*ds+1, dtype=int), \
                          np.linspace(-ds,+ds,2*ds+1, dtype=int))
    IN = np.ones(diam**2, dtype=bool)  
    IN[diam**2//2] = False
    isteps,jsteps = isteps.flatten()[IN], jsteps.flatten()[IN]
    
    for idx, istep in enumerate(isteps):
        jstep = jsteps[idx]
        Q_step = np.roll(Q, (istep,jstep))
        # if the spectrum is normalized, then no division is needed
        C += Q*np.conj(Q_step)
    C = np.abs(C)/np.sum(IN)
    return C

# frequency/spectrum matching functions
def create_complex_DCT(I, Cc, Cs):
    Ccc, Css = Cc*I*Cc.T, Cs*I*Cs.T
    Csc, Ccs = Cs*I*Cc.T, Cc*I*Cs.T
    
    C_dct = Ccc-Css + 1j*(-(Ccs+Csc))
    return C_dct

def get_cosine_matrix(I,N=None):
    (L,_) = I.shape
    if N==None:
        N = np.copy(L)
    C = np.zeros((L,L))
    for k in range(L):
        for n in range(N):
            if k == 0:
                C[k,n] = np.sqrt(1/L)
            else:
                C[k,n] = np.sqrt(2/L)*np.cos((np.pi*k*(1/2+n))/L)
    return(C)

def get_sine_matrix(I,N=None):
    (L,_) = I.shape
    if N==None:
        # make a square matrix
        N = np.copy(L)
    C = np.zeros((L,L))
    for k in range(L):
        for n in range(N):
            if k == 0:
                C[k,n] = np.sqrt(1/L)
            else:
                C[k,n] = np.sqrt(2/L)*np.sin((np.pi*k*(1/2+n))/L)
    return(C)

def cosi_corr(I1, I2, beta1=.35, beta2=.50, m=1e-4):
    mt,nt = I1.shape[0], I1.shape[1] # dimensions of the template
    
    W1 = raised_cosine(np.zeros((mt,nt)), beta1) 
    W2 = raised_cosine(np.zeros((mt,nt)), beta2)
    
    if I1.size==I2.size: # if templates are same size, no refinement is done
        tries = [0]
    else:
        tries = [0, 1]
    
    di,dj, m0 = 0,0,np.array([0, 0])
    for trying in tries: # implement refinement step to have more overlap
        if I1.ndim==3: # multi-spectral frequency stacking
            bands = I1.shape[2]
            I1sub,I2sub = reposition_templates_from_center(I1,I2,di,dj)
            
            for i in range(bands): # loop through all bands
                I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
                S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
                
                if i == 0:                    
                    Q = (W1*S1)*np.conj((W2*S2))
                else:
                    Q_b = (W1*S1)*np.conj((W2*S2))
                    Q = (1/(i+1))*Q_b + (i/(i+1))*Q            
        else:    
            I1sub,I2sub = reposition_templates_from_center(I1,I2,di,dj)           
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            
            Q = (W1*S1)*np.conj((W2*S2))
            
        # transform back to spatial domain
        C = np.real(np.fft.fftshift(np.fft.ifft2(Q)))
        ddi, ddj,_,_ = get_integer_peak_location(C)
        m_int = np.round(np.array([ddi, ddj])).astype(int)
        if np.amax(abs(np.array([ddi, ddj])))<.5:
            break
        else:
            di,dj = m_int[0], m_int[1]
    
    m0[0] += di
    m0[1] += dj

    WS = thresh_masking(S1, m)    
    Qn = normalize_power_spectrum(Q)

    return Qn, WS, m0   

def cosine_corr(I1, I2): # wip
    """ match two imagery through discrete cosine transformation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    create_complex_DCT, sign_only_corr
    
    Notes
    -----    
    [1] Lie, et.al. "DCT-based phase correlation motion estimation", 
    IEEE international conference on image processing, vol. 1, 2004.
    """    
    # construct cosine and sine basis matrices
    Cc, Cs = get_cosine_matrix(I1), get_sine_matrix(I1)

    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            C1 = create_complex_DCT(I1bnd, Cc, Cs)
            C2 = create_complex_DCT(I2bnd, Cc, Cs)
            if i == 0:
                Q = C1*np.conj(C2)
            else:
                Q_b = (C1)*np.conj(C2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        C1 = create_complex_DCT(I1sub, Cc, Cs)
        C2 = create_complex_DCT(I2sub, Cc, Cs)
        Q = (C1)*np.conj(C2)
    return Q       

def masked_cosine_corr(I1, I2, M1, M2): # wip
    '''
    work in progress
    '''    
    M1, M2 = M1.astype(dtype=bool), M2.astype(dtype=bool)
    
    # construct cosine and sine basis matrices
    Cc, Cs = get_cosine_matrix(I1), get_sine_matrix(I1)

    # look at how many frequencies can be estimated with this data
    (m,n) = M1.shape
    X1 = np.ones((m,n), dtype=bool)    
    min_span = int(np.floor(np.sqrt(min(np.sum(M1), np.sum(M2)))))
    X1[min_span:,:] = False    
    X1[:,min_span:] = False
    
    y = (I1[M1].astype(dtype=float)/255)-.5
    
    # build matrix
    Ccc = np.kron(Cc,Cc)
    # shrink size    
    Ccc = Ccc[M1.flatten(),:] # remove rows, as these are missing
    Ccc = Ccc[:,X1.flatten()] # remove collumns, since these can't be estimated
    Icc = np.linalg.lstsq(Ccc, y, rcond=None)[0]
    Icc = np.reshape(Icc, (min_span, min_span))
        
    iCC = Ccc.T*y
    
    np.reshape(Ccc.T*y, (min_span, min_span))
    
    
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            C1 = create_complex_DCT(I1sub, Cc, Cs)
            C2 = create_complex_DCT(I2sub, Cc, Cs)
            if i == 0:
                Q = C1*np.conj(C2)
            else:
                Q_b = (C1)*np.conj(C2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        C1 = create_complex_DCT(I1sub, Cc, Cs)
        C2 = create_complex_DCT(I2sub, Cc, Cs)
        Q = (C1)*np.conj(C2)
    return Q 

def phase_only_corr(I1, I2):
    """ match two imagery through phase only correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    phase_corr, symmetric_phase_corr, amplitude_comp_corr   
    
    Notes
    ----- 
    [1] Horner & Gianino, "Phase-only matched filtering", Applied optics, 
    vol. 23(6) pp.812--816, 1984.
    [2] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
    ternary matched filters with increased signal-to-noise ratios for 
    colored noise", Optics letters, vol. 16(13) pp. 1025--1027, 1991.
    """  
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            W2 = np.divide(1, np.abs(I2bnd), 
                           out=np.zeros_like(I2bnd), where=np.abs(I2bnd)!=0)
            if i == 0:
                Q = (S1)*np.conj((W2*S2))
            else:
                Q_b = (S1)*np.conj((W2*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        
        W2 = np.divide(1, np.abs(I2sub), 
                       out=np.zeros_like(I2sub), where=np.abs(I2sub)!=0)
        
        Q = (S1)*np.conj((W2*S2))
    return Q

def sign_only_corr(I1, I2): # to do
    """ match two imagery through phase only correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    C : np.array, size=(m,n), real
        displacement surface
    
    See Also
    --------
    cosine_corr 
    
    Notes
    -----    
    [1] Ito & Kiya, "DCT sign-only correlation with application to image
        matching and the relationship with phase-only correlation", 
        IEEE international conference on acoustics, speech and signal 
        processing, vol. 1, 2007. 
    """ 
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            C1, C2 = np.sign(fft.dctn(I1bnd, 2)), np.sign(fft.dctn(I2bnd, 2))
            if i == 0:
                Q = C1*np.conj(C2)
            else:
                Q_b = (C1)*np.conj(C2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        C1, C2 = np.sign(fft.dctn(I1sub, 2)), np.sign(fft.dctn(I2sub, 2))
        Q = (C1)*np.conj(C2)
        C = fft.idctn(Q,2)
    return C    

def symmetric_phase_corr(I1, I2):
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            W2 = np.divided(1, np.sqrt(abs(I1sub))*np.sqrt(abs(I2sub)) )
            if i == 0:
                Q = (S1)*np.conj((W2*S2))
            else:
                Q_b = (S1)*np.conj((W2*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        
        W2 = np.divide(1, np.sqrt(abs(I1sub))*np.sqrt(abs(I2sub)) )
        
        Q = (S1)*np.conj((W2*S2))
    return Q

def amplitude_comp_corr(I1, I2, F_0=0.04):
    """ match two imagery through amplitude compensated phase correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
    F_0 : float, default=4e-2
        cut-off intensity in respect to maximum
    
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    Notes
    -----    
    [1] Mu et al. "Amplitude-compensated matched filtering", Applied optics, 
    vol. 27(16) pp. 3461-3463, 1988.
    """
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            s_0 = F_0 * np.amax(abs(S2))
            
            W = np.divide(1, abs(I2sub) )
            A = np.divide(s_0, abs(I2sub)**2)
            W[abs(S2)>s_0] = A
            if i == 0:
                Q = (S1)*np.conj((W*S2))
            else:
                Q_b = (S1)*np.conj((W*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        s_0 = F_0 * np.amax(abs(S2))
            
        W = np.divide(1, abs(I2sub) )
        A = np.divide(s_0, abs(I2sub)**2)
        W[abs(S2)>s_0] = A[abs(S2)>s_0]
        
        Q = (S1)*np.conj((W*S2))
    return Q

def robust_corr(I1, I2):
    """ match two imagery through fast robust correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    Notes
    -----    
    [1] Fitch et al. "Fast robust correlation", IEEE transactions on image 
    processing vol. 14(8) pp. 1063-1073, 2005.
    [2] Essannouni et al. "Adjustable SAD matching algorithm using frequency 
    domain" Journal of real-time image processing, vol.1 pp.257-265
    """
    I1sub,I2sub = make_templates_same_size(I1,I2)
    
    p_steps = 10**np.arange(0,1,.5)
    for idx, p in enumerate(p_steps):
        I1p = 1/p**(1/3) * np.exp(1j*(2*p -1)*I1sub)
        I2p = 1/p**(1/3) * np.exp(1j*(2*p -1)*I2sub)
    
        S1p, S2p = np.fft.fft2(I1p), np.fft.fft2(I2p)
        if idx==0:
            Q = (S1p)*np.conj(S2p)
        else:
            Q += (S1p)*np.conj(S2p)
    return Q

def orientation_corr(I1, I2):
    """ match two imagery through orientation correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    phase_corr, windrose_corr   
    
    Notes
    ----- 
    [1] Fitch et al. "Orientation correlation", Proceeding of the Britisch 
    machine vison conference, pp. 1--10, 2002.
    [2] Heid & Kääb. "Evaluation of existing image matching methods for 
    deriving glacier surface displacements globally from optical satellite 
    imagery", Remote sensing of environment, vol. 118 pp. 339-355, 2012.
    """     
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd)             
            S1,S2 = normalize_power_spectrum(S1),normalize_power_spectrum(S2)
            
            if i == 0:
                Q = (S1)*np.conj(S2)
            else:
                Q_b = (S1)*np.conj(S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        S1,S2 = normalize_power_spectrum(S1),normalize_power_spectrum(S2)

        Q = (S1)*np.conj(S2)
    return Q

def windrose_corr(I1, I2):
    """ match two imagery through windrose phase correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    orientation_corr, phase_only_corr   
    
    Notes
    -----    
    [1] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
    ternary matched filters with increased signal-to-noise ratios for 
    colored noise", Optics letters, vol. 16(13) pp. 1025--1027, 1991.
    """ 
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd)     
            if i == 0:
                Q = (S1)*np.conj(S2)
            else:
                Q_b = (S1)*np.conj(S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.sign(np.fft.fft2(I1sub)), np.sign(np.fft.fft2(I2sub))

        Q = (S1)*np.conj(S2)
    return Q

def phase_corr(I1, I2):
    """ match two imagery through phase correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    orientation_corr, cross_corr   
    
    Notes
    -----    
    [1] Kuglin & Hines. "The phase correlation image alignment method",
    proceedings of the IEEE international conference on cybernetics and 
    society, pp. 163-165, 1975.
    """
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            
            if i == 0:
                Q = (S1)*np.conj(S2)
                Q = np.divide(Q, abs(Q))
            else:
                Q_b = (S1)*np.conj(S2)
                Q_b = np.divide(Q_b, abs(Q))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        Q = (S1)*np.conj(S2)
        Q = np.divide(Q, abs(Q))
    return Q

def cross_corr(I1, I2):
    """ match two imagery through cross correlation in FFT
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    phase_corr  
    
    Notes
    ----- 
    [1] Heid & Kääb. "Evaluation of existing image matching methods for 
    deriving glacier surface displacements globally from optical satellite 
    imagery", Remote sensing of environment, vol. 118 pp. 339-355, 2012.
    """ 
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            
            if i == 0:
                Q = (S1)*np.conj(S2)
            else:
                Q_b = (S1)*np.conj(S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        Q = (S1)*np.conj(S2)
    return Q

def binary_orientation_corr(I1, I2):
    """ match two imagery through binary phase only correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n)
        cross-spectrum
    
    See Also
    --------
    orientation_corr, phase_only_corr   
    
    Notes
    -----    
    [1] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
    ternary matched filters with increased signal-to-noise ratios for 
    colored noise", Optics letters, vol. 16(13) pp. 1025--1027, 1991.
    """ 
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd)          
            W = np.sign(np.real(S2))
            
            if i == 0:
                Q = (S1)*np.conj(W*S2)
            else:
                Q_b = (S1)*np.conj(W*S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        W = np.sign(np.real(S2))

        Q = (S1)*np.conj(W*S2)
    return Q
    
def masked_corr(I1, I2, M1, M2):
    """ match two imagery through masked normalized cross-correlation in FFT
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
    M1 : np.array, size=(m,n)
        array with mask
    M2 : np.array, size=(m,n)
        array with mask        
    Returns
    -------
    NCC : np.array, size=(m,n)
        correlation surface
    
    Notes
    -----    
    [1] Padfield. "Masked object registration in the Fourier domain", 
    IEEE transactions on image processing, vol. 21(5) pp. 2706-2718, 2011.
    """
    I1sub,I2sub = make_templates_same_size(I1,I2)
    M1sub,M2sub = make_templates_same_size(M1,M2)

    I1f, I2f = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
    M1f, M2f = np.fft.fft2(M1sub), np.fft.fft2(M2sub)
    
    fF1F2 = np.fft.ifft2( I1f*np.conj(I2f) )
    fM1M2 = np.fft.ifft2( M1f*np.conj(M2f) )
    fM1F2 = np.fft.ifft2( M1f*np.conj(I2f) )
    fF1M2 = np.fft.ifft2( I1f*np.conj(M2f) )
    
    ff1M2 = np.fft.ifft2( np.fft.fft2(I1sub**2)*np.conj(M2f) )
    fM1f2 = np.fft.ifft2( M1f*np.fft.fft2( np.flipud(I2sub**2) ) )
    
    
    NCC_num = fF1F2 - \
        (np.divide(
            np.multiply( fF1M2, fM1F2 ), fM1M2 ))
    NCC_den = np.multiply( \
                          np.sqrt(ff1M2 - np.divide( fF1M2**2, fM1M2) ),
                          np.sqrt(fM1f2 - np.divide( fM1F2**2, fM1M2) ))
    NCC = np.divide(NCC_num, NCC_den) 
    return NCC
    
# binary transform functions
def affine_binairy_registration(B1, B2):
    # preparation
    pT = np.sum(B1) # Lebesgue integral
    pO = np.sum(B2)
    
    Jac = pO/pT # Jacobian
    
    x = np.linspace(0,B1.shape[1]-1,B1.shape[1])
    y = np.linspace(0,B1.shape[0]-1,B1.shape[0])
    X1, Y1 = np.meshgrid(x,y)
    del x, y
    
    # calculating moments of the template
    x11 = Jac* np.sum(X1    * B1)
    x12 = Jac* np.sum(X1**2 * B1)
    x13 = Jac* np.sum(X1**3 * B1)
    
    x21 = Jac* np.sum(Y1    * B1)
    x22 = Jac* np.sum(Y1**2 * B1)
    x23 = Jac* np.sum(Y1**3 * B1)
    del X1, Y1
    
    x = np.linspace(0,B2.shape[1]-1,B2.shape[1])
    y = np.linspace(0,B2.shape[0]-1,B2.shape[0])
    X2, Y2 = np.meshgrid(x,y)
    del x, y
    
    # calculating moments of the observation
    y1   = np.sum(X2       * B2)
    y12  = np.sum(X2**2    * B2)
    y13  = np.sum(X2**3    * B2)
    y12y2= np.sum(X2**2*Y2 * B2)
    y2   = np.sum(Y2       * B2)
    y22  = np.sum(Y2**2    * B2)
    y23  = np.sum(Y2**3    * B2)
    y1y22= np.sum(X2*Y2**2 * B2)
    y1y2 = np.sum(X2*Y2    * B2)
    del X2, Y2
    
    # estimation
    mu = pO
    
    def func1(x):
        q11, q12, q13 = x
        return [mu*q11 + y1*q12 + y2*q13 - x11,
                mu*q11**2 + y12*q12**2 + y22*q13**2 + 2*y1*q11*q12 + \
                    2*y2*q11*q13 + 2*y1y2*q12*q13 - x12,
                mu*q11**3 + y13*q12**3 + y23*q13**3 + 3*y1*q11**2*q12 + \
                    3*y2*q11**2*q13 + 3*y12*q12**2*q11 + 3*y12y2*q12**2*q13 + \
                        3*y22*q11*q13**2 + 3*y1y22*q12*q13**2 + \
                            6*y1y2*q11*q12*q13 - x13]
    
    Q11, Q12, Q13 = fsolve(func1, (1.0, 1.0, 1.0))
    # test for complex solutions, which should be excluded
    
    def func2(x):
        q21, q22, q23 = x
        return [mu*q21 + y1*q22 + y2*q23 - x21,
                mu*q21**2 + y12*q22**2 + y22*q23**2 + 2*y1*q21*q22 + \
                    2*y2*q21*q23 + 2*y1y2*q22*q23 - x22,
                mu*q21**3 + y13*q22**3 + y23*q23**3 + 3*y1*q21**2*q22 + \
                    3*y2*q21**2*q23 + 3*y12*q22**2*q21 + 3*y12y2*q22**2*q23 + \
                        3*y22*q21*q23**2 + 3*y1y22*q22*q23**2 + \
                            6*y1y2*q21*q22*q23 - x23]
    
    Q21, Q22, Q23 = fsolve(func2, (1.0, 1.0, 1.0))
    # test for complex solutions, which should be excluded
    
    Q = np.array([[Q12, Q13, Q11], [Q22, Q23, Q21]])            
    return Q

# boundary describtors
def get_relative_group_distances(x, K=5):
    for i in range(1,K+1):
        if i ==1:
            x_minus =  np.expand_dims(np.roll(x, +i), axis= 1)
            x_plus = np.expand_dims(np.roll(x, -i), axis= 1)
        else:
            x_new = np.expand_dims(np.roll(x, +i), axis=1)
            x_minus = np.concatenate((x_minus, x_new), axis=1)
            x_new = np.expand_dims(np.roll(x, -i), axis=1)
            x_plus = np.concatenate((x_plus, x_new), axis=1)
            del x_new
    dx_minus = x_minus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)
    dx_plus = x_plus - np.repeat(np.expand_dims(x, axis=1), K, axis=1)   
    return dx_minus, dx_plus

def get_relative_distances(x, x_id, K=5):
    # minus
    start_idx = x_id-K
    ending_idx = x_id
    ids = np.arange(start_idx,ending_idx)
    x_min = x[ids % len(x)]
    # plus
    start_idx = x_id
    ending_idx = x_id + K
    ids = np.arange(start_idx,ending_idx)
    x_plu = x[ids % len(x)]   

    dx_minus = x_min - np.repeat(x[x_id], K)  
    dx_plus = x_plu - np.repeat(x[x_id], K) 
    return dx_minus, dx_plus

def beam_angle_statistics(x, y, K=5, xy_id=None):
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
    
    if xy_id is None: # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K)   
        ax = 1            
    else: # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # dot product instead of argument
    C_minus = np.arctan2(dy_minus, dx_minus)
    C_plus = np.arctan2(dy_plus, dx_plus)
    
    C = C_minus-C_plus
    C_1, C_2 = np.mean(C, axis=ax), np.std(C, axis=ax) # estimate moments
    BAS = np.concatenate((np.expand_dims(C_1, axis=ax),
                          np.expand_dims(C_2, axis=ax)), axis=ax)
    return BAS

def cast_angle_neighbours(x, y, sun, K=5, xy_id=None):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''
    if xy_id is None: # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K)               
        ax = 1
    else: # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # rotate towards the sun
    CAN = np.concatenate((np.arctan2(sun[0]*dx_minus + sun[1]*dy_minus, 
                      -sun[1]*dx_minus + sun[0]*dy_minus), 
           np.arctan2(sun[0]*dx_plus + sun[1]*dy_plus, 
                      -sun[1]*dx_plus + sun[0]*dy_plus)), axis=ax)
    return CAN

def neighbouring_cast_distances(x, y, sun, K=5, xy_id=None):
    '''
    debug:
        x = np.random.randint(20, size=12)-10
        y = np.random.randint(20, size=12)-10
        sun = sun/np.sqrt(np.sum(np.square(sun)))
    '''    
    if xy_id is None: # make descriptors for all coordinate
        dx_minus, dx_plus = get_relative_group_distances(x, K)
        dy_minus, dy_plus = get_relative_group_distances(y, K) 
        ax = 1              
    else: # make descriptor for single coordinate
        dx_minus, dx_plus = get_relative_distances(x, xy_id, K)
        dy_minus, dy_plus = get_relative_distances(y, xy_id, K)
        ax = 0
    # rotate towards the sun and take only one axes
    CD = np.concatenate((sun[0]*dx_minus + sun[1]*dy_minus, 
          sun[0]*dx_plus + sun[1]*dy_plus), axis=ax)
    return CD    

# supporting functions
def make_templates_same_size(I1,I2):
    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space
    
    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt
    assert I1.ndim==I2.ndim # should be the same dimension

    md, nd = (ms-mt)//2, (ns-nt)//2
    if md==0 | nd==0: # I2[+0:-0, ... does not seem to work
        I2sub = I2
    else:
        if I1.ndim==3:
            I2sub = I2[+md:-md, +nd:-nd, :]
        else:        
            I2sub = I2[+md:-md, +nd:-nd]        
    return I1, I2sub

def test_bounds_reposition(d, temp_size, search_size):
    """
    See Also
    --------
    reposition_templates_from_center
    """    
    space_bound = (search_size-temp_size) // 2
    if abs(d) > space_bound:
        warnings.warn("part of the template will be out of the image" +
                      "with this displacement estimate")
        reposition_dir = np.sign(d)
        d = reposition_dir * np.minimum(abs(d), space_bound)
    return d

def reposition_templates_from_center(I1,I2,di,dj):
    mt,nt = I1.shape[0],I1.shape[1] # dimenstion of the template
    ms,ns = I2.shape[0],I2.shape[1] # dimension of the search space
    
    di,dj = int(di),int(dj)
    di,dj = test_bounds_reposition(di,mt,ms), test_bounds_reposition(dj,nt,ns)
    
    assert ms>=mt # search domain should be of equal size or bigger
    assert ns>=nt
    assert I1.ndim==I2.ndim # should be the same dimension
    
    mc,nc = ms//2, ns//2 # center location    

    if I1.ndim==3:
        I2sub = I2[mc-(mt//2)-di : mc+(mt//2)-di, \
                   nc-(nt//2)-dj : nc+(nt//2)-dj, :]
    else: 
        I2sub = I2[mc-(mt//2)-di : mc+(mt//2)-di, \
                   nc-(nt//2)-dj : nc+(nt//2)-dj]       
    return I1, I2sub

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