# general libraries
import numpy as np

# image processing libraries
from scipy import ndimage
from scipy.optimize import fsolve
from skimage.feature import match_template

from eratosthenes.generic.filtering_statistical import make_2D_Gaussian

# spatial sub-pixel allignment functions
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
    assert sampleI.shape == sampleJ.shape
    Ugrd = np.zeros_like(sampleI, dtype="float")
    Vgrd = np.zeros_like(sampleI, dtype="float")

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

# spatial pattern matching functions
def normalized_cross_corr(I1, I2):  # processing
    """
    Simple normalized cross correlation
    input:   I1             array (n x m)     template with intensities
             I2             array (n x m)     search space with intensities
    output:  x              integer           location of maximum
             y              integer           location of maximum
    """
    result = match_template(I2, I1)
    return result

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

def sum_sq_diff(I1, I2):
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

# sub-pixel localization
def get_top_weighted(C, ds=1):
    (subJ,subI) = np.meshgrid(np.linspace(-ds,+ds, 2*ds+1), np.linspace(-ds,+ds, 2*ds+1))    
    subI = subI.ravel()
    subJ = subJ.ravel()
    
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    # estimate sub-pixel top
    idx_mid = np.int(np.floor((2.*ds+1)**2/2))
    
    Csub = C[y-ds:y+ds+1,x-ds:x+ds+1].ravel()
    Csub = Csub - np.mean(np.hstack((Csub[0:idx_mid],Csub[idx_mid+1:])))
    
    IN = Csub>0
    
    m0 = np.array([ np.divide(np.sum(subI[IN]*Csub[IN]), np.sum(Csub[IN])) , 
                   np.divide(np.sum(subJ[IN]*Csub[IN]), np.sum(Csub[IN]))])
    return (m0, snr)

def get_top_gaussian(C):
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    # estimate sub-pixel along each axis   
    dx = (np.log(C[x+1,y]) - np.log(C[x-1,y])) / \
        2*( (2*np.log(C[x,y])) -np.log(C[x-1,y]) -np.log(C[x+1,y]))
    dy = (np.log(C[x,y+1]) - np.log(C[x,y-1])) / \
        2*( (2*np.log(C[x,y])) -np.log(C[x,y-1]) -np.log(C[x,y+1]))       
    return (np.array([ dx, dy]), snr)

def get_top_parabolic(C):
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    # estimate sub-pixel along each axis   
    dx = (C[x+1,y] - C[x-1,y]) / 2*( (2*C[x,y]) -C[x-1,y] -C[x+1,y])
    dy = (C[x,y+1] - C[x,y-1]) / 2*( (2*C[x,y]) -C[x,y-1] -C[x,y+1])       
    return (np.array([ dx, dy]), snr)

def get_top_esinc(C, ds=1):
    
    # exponential esinc function
    # see Argyriou & Vlachos 2006 BMVC
    
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    # estimate sub-pixel per axis
    Cx = C[y,x-ds:x+ds+1].ravel()
    def funcX(x):
        a, b, c = x
        return [(Cx[0] - a*np.exp(-(b*(-1-c))**2)* \
                 ( np.sin(np.pi*(-1-c))/ np.pi*(-1-c)) )**2,
                (Cx[1] - a*np.exp(-(b*(+0-c))**2)* \
                 ( np.sin(np.pi*(+0-c))/ np.pi*(+0-c)) )**2,
                (Cx[2] - a*np.exp(-(b*(+1-c))**2)* \
                 ( np.sin(np.pi*(+1-c))/ np.pi*(+1-c)) )**2]
    
    xA, xB, xC = fsolve(funcX, (1.0, 1.0, 0.1))    
    
    Cy = C[y-ds:y+ds+1,x].ravel()
    def funcY(x):
        a, b, c = x
        return [(Cy[0] - a*np.exp(-(b*(-1-c))**2)* \
                 ( np.sin(np.pi*(-1-c))/ np.pi*(-1-c)) )**2,
                (Cy[1] - a*np.exp(-(b*(+0-c))**2)* \
                 ( np.sin(np.pi*(+0-c))/ np.pi*(+0-c)) )**2,
                (Cy[2] - a*np.exp(-(b*(+1-c))**2)* \
                 ( np.sin(np.pi*(+1-c))/ np.pi*(+1-c)) )**2]
    
    yA, yB, yC = fsolve(funcY, (1.0, 1.0, 0.1)) 
    
    m0 = np.array([ xC, yC])
    return (m0, snr)

def tpss(Q, W, m0, p=1e-4, l=4, j=5, n=3):
    '''
    TPSS two point step size for phase correlation minimization
    input:   Q        array (n x m)     cross spectrum
             m0       array (1 x 2)     initial displacement  
             p        float             closing error threshold
             l        integer           number of refinements in iteration
             j        integer           number of sub routines during an estimation
             n        integer           mask convergence factor
    output:  m        array (1 x 2)     sub-pixel displacement
             snr      float             signal-to-noise
    '''
    
    (m, n) = Q.shape
    fy = 2*np.pi*(np.arange(0,m)-(m/2)) /m
    fx = 2*np.pi*(np.arange(0,n)-(n/2)) /n
        
    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx = np.fft.fftshift(Fx)
    Fy = np.fft.fftshift(Fy)

    # initialize
    m = m0;
    W_0 = W;
    
    m_min = m0 + np.array([+.1, +.1])
    C_min = np.random.random(10) + np.random.random(10) * 1j
    
    C_min = 1j*np.sin(Fx*m_min[0] + Fy*m_min[1])
    C_min += np.cos(Fx*m_min[0] + Fy*m_min[1])
    
    QC_min = Q-C_min # np.abs(Q-C_min)
    dXY_min = np.multiply(2*W_0, (QC_min * np.conjugate(QC_min)) )
    
    g_min = np.real(np.array([np.nansum(Fx*dXY_min), np.nansum(Fy*dXY_min)]))
    
    print(m)
    for i in range(l):
        k = 1
        while True:
            # main body of iteration
            C = 1j*np.sin(Fx*m[0] + Fy*m[1])
            C += np.cos(Fx*m[0] + Fy*m[1])
            QC = Q-C # np.abs(Q-C)
            dXY = 2*W*(QC*np.conjugate(QC))
            g = np.real(np.array([np.nansum(np.multiply(Fx,dXY)), \
                                  np.nansum(np.multiply(Fy,dXY))]))
            
            # difference
            dm = m - m_min
            # if dm.any()<np.finfo(float).eps:
            #     break
            dg = g - g_min
            alpha = (np.transpose(dm)*dg)/np.sum(dg**2)
            
            if np.all(np.abs(m - m_min)<=p):
                break
            if k>=j:
                break
            
            # update
            m_min = np.copy(m)
            g_min = np.copy(g)
            dXY_min = np.copy(dXY)
            m -= alpha*dg
            print(m)
            k += 1
            
        # optimize weighting matrix
        phi = np.abs(QC*np.conjugate(QC))/2
        W = W*(1-(dXY/8))**n
    snr = 1 - (np.sum(phi)/(4*np.sum(W)))
    return (m, snr)

# frequency pre-processing
def perdecomp(img):
    '''
    Calculted the periodic and smooth components of an image.
    based upon Moisan
    '''
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

# frequency matching functions
def raised_cosine(I, beta=0.35):
    '''
    input:   I       array (n x m)     image template
             beta    float             roll-off factor
    output:  W       array (n x m)     weighting mask
    '''
    
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

def thresh_masking(S, m=1/1e3, s=10):
    '''
    input:   S       array (n x m)     spectrum
                                       i.e.: S = np.fft.fft2(I)
             m       float             cut-off
             s       integer           kernel size of the median filter
    output:  M       array (n x m)     mask
    '''
    
    Sbar = np.abs(S)
    th = np.max(Sbar)*m
    
    # compose filter
    M = Sbar>th
    M = ndimage.median_filter(M, size=(s,s))
    return M

def cosi_corr(I1, I2, beta1=.35, beta2=.50, m=1e-4, p=1e-4):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            
            if i == 0:
                W1, W2 = raised_cosine(I1sub, beta1), raised_cosine(I1sub, beta2)
                Q = (W1*S1)*np.conj((W2*S2))
            else:
                Q_b = (W1*S1)*np.conj((W2*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        
        W1, W2 = raised_cosine(I1sub, beta1), raised_cosine(I1sub, beta2)
        
        Q = (W1*S1)*np.conj((W2*S2))

    # transform back to spatial domain
    C = np.real(np.fft.fftshift(np.fft.ifft2(Q)))
    (m0, SNR) = get_top_weighted(C)
    m_int = np.round(m0).astype(int)
    
    # resample if not within one-pixel displacement
    if np.amax(abs(m0))>.5:
        md_p, md_m = md+m_int[0], md-m_int[0]
        nd_p, nd_m = nd+m_int[1], nd-m_int[1]
        # move image template
        if I1.ndim==3:
            
            for i in range(bt): # loop through all bands
                I1sub = I1[:,:,i]
                I2sub = I2[md_p:-md_m, nd_p:-nd_m, i]
                
                S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
                
                if i == 0:
                    Q = (W1*S1)*np.conj((W2*S2))
                else:
                    Q_b = (W1*S1)*np.conj((W2*S2))
                    Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        else:
            I2sub = I2[md_p:-md_m, nd_p:-nd_m]
            S2 = np.fft.fft2(I2sub)
            Q = (W1*S1)*np.conj((W2*S2))
    
    WS = thresh_masking(S1, m)
    
    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    Qn[Q==0] = 0
    return Qn, WS, m0      
   # # sub-pixel localization
   # (m, snr) = tpss(Qn, WS, m0, p)
   # m_hat = m_int+m
   # return (m_hat, snr, SNR)  

def hoge(I1, I2, rad=0.1, th=0.001, ker=10):
    # following Hoge 2003
    # "A subspace identification extension to the phase correlation method"
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)   
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd, i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            S12 = S1*np.conj(S2)
            S = np.abs( S12)
            S[S==0] = 1
            
            if i == 0:
                Qn = np.divide(S12,S)
            else:
                Q_b = np.divide(S12,S)
                Qn = (1/(i+1))*Q_b + (i/(i+1))*Qn
        
    else:
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        (m,n) = I1sub.shape

        S1 = np.fft.fftshift(np.fft.fft2(I1sub))
        S2 = np.fft.fftshift(np.fft.fft2(I2sub))
        
        # normalize spectrum
        S12 = S1*np.conj(S2)
        S = np.abs( S12)
        S[S==0] = 1
        
        Qn = np.divide(S12,S)
        
    # filtering through magnitude
    M = thresh_masking(S1, m=th, s=ker)

    # decompose axis
    n_elements = 1
    u,s,v = np.linalg.svd(M*Qn) # singular-value decomposition
    sig = np.zeros((m,n))
    sig[:m,:m] = np.diag(s)
    sig = sig[:,:n_elements]# select first element only
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

def phase_only_corr(I1, I2):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            W2 = np.divided(1, abs(I2sub))
            if i == 0:
                Q = (S1)*np.conj((W2*S2))
            else:
                Q_b = (S1)*np.conj((W2*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        
        W2 = np.divided(1, abs(I2sub))
        
        Q = (S1)*np.conj((W2*S2))
    return Q

def symmetric_phase_corr(I1, I2):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            W2 = np.divided(1, np.sqrt(abs(I1sub))*np.sqrt(abs(I2sub)) )
            if i == 0:
                Q = (S1)*np.conj((W2*S2))
            else:
                Q_b = (S1)*np.conj((W2*S2))
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        
        W2 = np.divided(1, np.sqrt(abs(I1sub))*np.sqrt(abs(I2sub)) )
        
        Q = (S1)*np.conj((W2*S2))
    return Q

def amplitude_comp_corr(I1, I2, F_0=0.04):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
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
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        s_0 = F_0 * np.amax(abs(S2))
            
        W = np.divide(1, abs(I2sub) )
        A = np.divide(s_0, abs(I2sub)**2)
        W[abs(S2)>s_0] = A
        
        Q = (S1)*np.conj((W*S2))
    return Q

def orientation_corr(I1, I2):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            S1a, S2a = np.angle(S1), np.angle(S2)
            S1, S2 = 1j*np.sin(S1a), 1j*np.sin(S2a)
            S1 += np.cos(np.angle(S1a))
            S2 += np.cos(np.angle(S2a))
            
            if i == 0:
                Q = (S1)*np.conj(S2)
            else:
                Q_b = (S1)*np.conj(S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        S1a, S2a = np.angle(S1), np.angle(S2)
        S1, S2 = 1j*np.sin(S1a), 1j*np.sin(S2a)
        S1 += np.cos(np.angle(S1a))
        S2 += np.cos(np.angle(S2a))

        Q = (S1)*np.conj(S2)
    return Q

def binary_orientation_corr(I1, I2):
    if I1.ndim==3: # multi-spectral frequency stacking
        (mt,nt,bt) = I1.shape
        (ms,ns,bs) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        for i in range(bt): # loop through all bands
            I1sub = I1[:,:,i]
            I2sub = I2[md:-md, nd:-nd,i]
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)            
            W = np.sign(np.real(S2))
            
            if i == 0:
                Q = (S1)*np.conj(W*S2)
            else:
                Q_b = (S1)*np.conj(W*S2)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        (mt,nt) = I1.shape
        (ms,ns) = I2.shape
        md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
        I1sub = I1
        I2sub = I2[md:-md, nd:-nd]
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        W = np.sign(np.real(S2))

        Q = (S1)*np.conj(W*S2)
    return Q
    
def masked_corr(I1, I2, M1, M2): # padfield
    (mt,nt) = I1.shape
    (ms,ns,bs) = I2.shape
    md, nd = np.round((ms-mt)/2).astype(int), np.round((ns-nt)/2).astype(int)
        
    I1sub, M1sub = I1, M1
    I2sub, M2sub = I2[md:-md, nd:-nd], M2[md:-md, nd:-nd]

    I1f = np.fft.fft2(I1sub)
    I2f = np.fft.fft2(I2sub)
    M1f = np.fft.fft2(M1sub)
    M2f = np.fft.fft2(M2sub)
    
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

# supporting functions

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