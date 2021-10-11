# general libraries
import numpy as np

from .matching_tools import \
    get_integer_peak_location, reposition_templates_from_center, \
    make_templates_same_size
from .matching_tools_frequency_filters import \
    raised_cosine, thresh_masking, normalize_power_spectrum, gaussian_mask

# general frequency functions
def create_complex_DCT(I, Cc, Cs):  #wip
    Ccc, Css = Cc*I*Cc.T, Cs*I*Cs.T
    Csc, Ccs = Cs*I*Cc.T, Cc*I*Cs.T
    
    C_dct = Ccc-Css + 1j*(-(Ccs+Csc))
    return C_dct

def get_cosine_matrix(I,N=None): #wip
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

def get_sine_matrix(I,N=None): #wip
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

def upsample_dft(Q, up_m=0, up_n=0, upsampling=1, \
                 i_offset=0, j_offset=0):
    (m,n) = Q.shape
    if up_m==0:
        up_m = m.copy()
    if up_n==0:
        up_n = n.copy()
    
    kernel_collumn = np.exp((1j*2*np.pi/(n*upsampling)) *\
                            ( np.fft.fftshift(np.arange(n) - \
                                              (n//2))[:,np.newaxis] )*\
                            ( np.arange(up_n) - j_offset ))
    kernel_row = np.exp((1j*2*np.pi/(m*upsampling)) *\
                        ( np.arange(up_m)[:,np.newaxis] - i_offset )*\
                        ( np.fft.fftshift(np.arange(m) - (m//2)) ))    
    Q_up = np.matmul(kernel_row, np.matmul(Q,kernel_collumn))
    return Q_up

def pad_dft(Q, m_new, n_new):
    (m,n) = Q.shape
    
    Q_ij = np.fft.fftshift(Q) # in normal configuration
    center_old = np.array([m//2, n//2])
    
    Q_new = np.zeros((m_new, n_new), dtype=np.complex64)
    center_new =  np.array([m_new//2, n_new//2])
    center_offset = center_new - center_old
    
    # fill the old data in the new array
    Q_new[np.maximum(center_offset[0], 0):np.minimum(center_offset[0]+m, m_new),\
          np.maximum(center_offset[1], 0):np.minimum(center_offset[1]+n, n_new)]\
        = \
        Q_ij[np.maximum(-center_offset[0], 0):\
             np.minimum(-center_offset[0]+m_new, m),\
             np.maximum(-center_offset[1], 0):\
             np.minimum(-center_offset[1]+n_new, n)]
            
    Q_new = (np.fft.fftshift(Q_new)*m_new*n_new)/(m*n) # scaling
    return Q_new

# frequency/spectrum matching functions
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
    Q : np.array, size=(m,n), dtype=complex
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
        
        from scipy import fftpack
        C1 = (fftpack.dct(fftpack.dct(I1sub, axis=0), axis=1) - \
            fftpack.dst(fftpack.dst(I1sub, axis=0), axis=1)) - 1j*(\
                  fftpack.dct(fftpack.dst(I1sub, axis=0), axis=1) +\
                  fftpack.dst(fftpack.dct(I1sub, axis=0), axis=1))

        C2 = (fftpack.dct(fftpack.dct(I2sub, axis=0), axis=1) - \
            fftpack.dst(fftpack.dst(I2sub, axis=0), axis=1)) - 1j*(\
                  fftpack.dct(fftpack.dst(I2sub, axis=0), axis=1) +\
                  fftpack.dst(fftpack.dct(I2sub, axis=0), axis=1))
        
        # C1 = create_complex_DCT(I1sub, Cc, Cs)
        # C2 = create_complex_DCT(I2sub, Cc, Cs)
        Q = (C1)*np.conj(C2)
        Q = Q/np.abs(Q)
        C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    return C       

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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    phase_corr, symmetric_phase_corr, amplitude_comp_corr   

    Example
    -------
    >>> import numpy as np
    >>> from ..generic.test_tools import create_sample_image_pair
    >>> from .matching_tools import get_integer_peak_location
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_only_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    ----- 
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1], \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{W} = 1 / \mathbf{S}_2
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 [\mathbf{W}\mathbf{S}_2]^{\star}
    
    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a complex conjugate operation

    .. [1] Horner & Gianino, "Phase-only matched filtering", Applied optics, 
       vol. 23(6) pp.812--816, 1984.
    .. [2] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
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
        
        C1,C2 = fft.dctn(I1sub, 2), fft.dctn(I2sub, 2)
#        C1,C2 = np.multiply(C1,1/C1), np.multiply(C2,1/C2) 
        C1,C2 = np.sign(C1), np.sign(C2)
        Q = (C1)*np.conj(C2)
        C = np.real(fft.idctn(Q,2))
        
#        iC1 = fft.idctn(C1,2)
#        import matplotlib.pyplot as plt
#        plt.imshow(iC1), plt.show()
    return C    

def symmetric_phase_corr(I1, I2):
    """ match two imagery through symmetric phase only correlation (SPOF)
    also known as Smoothed Coherence Transform (SCOT)
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
    
    Returns
    -------
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum

    Example
    -------
    >>> import numpy as np
    >>> from ..generic.test_tools import create_sample_image_pair
    >>> from .matching_tools import get_integer_peak_location
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = symmetric_phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1], \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{W} = 1 / \sqrt{||\mathbf{S}_1||||\mathbf{S}_2||}
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 [\mathbf{W}\mathbf{S}_2]^{\star}
    
    where :math:`\mathcal{F}` denotes the Fourier transform and :math:`\star` a complex conjugate operation

    .. [1] Nikias & Petropoulou. "Higher order spectral analysis: a nonlinear 
       signal processing framework", Prentice hall. pp.313-322, 1993.
    .. [2] Wernet. "Symmetric phase only filtering: a new paradigm for DPIV 
       data processing", Measurement science and technology, vol.16 pp.601-618, 
       2005.
    """
    if I1.ndim==3: # multi-spectral frequency stacking
        bands = I1.shape[2]
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        for i in range(bands): # loop through all bands
            I1bnd, I2bnd = I1sub[:,:,i], I2sub[:,:,i]
            
            S1, S2 = np.fft.fft2(I1bnd), np.fft.fft2(I2bnd) 
            
            S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
            W2 = np.divided(1, np.sqrt(abs(S1sub))*np.sqrt(abs(S2sub)) )
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = amplitude_comp_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Mu et al. "Amplitude-compensated matched filtering", Applied optics, 
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = robust_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Fitch et al. "Fast robust correlation", IEEE transactions on image 
       processing vol. 14(8) pp. 1063-1073, 2005.
    .. [2] Essannouni et al. "Adjustable SAD matching algorithm using frequency 
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    phase_corr, windrose_corr   

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = orientation_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    ----- 
    .. [1] Fitch et al. "Orientation correlation", Proceeding of the Britisch 
       machine vison conference, pp. 1--10, 2002.
    .. [2] Heid & Kääb. "Evaluation of existing image matching methods for 
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    orientation_corr, phase_only_corr   

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = windrose_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    orientation_corr, cross_corr   

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Kuglin & Hines. "The phase correlation image alignment method",
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

def gaussian_transformed_phase_corr(I1, I2):
    """ match two imagery through Gaussian transformed phase correlation
    
    Parameters
    ----------    
    I1 : np.array, size=(m,n)
        array with intensities
    I2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    phase_corr 

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = gaussian_transformed_phase_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Eckstein et al. "Phase correlation processing for DPIV 
       measurements", Experiments in fluids, vol.45 pp.485-500, 2008.
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
                M = gaussian_mask(S1)
                Q = np.multiply(M, Q)
            else:
                Q_b = (S1)*np.conj(S2)
                Q_b = np.divide(Q_b, abs(Q))
                Q_b = np.multiply(M, Q_b)
                Q = (1/(i+1))*Q_b + (i/(i+1))*Q
        
    else:    
        I1sub,I2sub = make_templates_same_size(I1,I2)
        
        S1, S2 = np.fft.fft2(I1sub), np.fft.fft2(I2sub)
        Q = (S1)*np.conj(S2)
        Q = np.divide(Q, abs(Q))
        
        M = gaussian_mask(Q)
        Q = np.multiply(M, Q)
    return Q
        
def upsampled_cross_corr(S1, S2, upsampling=2):
    """ apply cros correlation, and upsample the correlation peak
    
    Parameters
    ----------    
    S1 : np.array, size=(m,n)
        array with intensities
    S2 : np.array, size=(m,n)
        array with intensities
        
    Returns
    -------
    di,dj : float     
        sub-pixel displacement
    
    See Also
    --------
    pad_dft, upsample_dft 

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> di,dj = upsampled_cross_corr(im1, im2)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    ----- 
    .. [1] Guizar-Sicairo, et al. "Efficient subpixel image registration 
       algorithms", Applied optics, vol. 33 pp.156--158, 2008.
    """  
    (m,n) = S1.shape
    S1,S2 = pad_dft(S1, 2*m, 2*n), pad_dft(S2, 2*m, 2*n)
#    Q = S1*conj(S2)
    Q = normalize_power_spectrum(S1)*np.conj(normalize_power_spectrum(S2))
#    Q = normalize_power_spectrum(Q)
    C = np.real(np.fft.ifft2(Q))
    
    ij = np.unravel_index(np.argmax(C), C.shape, order='F')
    di, dj = ij[::-1]

    # transform to shifted fourier coordinate frame (being twice as big)
    i_F = np.fft.fftshift(np.arange(-np.fix(m),m))    
    j_F = np.fft.fftshift(np.arange(-np.fix(n),n))    

    i_offset, j_offset = i_F[di]/2, j_F[dj]/2
    
    if upsampling >2:
        i_shift = 1 + np.round(i_offset*upsampling)/upsampling
        j_shift = 1 + np.round(j_offset*upsampling)/upsampling
        F_shift = np.fix(np.ceil(1.5*upsampling)/2)
        
        CC = np.conj(upsample_dft(Q,\
                                  up_m=np.ceil(upsampling*1.5),\
                                  up_n=np.ceil(upsampling*1.5),\
                                  upsampling=upsampling,\
                                  i_offset=F_shift-(i_shift*upsampling),\
                                  j_offset=F_shift-(j_shift*upsampling)))
        ij = np.unravel_index(np.argmax(CC), CC.shape, order='F')
        ddi, ddj = ij[::-1]
        ddi -= (F_shift )
        ddj -= (F_shift )
        
        i_offset += ddi/upsampling
        j_offset += ddj/upsampling
    return i_offset,j_offset

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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    phase_corr  

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = cross_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    ----- 
    .. [1] Heid & Kääb. "Evaluation of existing image matching methods for 
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
    Q : np.array, size=(m,n), dtype=complex
        cross-spectrum
    
    See Also
    --------
    orientation_corr, phase_only_corr   

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> Q = binary_orientation_corr(im1, im2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Kumar & Juday, "Design of phase-only, binary phase-only, and complex 
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

    Example
    -------
    >>> import numpy as np
    >>> from .matching_tools import get_integer_peak_location
    >>> from ..generic.test_tools import create_sample_image_pair
    
    >>> im1,im2,ti,tj,_ = create_sample_image_pair(d=2**4, max_range=1)
    >>> msk1,msk2 = np.ones_like(im1), np.ones_like(im2)
    >>> Q = masked_corr(im1, im2, msk1, msk2)
    >>> C = np.fft.ifft2(Q)
    >>> di,dj,_,_ = get_integer_peak_location(C)
    
    >>> assert(np.isclose(ti, di, atol=1))
    >>> assert(np.isclose(ti, di, atol=1))
    
    Notes
    -----    
    .. [1] Padfield. "Masked object registration in the Fourier domain", 
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