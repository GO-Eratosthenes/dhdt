import numpy as np
from scipy import ndimage 

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

def get_top(Q, ds=1):

    (subJ,subI) = np.meshgrid(np.linspace(-ds,+ds, 2*ds+1), np.linspace(-ds,+ds, 2*ds+1))    
    #subI = np.flipud(subI)
    #subJ = np.fliplr(subJ)
    subI = subI.ravel()
    subJ = subJ.ravel()
    # transform back to spatial domain
    C = np.real(np.fft.fftshift(np.fft.ifft2(Q)))
    
    # find highest score
    max_corr = np.amax(C)
    snr = max_corr/np.mean(C)
    ij = np.unravel_index(np.argmax(C), C.shape)
    x, y = ij[::-1]
    
    # estimate sub-pixel top
    idx_mid = np.int(np.floor((2.*ds+1)**2/2))
    
    Csub = C[y-ds:y+ds+1,x-ds:x+ds+1].ravel()
    Csub = Csub - np.mean(np.hstack((Csub[0:idx_mid],Csub[idx_mid+1:])))
    #Csub = Csub - np.min(Csub)
    
    IN = Csub>0
    
    m0 = np.array([ np.divide(np.sum(subI[IN]*Csub[IN]), np.sum(Csub[IN])) , 
                   np.divide(np.sum(subJ[IN]*Csub[IN]), np.sum(Csub[IN]))])
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

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def perdecomp(u):
    '''
    Calculted the periodic and smooth components of an image.
    based upon Moisan
    '''
    u = u.astype(float)
    
    (m, n) = u.shape
    v = np.zeros((m, n), dtype=float)
    
    v[+0,:] = +u[0,:] -u[-1,:]
    v[-1,:] = -v[0,:]
    
    v[:,+0] = v[:,+0] +u[:,+0] -u[:,-1]
    v[:,-1] = v[:,-1] -u[:,+0] +u[:,-1]

    fy = np.cos( 2*np.pi*( np.arange(0,m) )/m )
    fx = np.cos( 2*np.pi*( np.arange(0,n) )/n )

    Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
    Fy = np.repeat(fy[:,np.newaxis],n,axis=1)
    Fx[0,0] = 0
    
    s = np.real( np.fft.ifft2( np.fft.fft2(v) *.5/ (2-Fx-Fy)))
    p = u-s
    return (p, s)    
    

# main testing
from PIL import Image

img = Image.open('lena.jpg')
I = np.array(img)
I1 = np.double(I[:,:,0])/256

x = np.linspace(0,I1.shape[1]-1,I1.shape[1])
y = np.linspace(0,I1.shape[0]-1,I1.shape[0])
X, Y = np.meshgrid(x,y)

d = (+10.15,-5.25) # artificial displacement
ts = 2**7

# bi-linear interpolation
I2 = bilinear_interpolate(I1, X+d[0], Y+d[1])


I1sub = I1[ts:-ts,ts:-ts]
I2sub = I2[ts:-ts,ts:-ts]

#(P1sub, S1sub) = perdecomp(I1sub)


S1 = np.fft.fft2(I1sub)
S2 = np.fft.fft2(I2sub)

# hoge's approach
thres = 0.001 # 
kern = 10 # median filtering of spectral mask
rad = 0.10
(m,n) = I1sub.shape

S1 = np.fft.fftshift(np.fft.fft2(I1sub))
S2 = np.fft.fftshift(np.fft.fft2(I2sub))

# normalize spectrum
S12 = S1*np.conj(S2)
S = np.abs( S12)
S[S==0] = 1

Qn = np.divide(S12,S)


# [ mask ] = adapMasking( S1 );
# filtering through magnitude
M = np.zeros((m,n))
t = np.amax(abs(S1))*thres
M[abs(S1)>t] = 1
M = ndimage.median_filter(M, footprint=np.ones((kern,kern)))

# def svds(A, n_elements=1)
n_elements = 1
u,s,v = np.linalg.svd(M*Qn) # singular-value decomposition
sig = np.zeros((m,n))
sig[:m,:m] = np.diag(s)
sig = sig[:,:n_elements]# select first element only
# v = v[:n_elements,:]
# reconstruct
# b = u.dot(sig.dot(v))
t_m = v.dot(sig)
t_n = u.dot(sig)# transform

idx_sub = np.arange(np.ceil((0.5-rad)*len(t)), \
                    np.ceil((0.5+rad)*len(t))+1).astype(int)
y_ang = np.unwrap(np.angle(t_n[idx_sub]))
A = np.vstack([np.transpose(idx_sub-1), np.ones((len(idx_sub)))]).T
dx = np.linalg.lstsq(A, y_ang)

y_ang = np.unwrap(np.angle(t_m[idx_sub]))
dy = np.linalg.lstsq(A, y_ang)



W1 = raised_cosine(I1sub, beta=0.35)
W2 = raised_cosine(I2sub, beta=0.50)

Q = (W1*S1)*np.conj((W2*S2))

(P1sub,_) = perdecomp(I1sub)
(P2sub,_) = perdecomp(I2sub)
S1 = np.fft.fft2(P1sub)
S2 = np.fft.fft2(P2sub)
Q = S1*np.conj(S2)


Qa = np.fft.fftshift(np.angle(Q)) # from -pi() to +pi()

from skimage.transform import hough_line, hough_line_peaks

zero_cross = np.logical_or(np.gradient(np.sign(Qa), edge_order=1, axis=0)!=0, \
                           np.gradient(np.sign(Qa), edge_order=1, axis=1)!=0)

tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
h, angles, dists = hough_line(zero_cross, theta=tested_angles)

(M,N) = Q.shape

score, theta, rho = hough_line_peaks(h, angles, dists)    

A = np.transpose(np.array([rho/np.cos(theta), rho/np.cos(theta)]))


(m0, SNR) = get_top(Q)

WS = thresh_masking(S1, m=1/1e4)
# Qn = np.divide(Q,abs(Q)) <- gives errors....?
Qn = 1j*np.sin(np.angle(Q))
Qn += np.cos(np.angle(Q))
Qn[Q==0] = 0

Qa = np.angle(Qn)

(m, snr) = tpss(Qn, WS, m0, p=1e-6)


