import numpy as np

def matting_laplacian(I, Label, Mask=None, win_rad=1, eps=-10**7):
    """
    see Levin et al. 2006
    A closed form solution to natural image matting    
    """
    
    dI = np.arange(start=-win_rad,stop=+win_rad+1)
    dI,dJ = np.meshgrid(dI,dI)
    
    dI = dI.ravel()
    dJ = dJ.ravel()
    N = dI.size
    for i in range(dI.size):
        if i == 0:
            Istack = np.expand_dims( np.roll(I, 
                                              [dI[i], dJ[i]], axis=(0, 1)), 
                                    axis=3)
        else:
            Istack = np.concatenate((Istack, 
                                    np.expand_dims(np.roll(I, 
                                        [dI[i], dJ[i]], axis=(0, 1)),
                                                    axis=3)), axis=3)
                                    
    muI = np.mean(Istack, axis=3)
    dI = Istack - np.repeat(np.expand_dims(muI, axis=3), 
                            N, axis=3)
    del Istack
    aap = np.einsum('ijkl,qrs->km',dI,dI) /(N - 1)
    aap = np.einsum('ijkl,ijkn->ijk',dI,dI) /(N - 1)
    
    win_siz = 2*win_rad+1
    
    
    
    
    
    mean_kernel = np.ones((win_siz, win_siz))/win_siz**2
    
    muI = I.copy()
    varI = I.copy()
    for i in range(I.shape[2]):
        muI[:,:,i] = ndimage.convolve(I[:,:,i], mean_kernel, mode='reflect')
        dI = (I[:,:,i] - muI[:,:,i])**2
        varI[:,:,i] = ndimage.convolve(dI, mean_kernel, mode='reflect')
        del dI
    
    # local mean colors
    L = 0
    return L
