# test different matching algorithms

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eratosthenes.generic.filtering_statistical import mad_filtering
from eratosthenes.generic.test_tools import create_sample_image_pair, \
    construct_phase_plane, signal_to_noise, test_phase_plane_localization, \
    test_normalize_power_spectrum, test_subpixel_localization, \
    construct_correlation_peak
from eratosthenes.processing.matching_tools import \
    get_integer_peak_location
from eratosthenes.processing.matching_tools_frequency_correlators import \
    phase_corr, phase_only_corr, robust_corr, \
    orientation_corr, windrose_corr, binary_orientation_corr, \
    cosi_corr, sign_only_corr, cosine_corr, cross_corr, \
    symmetric_phase_corr, amplitude_comp_corr, upsampled_cross_corr
from eratosthenes.processing.matching_tools_frequency_subpixel import \
    phase_svd, phase_pca, phase_tpss, phase_difference, phase_ransac, \
    phase_lsq, phase_radon, phase_hough, phase_newton
from eratosthenes.processing.matching_tools_frequency_filters import \
    normalize_power_spectrum, perdecomp, low_pass_circle, raised_cosine, \
    local_coherence, thresh_masking, adaptive_masking
from eratosthenes.processing.matching_tools_spatial_correlators import \
    normalized_cross_corr, cumulative_cross_corr, sum_sq_diff, \
    simple_optical_flow
from eratosthenes.processing.matching_tools_spatial_subpixel import \
    get_top_moment, get_top_gaussian, get_top_parabolic, \
    get_top_birchfield, get_top_2d_gaussian, get_top_paraboloid

temp_size = 2**6
im1,im2,di,dj,im1_big = create_sample_image_pair(d=temp_size, max_range=1)
print('{:.4f}'.format(di))
print('{:.4f}'.format(dj))


(di_hat,dj_hat,di_lamb,dj_lamb) = simple_optical_flow(im1, im2, temp_size, \
                                                      temp_size//2, \
                                                      temp_size//2, 
                                                      sigma=2)

S1, S2 = np.fft.fft2(im1), np.fft.fft2(im2) 
di_02,dj_02 = upsampled_cross_corr(S1,S2, upsampling=2) #wip
di_04,dj_04 = upsampled_cross_corr(S1,S2, upsampling=4)
di_08,dj_08 = upsampled_cross_corr(S1,S2, upsampling=8)
di_16,dj_16 = upsampled_cross_corr(S1,S2, upsampling=16)
di_32,dj_32 = upsampled_cross_corr(S1,S2, upsampling=32)

C_hat = cosine_corr(im1, im2)
di_hat,dj_hat,snr,corr = get_integer_peak_location(C_hat)


print('test phase cycle localizators')

im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**4, max_range=10)
Q_tilde = construct_phase_plane(im1, di, dj)

di_tilde, dj_tilde = phase_hough(Q_tilde, sample_fraction=1)

Q_hat = phase_only_corr(im1, im2)
Q_hat = robust_corr(im1, im2)
di_hat, dj_hat = phase_hough(Q_hat, sample_fraction=.01)

# load data
im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**7, max_range=10)

print('testing (spatial) correlators')
C_hat = normalized_cross_corr(im2, im1_big)

#C_hat = cumulative_cross_corr(im2, im1_big)    
#di_hat,dj_hat,snr,corr = get_integer_peak_location(C_hat)
#assert( np.isclose(np.round(di), di_hat) and \
#       np.isclose(np.round(dj), dj_hat) )
#ddi,ddj = di-di_hat, dj-dj_hat
#print('sum_sq_diff correlator is OK, maximum difference is '+\
#      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')
    
C_hat = -1*sum_sq_diff(im2, im1_big)    
di_hat,dj_hat,snr,corr = get_integer_peak_location(C_hat)
assert( np.isclose(np.round(di), di_hat) and \
       np.isclose(np.round(dj), dj_hat) )
ddi,ddj = di-di_hat, dj-dj_hat
print('sum_sq_diff correlator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')
    
di_hat,dj_hat,snr,corr = get_integer_peak_location(C_hat)
assert( np.isclose(np.round(di), di_hat) and \
       np.isclose(np.round(dj), dj_hat) )
ddi,ddj = di-di_hat, dj-dj_hat
print('normalized cross correlator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels') 
   
print('testing (spatial) subpixel estimators')    

di, dj = 2*np.random.random()-1, 2*np.random.random()-1
di, dj = 10*di, 10*dj
C_tilde = construct_correlation_peak(np.zeros((2**7,2**7)), di, dj)

# test test-function
di_tilde,dj_tilde,_,_ = get_integer_peak_location(C_tilde)
assert np.round(di)==di_tilde
assert np.round(di)==di_tilde

ddi,ddj,ii,jj = get_top_moment(C_tilde, ds=1)
assert np.isclose(di, ii+ddi, atol=2e-1)
assert np.isclose(dj, jj+ddj, atol=2e-1)
dddi,dddj = (di-(ii+ddi)), (dj-(jj+ddj))
print('moment locator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(dddi), abs(dddj))) + ' pixels') 

di, dj = 2*np.random.random()-1, 2*np.random.random()-1
di, dj = 10*di, 10*dj
C_tilde = construct_correlation_peak(np.zeros((2**7,2**7)), di, dj)
    
    
ddi,ddj,ii,jj = get_top_gaussian(C_tilde)
assert np.isclose(di, ii+ddi, atol=5e-1)
assert np.isclose(dj, jj+ddj, atol=5e-1)
dddi,dddj = (di-(ii+ddi)), (dj-(jj+ddj))
print('gaussian locator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(dddi), abs(dddj))) + ' pixels')    
    
ddi,ddj,ii,jj = get_top_parabolic(C_tilde)
assert np.isclose(di, ii+ddi, atol=5e-1)
assert np.isclose(dj, jj+ddj, atol=5e-1)
dddi,dddj = (di-(ii+ddi)), (dj-(jj+ddj))
print('parabolic locator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(dddi), abs(dddj))) + ' pixels')    

di, dj = 2*np.random.random()-1, 2*np.random.random()-1
di, dj = 10*di, 10*dj
C_tilde = construct_correlation_peak(np.zeros((2**7,2**7)), di, dj, fwhm=3)
print('{:.4f}'.format(di))
print('{:.4f}'.format(dj))

ddi,ddj,ii,jj = get_top_paraboloid(C_tilde)
(dj, (jj+ddj))
(di, (ii+ddi))

ddi,ddj,ii,jj = get_top_2d_gaussian(C_tilde)


        
ddi,ddj,ii,jj = get_top_birchfield(C_tilde)
(dj, (jj+ddj))
(di, (ii+ddi))
assert np.isclose(di, ii+ddi, atol=5e-1)
assert np.isclose(dj, jj+ddj, atol=5e-1)
dddi,dddj = (di-(ii+ddi)), (dj-(jj+ddj))
print('birchfield locator is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(dddi), abs(dddj))) + ' pixels')     
    
    
print('test phase plane localizators')
# load data
im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**7, max_range=1)
Q_tilde = construct_phase_plane(im1, di, dj)
W_tilde = np.ones_like(Q_tilde, dtype=np.bool_)

#di_hat, dj_hat = phase_ransac(Q_tilde, precision_threshold=.05)
#(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)
#print('phase_ransac is OK, maximum difference is '+\
#      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')

beta_1,beta_2 = 0.35, 0.5
W_1,W_2 = raised_cosine(im1,beta_1), raised_cosine(im2,beta_2)
S_1,S_2 = np.fft.fft2(im1), np.fft.fft2(im2)
Q_hat = (np.multiply(W_1,S_1))*np.conjugate(np.multiply(W_2,S_2))
W_hat = adaptive_masking(Q_hat)
print('di:{:.4f}'.format(di)+' dj:{:.4f}'.format(dj))
m_close = np.array([np.round(1e1*di)/1e1, \
                    np.round(1e1*dj)/1e1])
#(m, snr) = phase_newton(Q_tilde, np.ones_like(Q_tilde,dtype=np.float32), \
#                      m_close, j=5)

(m, snr) = phase_newton(Q_hat, W_hat, np.array([0.0, 0.0]), j=5e2)
(ddi,ddj) = test_subpixel_localization(m[0],m[1], di,dj, tolerance=.1)
print('phase_tpss is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')    
    
di_hat, dj_hat = phase_svd(Q_tilde, W_tilde)
(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)
print('phase_svd is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')    
    
    
    
Q_hat = phase_only_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('phase_only_corr correlator is OK, snr is '+'{:.2f}'.format(snr))



print('test phase cycle localizators')

im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**7, max_range=10)
Q_tilde = construct_phase_plane(im1, di, dj)
print('di:{:.4f}'.format(di)+' dj:{:.4f}'.format(dj))

di_hat, dj_hat = phase_difference(Q_tilde)
(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)
print('phase_difference is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels') 

di_hat, dj_hat = phase_radon(Q_tilde, coord_system='ij')
(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)
print('phase_difference is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels') 



# load data
im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**7, max_range=10)
Q_tilde = construct_phase_plane(im1, di, dj)

(im1,_), (im2,_) = perdecomp(im1), perdecomp(im2)

print('testing (frequency) correlators')

Q_hat = phase_only_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('phase_only_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

Q_hat = symmetric_phase_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('symmetric_phase_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

Q_hat = amplitude_comp_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('amplitude_comp_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

Q_hat = phase_corr(im1, im2)
_ = test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('phase_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

Q_hat = orientation_corr(im1, im2)
_ = test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('orientation_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

#Q_hat = windrose_corr(im1, im2)
#_ = test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
#snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
#print('windrose_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

Q_hat = cross_corr(im1, im2)
_ = test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('cross_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

print('binary_orientation_corr does not seem to do a good job')
# Q_hat = binary_orientation_corr(im1, im2)
# _ = test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
# snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
# print('binary_orientation_corr correlator is OK, snr is '+'{:.2f}'.format(snr))

(Q_hat,W,m0) = cosi_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('cosi_corr is correlator OK, snr is '+'{:.2f}'.format(snr))

Q_hat = robust_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=1)
snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
print('robust_corr is correlator OK, snr is '+'{:.2f}'.format(snr))



(m, n) = Q_tilde.shape
fi = np.flip( (np.arange(0,m)-(m/2)) /m)
fj = np.flip( (np.arange(0,n)-(n/2)) /n)
    
Fj = np.repeat(fj[np.newaxis,:],m,axis=0)
Fi = np.repeat(fi[:,np.newaxis],n,axis=1)
Fj = np.fft.fftshift(Fj)
Fi = np.fft.fftshift(Fi)    
    
di_hat, dj_hat = phase_lsq(Fi.flatten(), Fj.flatten(), Q_tilde.flatten())  
(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)    
print('phase_lsq is OK, maximum difference is '+\
      '{:.2f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')
    



C_hat = sign_only_corr(im1, im2)
test_phase_plane_localization(Q_hat, di, dj, tolerance=.5)
snr = signal_to_noise(Q_tilde, Q_hat, norm=1)
print('cosine_corr is OK, snr is '+'{:.2f}'.format(snr))


test_phase_plane_localization



#Q_hat = phase_corr(im1, im2) # cross-power spectrum
#C_hat = sign_only_corr(im1, im2)
Q_hat = cosine_corr(im1, im2)
#Q_hat = phase_only_corr(im1, im2)
#Q_hat = robust_corr(im1, im2)
#(Q_hat,W,m0) = cosi_corr(im1, im2)




Q_hat = normalize_power_spectrum(Q_hat)
Q = np.fft.fftshift(Q_hat)

C = local_coherence(Q_hat, ds=1)

# look at signal-to-noise
snr = signal_to_noise(Q_hat,Q_tilde,norm=2)
print(snr)

(m,n) = Q.shape
fy = np.flip((np.arange(0,m)-(m/2)) /m)
fx = np.flip((np.arange(0,n)-(n/2)) /n)
    
Fx = np.repeat(fx[np.newaxis,:],m,axis=0)
Fy = np.repeat(fy[:,np.newaxis],n,axis=1)  

#IN = Q!=0
IN = np.fft.fftshift(low_pass_circle(Q, r =.35))

Q_dummy = np.fft.fftshift(Q_tilde)

di = phase_ransac(Q)


di_hat = phase_difference_1d(Q)

di_hat,dj_hat = phase_pca(Fy[IN], Fx[IN], Q_dummy[IN])  
di_hat,dj_hat = phase_pca(Fy[IN], Fx[IN], Q[IN])





# Fr = np.cos(np.radians(theta))*Fx \
#     - np.sin(np.radians(theta))*Fy







snr_spat = test_phase_plane_localization(Q_hat, di, dj, tolerance=0.5)


imShow= False
if imShow:
    ax = plt.subplot()
    im = ax.imshow(np.angle(Q_hat))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    
    
    ax = plt.subplot()
    im = ax.imshow(np.angle(Q_tilde))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


#estimate period
Q_shift = np.roll(Q, (1,1))

Q_di = np.multiply(np.conj(Q), np.roll(Q, (1,0)))
Q_dj = np.multiply(np.conj(Q), np.roll(Q, (0,1)))

ang_Q_di, ang_Q_dj = np.angle(Q_di), np.angle(Q_dj)

ang_dQ = np.arctan2(ang_Q_di, ang_Q_dj)

if imShow:
    ax = plt.subplot()
    im = ax.imshow(np.angle(Q_di))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    
    
    ax = plt.subplot()
    im = ax.imshow(np.angle(Q_tilde))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
