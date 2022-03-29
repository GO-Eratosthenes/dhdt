# test different matching algorithms

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eratosthenes.generic.handler_im import \
    nan_resistant_conv2, nan_resistant_diff2
from eratosthenes.generic.filtering_statistical import mad_filtering
from eratosthenes.generic.test_tools import create_sample_image_pair, \
    construct_phase_plane, signal_to_noise, test_phase_plane_localization, \
    test_normalize_power_spectrum, test_subpixel_localization, \
    construct_correlation_peak, \
    create_sheared_image_pair, create_scaled_image_pair
from eratosthenes.preprocessing.image_transforms import \
    histogram_equalization
from eratosthenes.processing.matching_tools import \
    get_integer_peak_location
from eratosthenes.processing.matching_tools_frequency_affine import \
    scaling_through_power_summation
from eratosthenes.processing.matching_tools_frequency_correlators import \
    phase_corr, phase_only_corr, robust_corr, \
    orientation_corr, windrose_corr, binary_orientation_corr, \
    cosi_corr, sign_only_corr, cosine_corr, cross_corr, \
    symmetric_phase_corr, amplitude_comp_corr, upsampled_cross_corr, \
    projected_phase_corr
from eratosthenes.processing.matching_tools_frequency_subpixel import \
    phase_svd, phase_pca, phase_tpss, phase_difference, phase_ransac, \
    phase_lsq, phase_radon, phase_hough, phase_gradient_descend,\
    phase_weighted_pca, phase_secant #, phase_jac # jac:mag er later wel uit
from eratosthenes.processing.matching_tools_frequency_filters import \
    normalize_power_spectrum, perdecomp, low_pass_circle, raised_cosine, \
    local_coherence, thresh_masking, adaptive_masking, gaussian_mask, \
    low_pass_ellipse

from eratosthenes.processing.matching_tools_differential import \
    simple_optical_flow, hough_optical_flow, affine_optical_flow

from eratosthenes.processing.matching_tools_spatial_correlators import \
    normalized_cross_corr, cumulative_cross_corr, sum_sq_diff, \
    sum_sad_diff, weighted_normalized_cross_correlation
from eratosthenes.processing.matching_tools_spatial_subpixel import \
    get_top_moment, get_top_gaussian, get_top_parabolic, \
    get_top_birchfield, get_top_2d_gaussian, get_top_paraboloid, \
    get_top_blue

im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**4, max_range=1, integer=False)
print('di:{:+.4f}'.format(di)+' dj:{:+.4f}'.format(dj))
mu = np.mean(im2.ravel())

W1 = np.random.rand(2**5,2**5)<.5
W2 = np.random.rand(2**9,2**9)<.5
#W1, W2 = im2>mu, im1_big>mu
wncc = weighted_normalized_cross_correlation(im2,im1_big,
                                             W1, W2)




di, dj = 2*np.random.random()-1, 2*np.random.random()-1
#di, dj = 10*di, 10*dj
C_tilde = construct_correlation_peak(np.zeros((2**7,2**7)), di, dj)

ddi,ddj,ii,jj = get_top_blue(C_tilde, ds=1)


from eratosthenes.presentation.image_io import make_image

from eratosthenes.preprocessing.shadow_filters import anistropic_diffusion_scalar

im1_ani = anistropic_diffusion_scalar(im1_big)


m,n = 20,16
nan_elem = 40
I = np.random.random((m,n))
I_clean = np.copy(I)
nan_idx = np.random.randint(m*n, size=nan_elem)
I[np.unravel_index(nan_idx, I.shape, 'C')] = np.nan

#dy = np.outer(np.array([2,4,2]),np.array([1,2,1]))
dy = np.outer(np.array([-1,0,+1]),np.array([1,2,1]))
I_dy = nan_resistant_diff2(I,dy)




di_hof,dj_hof,sc_hof = hough_optical_flow(im1,im2,param_resol=40, sample_fraction=1)
print('di:{:+.4f}'.format(di_hof[0])+' dj:{:+.4f}'.format(dj_hof[0]))



im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**4, max_range=1, integer=False)
C_cos = cosine_corr(np.pad(im1,((4,16),(10,10))),\
                             np.pad(im1,((0,20),(0,20))))

C_cos = cosine_corr(im1,im2)

# di_cos, dj_cos






from eratosthenes.preprocessing.shadow_filters import anistropic_diffusion_scalar

im1_ani = anistropic_diffusion_scalar(im1_big)
wncc = weighted_normalized_cross_correlation(im2,im1_big[200:-200,200:-200])


im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**5, max_range=1, integer=False)


beta_1,beta_2 = 0.35, 0.5
W_1,W_2 = raised_cosine(im1,beta_1), raised_cosine(im2,beta_2)
im1,_ = perdecomp(im1)
im2,_ = perdecomp(im2)
S_1,S_2 = np.fft.fft2(im1), np.fft.fft2(im2)
Q_hat = (np.multiply(W_1,S_1))*np.conjugate(np.multiply(W_2,S_2))

C_hat = np.real(np.fft.ifft2(Q_hat))

Q_tilde = construct_phase_plane(im1, di, dj, indexing='ij')
W = low_pass_ellipse(Q_tilde, r1=.1, r2=.3)

C_tilde = np.real(np.fft.fftshift(np.fft.ifft2(W*Q_tilde)))

np.fft.fftshift(C_hat)

im1, im2, di, dj, im1_big = create_sheared_image_pair(d=2**5,
                                                      sh_i=0.00, sh_j=0.04,
                                                      max_range=1)
print(di, dj)
u, v, A, snr = affine_optical_flow(im1, im2, model='affine', iteration=10)


im1, im2, di, dj, im1_big = create_sheared_image_pair(d=2**5,
                                                      sh_i=0.00, sh_j=0.00,
                                                      max_range=1)

u,v = hough_optical_flow(im1, im2, num_estimates=1)





print('test phase plane localizators')
#tries = 2
#d_stack = np.zeros((tries,4))
#for i in np.arange(tries):

im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**5, max_range=1, integer=False)    
di_hof,dj_hof = hough_optical_flow(im1,im2,param_resol=100, sample_fraction=1)    
  
    # load data
#im1,im2,di,dj,im1_big = create_sample_image_pair(d=2**7, max_range=100, integer=True)
#Q_tilde = construct_phase_plane(im1, di, dj)
#W_tilde = np.ones_like(Q_tilde, dtype=np.bool_)
#print('di:{:+.4f}'.format(di)+' dj:{:+.4f}'.format(dj))


#im1,im2,di,dj,_ = create_sheared_image_pair(d=2**7,sh_x=-0.1, sh_y=0.2, max_range=1)


im1,im2,di,dj,_ = create_scaled_image_pair(d=2**7,sc_x=1.3, sc_y=1.0, max_range=0)

im1,_ = perdecomp(im1)
im2,_ = perdecomp(im2)
S_1,S_2 = np.fft.fft2(im1), np.fft.fft2(im2)

a,b = scaling_through_power_summation(S_1,S_2)


di_soc, dj_soc = sign_only_corr(im1, im2)

di_prj, dj_prj = projected_phase_corr(im1, im2)



di_cos, dj_cos = cosine_corr(im1, im2)





beta_1,beta_2 = 0.35, 0.5
W_1,W_2 = raised_cosine(im1,beta_1), raised_cosine(im2,beta_2)
im1,_ = perdecomp(im1)
im2,_ = perdecomp(im2)
S_1,S_2 = np.fft.fft2(im1), np.fft.fft2(im2)
Q_hat = (np.multiply(W_1,S_1))*np.conjugate(np.multiply(W_2,S_2))

# test Jacobian
#dQd1,dQd2 = phase_jac(Q_tilde, np.array([di+.3, dj-.4]))

di_sec, dj_sec = phase_secant(Q_tilde)
print('di:{:+.4f}'.format(di_sec)+' dj:{:+.4f}'.format(dj_sec))
#di_gd, dj_gd = phase_gradient_descend(Q_tilde)    

W_hat = adaptive_masking(Q_hat)
di_sec, dj_sec = phase_secant(Q_hat, W=W_hat)
(m, snr) = phase_tpss(Q_hat, W_hat, np.array([0.0, 0.0]), j=1e2)


#OK di_svd, dj_svd = phase_svd(Q_tilde, W_tilde, rad=0.4)
#OK di_lsq, dj_lsq = phase_lsq(Q_tilde, W_tilde)
di_pca, dj_pca = phase_pca(Q_tilde, W_tilde)

plt.imshow(np.fft.fftshift(np.angle(Q_tilde)), vmin=-np.pi, vmax=+np.pi), 
plt.colorbar(), plt.show()
plt.imshow(np.fft.fftshift(np.angle(Q_hat)), vmin=-np.pi, vmax=+np.pi), 
plt.colorbar(), plt.show()






W_gauss = gaussian_mask(Q_tilde)

di_pcw, dj_pcw = phase_weighted_pca(Q_tilde, W_gauss)
#    d_stack[i,:] = np.array([di, dj, di_pca, dj_pca])

#import matplotlib.pyplot as plt
#plt.scatter(d_stack[:,0],d_stack[:,2]), plt.show()

#plt.scatter(d_stack[:,1],d_stack[:,3]), plt.show()


W_hat = adaptive_masking(Q_hat)

(m, snr) = phase_tpss(Q_hat, W_hat, np.array([0.0, 0.0]), j=5e2)
(ddi,ddj) = test_subpixel_localization(m[0],m[1], di,dj, tolerance=.1)
print('phase_tpss is OK, maximum difference is '+\
      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')    


di_gd, dj_gd = phase_gradient_descend(Q_hat,W_hat) 
di_pca, dj_pca = phase_pca(Q_hat, W_hat)



#di_hat, dj_hat = phase_ransac(Q_tilde, precision_threshold=.05)
#(ddi,ddj) = test_subpixel_localization(di_hat,dj_hat,di,dj, tolerance=.1)
#print('phase_ransac is OK, maximum difference is '+\
#      '{:.4f}'.format(np.maximum(abs(ddi), abs(ddj))) + ' pixels')


print('di:{:.4f}'.format(di)+' dj:{:.4f}'.format(dj))
m_close = np.array([np.round(1e1*di)/1e1, \
                    np.round(1e1*dj)/1e1])
#(m, snr) = phase_newton(Q_tilde, np.ones_like(Q_tilde,dtype=np.float32), \
#                      m_close, j=5)




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



temp_size = 2**6
im1,im2,di,dj,im1_big = create_sample_image_pair(d=temp_size, max_range=1)
print('{:.4f}'.format(di))
print('{:.4f}'.format(dj))


(di_hat,dj_hat,di_lamb,dj_lamb) = simple_optical_flow(im1, im2, temp_size, \
                                                      temp_size//2, \
                                                      temp_size//2, 
                                                      sigma=2)

S1, S2 = np.fft.fft2(im1), np.fft.fft2(im2) 
di_02,dj_02 = upsampled_cross_corr(S1,S2, upsampling=2) #todo
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
