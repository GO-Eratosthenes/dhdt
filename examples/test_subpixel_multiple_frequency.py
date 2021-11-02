import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eratosthenes.generic.test_tools import create_sample_image_pair, \
    construct_phase_plane, signal_to_noise, test_phase_plane_localization, \
    test_normalize_power_spectrum, test_subpixel_localization, \
    construct_correlation_peak
from eratosthenes.preprocessing.image_transforms import mat_to_gray
from eratosthenes.processing.matching_tools import get_integer_peak_location
from eratosthenes.processing.matching_tools_spatial_subpixel import \
    get_top_moment
from eratosthenes.processing.matching_tools_frequency_correlators import \
    phase_corr, phase_only_corr, robust_corr, \
    orientation_corr, windrose_corr, binary_orientation_corr, \
    cosi_corr, sign_only_corr, cosine_corr, cross_corr, \
    symmetric_phase_corr, amplitude_comp_corr
from eratosthenes.processing.matching_tools_frequency_subpixel import \
    phase_svd, phase_pca, phase_tpss, phase_difference, phase_ransac, \
    phase_lsq, phase_radon, phase_hough, phase_secant
from eratosthenes.processing.matching_tools_frequency_filters import \
    normalize_power_spectrum, perdecomp, low_pass_circle, raised_cosine, \
    local_coherence, thresh_masking

import matplotlib.pyplot as plt
import scipy.stats as st
from PIL import Image
from matplotlib import cm

print('test phase cycle localizators')
im_rad = 2**5

im1,im2,di,dj,im1_big = create_sample_image_pair(d=im_rad, max_range=10)
Q_tilde = construct_phase_plane(im1, di, dj)
print(str(di)+' '+str(dj))

im_show = False
mag_study = False
n_samp = 40

noise_var = 10**np.linspace(-3,-.01,n_samp)
snr_var,coh_var = np.zeros_like(noise_var), np.zeros_like(noise_var)
spat_var, houg_var = np.zeros_like(noise_var), np.zeros_like(noise_var)
spat_tim, houg_tim = np.zeros_like(noise_var), np.zeros_like(noise_var)
houg_est, diff_est = np.zeros((n_samp,2)), np.zeros((n_samp,2))
diff_var, rans_var = np.zeros_like(noise_var), np.zeros_like(noise_var)
diff_tim, rans_tim = np.zeros_like(noise_var), np.zeros_like(noise_var)
rado_var, rado_tim = np.zeros_like(noise_var), np.zeros_like(noise_var)
seca_var, seca_tim = np.zeros_like(noise_var), np.zeros_like(noise_var)

for idx, sigm in enumerate(noise_var):
    # increase noise and look at SNR
    im1_noise = im1 + np.random.normal(scale=sigm, 
                                       size=(2*im_rad, 2*im_rad))
    im2_noise = im2 + np.random.normal(scale=sigm, 
                                       size=(2*im_rad, 2*im_rad))
    im1_noise,_ = perdecomp(im1_noise)
    im2_noise,_ = perdecomp(im2_noise)
    Q_hat = phase_only_corr(im1_noise, im2_noise)
    
    snr = signal_to_noise(Q_hat, Q_tilde, norm=1)
    Q_n = normalize_power_spectrum(Q_hat)
    Coh = local_coherence(Q_n)
    coh = np.sum(Coh)/Q_n.size
    
    snr_var[idx], coh_var[idx] = snr, coh
    
    if mag_study:    
        # is magnitude related to signal?
        plt.imshow(np.fft.fftshift(np.angle(Q_hat)),cmap='twilight'), plt.show()
        plt.imshow(np.fft.fftshift(np.log(np.abs(Q_hat))),cmap='RdPu'), plt.show()
        Q_diff = np.abs(Q_n-Q_tilde)
        
        M_10 = np.log10(np.abs(Q_hat)).flatten()
        Q_d = Q_diff.flatten()
        
        xx,yy = np.mgrid[-3:10:100j, 0:2:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([M_10, Q_d])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(np.rot90(f), cmap='Blues', extent=[-3, 10, 0, 2])
        cfset = ax.contour(xx,yy, f, cmap='Blues')
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('frequency magnitude')
        ax.set_ylabel('vector difference')
        ax.set_xlim(-3,+10)
        ax.set_ylim(+0,+2)
        ax.set_aspect('auto')
        mag_tick = 10**ax.get_xticks()
        ax.set_xticklabels(["%.1E" % x for x in mag_tick])
        plt.show()

        C_f = Coh.flatten()
        Q_d = Q_diff.flatten()
        
        #plt.scatter(C_f,Q_d), plt.show()
        
        xx,yy = np.mgrid[0:1:100j, 0:2:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([C_f, Q_d])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(np.rot90(f), cmap='Blues', extent=[0, 1, 0, 2])
        cfset = ax.contour(xx,yy, f, cmap='Blues')
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('local coherence')
        ax.set_ylabel('vector difference')
        ax.set_xlim(+0,+1)
        ax.set_ylim(+0,+2)
        ax.set_aspect('auto')
        plt.show()
    
    if im_show:    
        Q_mat = mat_to_gray(np.fft.fftshift(np.angle(Q_hat)))
        im = Image.fromarray(np.uint8(cm.twilight(Q_mat)*255))
        im.save('Q'+ \
                '-snr-'+ '{:.3f}'.format(snr) + \
                '-noise-'+ '{:.5f}'.format(sigm) + '.png')
        C_mat = mat_to_gray(np.fft.fftshift(Coh))
        im = Image.fromarray(np.uint8(cm.plasma(C_mat)*255))
        im.save('C'+ \
                '-snr-'+ '{:.3f}'.format(snr) + \
                '-noise-'+ '{:.5f}'.format(sigm) + '.png') 
    
    # test different sub-pixel localizators
    
    # also traditional!
    tic = time.perf_counter()
    C_hat = np.real(np.fft.fftshift(np.fft.ifft2(Q_n)))
    ddi,ddj,di_hat,dj_hat = get_top_moment(C_hat)
    toc = time.perf_counter()
    spat_var[idx] = np.sqrt((di-(ddi+di_hat))**2 + (dj-(ddj+dj_hat))**2)
    spat_tim[idx] = toc-tic
    #di_hat,dj_hat,_,_ = get_integer_peak_location(C_hat)
    #spat_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    
    # ransac
    #tic = time.perf_counter()
    #di_hat,dj_hat = phase_ransac(Q_n, max_displacement=10)
    #toc = time.perf_counter()
    #rans_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    #rans_tim[idx] = toc-tic
    
    # phase-difference
    tic = time.perf_counter()
    W = thresh_masking(Q_hat)
    di_hat,dj_hat = phase_difference(Q_hat, W)
    toc = time.perf_counter()
    diff_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    diff_tim[idx] = toc-tic
    diff_est[idx,:] = np.array([di_hat, dj_hat])

    # phase-difference
    tic = time.perf_counter()
    di_hat,dj_hat = phase_radon(Q_hat)
    toc = time.perf_counter()
    rado_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    rado_tim[idx] = toc-tic
    
    # hough
    tic = time.perf_counter()
    di_hat,dj_hat = phase_hough(Q_n, max_displacement=6, param_spacing=.1, \
                                sample_fraction=.25, \
                                W=np.abs(np.log10(Q_hat))) # W=Coh
    toc = time.perf_counter()
    houg_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    houg_tim[idx] = toc-tic
    houg_est[idx,:] = np.array([di_hat, dj_hat])

    # phase-difference
    tic = time.perf_counter()
    di_hat,dj_hat = phase_secant(Q_hat, W)
    toc = time.perf_counter()
    seca_var[idx] = np.sqrt((di-di_hat)**2 + (dj-dj_hat)**2)
    seca_tim[idx] = toc-tic
    
if im_show:
    fig,ax = plt.subplots()
    ax.set_xlabel('signal to noise ratio - SNR')
    ax.set_ylabel('sum of local coherence')
    ax.set_aspect('equal'), ax.grid()
    ax.scatter(snr_var,coh_var,marker='2')
    plt.show()

    fig,ax = plt.subplots()
    ax.set_xlabel('signal to noise ratio - SNR')
    ax.set_ylabel('estimation error [pixel]')
#    ax.set_aspect('equal'), ax.grid()
    ax.scatter(snr_var,rans_var,marker='4',label='RANSAC')
    ax.scatter(snr_var,diff_var,marker='3',label='phase difference')
    ax.scatter(snr_var,spat_var,marker='2',label='correlation surface')
    ax.scatter(snr_var,houg_var,marker='1',label='Hough tranform')
    ax.scatter(snr_var,rado_var,marker='4',label='Radon tranform')
    ax.scatter(snr_var,seca_var,marker='3',label='Broyden''s method' )
    ax.legend(loc='upper left')
    ax.invert_xaxis()
    plt.yscale('log')
    plt.title('template-size: ' + str(2*im_rad))
    plt.show()

    fig,ax = plt.subplots()
    ax.set_xlabel('signal to noise ratio - SNR')
    ax.set_ylabel('estimation error [pixel]')
#    ax.set_aspect('equal'), ax.grid()
    ax.scatter(snr_var,rans_tim,marker='4',label='RANSAC')
    ax.scatter(snr_var,diff_tim,marker='3',label='phase difference')
    ax.scatter(snr_var,spat_tim,marker='2',label='correlation surface')
    ax.scatter(snr_var,houg_tim,marker='1',label='Hough tranform')
    ax.legend(loc='upper left')
    ax.invert_xaxis()
    plt.yscale('log')
    plt.title('template-size: ' + str(2*im_rad))
    plt.show()



di_tilde, dj_tilde = phase_hough(Q_tilde, sample_fraction=1)

Q_hat = phase_only_corr(im1, im2)
Q_hat = robust_corr(im1, im2)
di_hat, dj_hat = phase_hough(Q_hat, sample_fraction=.01)

