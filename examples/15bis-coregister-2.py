import matplotlib.pyplot as plt

import os
import numpy as np
from scipy import ndimage


from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.mapping_tools import \
    pix2map
from eratosthenes.generic.handler_im import \
    bilinear_interpolation, rescale_image
from eratosthenes.generic.terrain_tools import \
    ridge_orientation, terrain_curvature
from eratosthenes.generic.gis_tools import get_mask_boundary

from eratosthenes.input.read_sentinel2 import \
    list_central_wavelength_msi
from eratosthenes.preprocessing.image_transforms import \
    mat_to_gray, normalize_histogram, gamma_adjustment, histogram_equalization
from eratosthenes.preprocessing.shadow_geometry import \
    cast_orientation
from eratosthenes.preprocessing.shadow_filters import \
    fade_shadow_cast, anistropic_diffusion_scalar
from eratosthenes.preprocessing.acquisition_geometry import \
    get_template_acquisition_angles, get_template_aspect_slope

from eratosthenes.preprocessing.shadow_transforms import \
    entropy_shade_removal, shadow_free_rgb, shade_index, \
    normalized_range_shadow_index
from eratosthenes.preprocessing.color_transforms import rgb2lms

from eratosthenes.processing.matching_tools import \
    get_coordinates_of_template_centers
from eratosthenes.processing.coupling_tools import \
    match_pair
from eratosthenes.processing.matching_tools_differential import hough_sinus

from eratosthenes.postprocessing.solar_tools import \
    make_shading, make_shadowing
from eratosthenes.presentation.image_io import \
    output_image, resize_image, output_mask

toi = 15
window_size = 2**3 #2**3# #
boi = ['red', 'green', 'blue', 'nir']
s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['common_name'].isin(boi)]

if toi==15:
    s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019/'
elif toi == 25:
    s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-25-10-2019/'

fpath = os.path.join(s2path, 'shadow.tif')

M_dir, M_name = os.path.split(fpath)
#dat_path = '/Users/Alten005/GO-eratosthenes/start-code/examples/'
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.getcwd()!=dir_path:
    os.chdir(dir_path) # change to directory where script is situated

(M, spatialRefM, geoTransformM, targetprjM) = read_geo_image(fpath)
M *= -1

Z = read_geo_image(os.path.join(Z_dir, Z_file))[0]
R = read_geo_image(os.path.join(Z_dir, R_file))[0]

# create observation angles
if toi==15:
    fpath = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019-full', \
                          'T05VMG_20191015T213531_B08.jp2')
elif toi==25:
    fpath = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019-full', \
                         'T05VMG_20191015T213531_B08.jp2')

_,spatialRefI,geoTransformI,targetprjI = read_geo_image(fpath)

path_meta = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'+\
    'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE/'+\
    'GRANULE/L1C_T05VMG_A022534_20191015T213843/QI_DATA'

#det_stack =read_detector_mask(path_meta, (10980, 10980, len(s2_df)), s2_df, geoTransformI)
#Zn,Az = read_view_angles_s2(s2path, 'metadata.xml', det_stack, s2_df)

#X_grd,Y_grd = pix_centers(geoTransformM, M.shape[0], M.shape[1], make_grid=True)
#I_grd,J_grd = map2pix(geoTransformI, X_grd, Y_grd)

#I_sub = bilinear_interpolation(Az[:,:,0], J_grd, I_grd)

#fpath = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
#                      'viewZn.tif')
#make_geo_im(I_sub, geoTransformM, spatialRefM, fpath)

if toi==15:
    Det = read_geo_image(os.path.join(s2path, 'detIdBlue.tif'))[0]
    Zn = read_geo_image(os.path.join(s2path, 'viewZn.tif'))[0]
    Az = read_geo_image(os.path.join(s2path, 'viewAz.tif'))[0]

Blue = read_geo_image(os.path.join(s2path, 'B2.tif'))[0]
Green = read_geo_image(os.path.join(s2path, 'B3.tif'))[0]
Red = read_geo_image(os.path.join(s2path, 'B4.tif'))[0]
Near = read_geo_image(os.path.join(s2path, 'B8.tif'))[0]


Sn = normalized_range_shadow_index(mat_to_gray(Red),
                                   mat_to_gray(Green),
                                   mat_to_gray(Blue))
Sn = normalize_histogram(-Sn)
output_image(Sn, 'Red-normalized-range-shadow-index.jpg', cmap='gray')

Li,Mi,Si = rgb2lms(mat_to_gray(Red), mat_to_gray(Green), mat_to_gray(Blue))
RGB = shadow_free_rgb(mat_to_gray(Blue), mat_to_gray(Green),
                      mat_to_gray(Red), mat_to_gray(Near))

RGB = np.dstack((gamma_adjustment(Red, gamma=.25),
                 gamma_adjustment(Green, gamma=.25),
                 gamma_adjustment(Blue, gamma=.25)))

RGB = mat_to_gray(1.5*RGB)
RGB[RGB<.3] = .3
output_image(RGB, 'Red-rgb-gamma.jpg')

Si,Ri = entropy_shade_removal(mat_to_gray(Blue),
                              mat_to_gray(Red),
                              mat_to_gray(Near), a=138)
#Si,Ri = normalize_histogram(Si), normalize_histogram(Ri)

output_image(Si, 'Red-entropy-shade-removal.jpg', cmap='gray')
output_image(Ri, 'Red-entropy-albedo.jpg', cmap='gray')


ani = anistropic_diffusion_scalar(np.dstack((Sn,Si,Ri)), n=8)
output_image(ani[:,:,0], 'Red-entropy-anistropy.jpg', cmap='gray')


#Si = s_curve(Si, a=20, b=0)
#Si += .5
#Si[Si<-.5] = -.5

Si_ani = anistropic_diffusion_scalar(Si)
Sc = (mat_to_gray(-Sn)*mat_to_gray(Si))

sample_I, sample_J = get_coordinates_of_template_centers(Zn, window_size)

Slp,Asp = get_template_aspect_slope(Z,
                                    sample_I,sample_J,
                                    window_size)
output_image(resize_image(Asp, Z.shape, method='nearest'),
             'Red-aspect-w'+str(2*window_size)+'.jpg', cmap='twilight')
output_image(resize_image(Slp, Z.shape, method='nearest'),
             'Red-slope-w'+str(2*window_size)+'.jpg', cmap='magma')

#if toi==15:
#    Azi,Zen = get_template_acquisition_angles(Az, Zn, Det.astype('int64'),
#                                              sample_I, sample_J,
#                                              window_size)

zn, az = 90-21.3, 174.2 # 15 Oct 2019
#Zn, Az = 90-17.7, 174.8 # 25 Oct 2019

# generate shadow image
Shw =  make_shadowing(Z, az, zn)
Shd =  make_shading(Z, az, zn)


import morphsnakes as ms

counts = 30
#M_acwe_si = ms.morphological_chan_vese(Si, counts, init_level_set=Shw,
#                                       smoothing=0, lambda1=1, lambda2=1,
#                                       albedo=Ri)
for i in np.linspace(0,counts,counts+1).astype(np.int8):
    M_acwe_si = ms.morphological_chan_vese(Si, i, init_level_set=Shw,
                                           smoothing=0, lambda1=1, lambda2=1,
                                           albedo=Ri)
    Bnd = get_mask_boundary(M_acwe_si)
    output_mask(Bnd,'Red-shadow-polygon-raw'+str(i).zfill(3)+'.png')

output_image(-1.*M_acwe, 'Red-snake-shadow-si.jpg', cmap='gray')


# fading shadowing
Shf = fade_shadow_cast(Shw, az)
Shi = (Shd+.5) *(1-(0.75*Shf))

output_image(Shw!=True, 'Red-shadow.jpg', cmap='gray')
output_image(-1*Shf, 'Red-shadow-fade.jpg', cmap='gray')

#output_image(Shi, 'Red-artificial-scene.jpg', cmap='gray')

#plt.imshow(Shd, cmap='gray'), plt.show()
#plt.imshow(M, cmap='gray'), plt.show()

# filter along sun direction....?

#from eratosthenes.preprocessing.image_transforms import \
#    log_adjustment, mat_to_gray
#L = log_adjustment(mat_to_gray(M)*2**8)/2**16

#H = match_histograms(L,Shd)

F = cast_orientation(Shd, Az, indexing='xy')

# fa = rotated_sobel(Az, indexing='xy')
# from scipy import ndimage
# D = ndimage.convolve(Shd, fa)

N = cast_orientation(Si, Az, indexing='xy')

# make Mask
Stable = (R==0) & (Shw!=1)

# window_size
t_rad = 3
t_size = (2*t_rad)+1
s_size = 7
num_disp = 1
sample_X,sample_Y = pix2map(geoTransformM, sample_I, sample_J)
match_X,match_Y,match_score = match_pair(Shi, Si,
                                         Stable, Stable,
                                         geoTransformM, geoTransformM,
                                         sample_X, sample_Y,
                                         temp_radius=window_size,
                                         search_radius=window_size,
                                         correlator='robu_corr', subpix='moment',
                                         metric='peak_entr')
#                                         correlator='hough_opt_flw',
#                                         preprocessing='hist_equal',
#                                         num_estimates=num_disp,
#                                         max_amp=2)


if num_disp>1:
    dY,dX = np.repeat(sample_Y[:,:,np.newaxis], num_disp, axis=2)-match_Y, \
            np.repeat(sample_X[:,:,np.newaxis], num_disp, axis=2)-match_X
else:
    dY,dX = sample_Y-match_Y, sample_X-match_X

IN = ~np.isnan(dX)
IN[IN] = np.remainder(dX[IN],1)!=0

#plt.imshow(dY, vmin=-20, vmax=+20), plt.show()
#plt.imshow(dX, vmin=-20, vmax=+20), plt.show()

#plt.figure(), plt.hexbin(dX[IN], dY[IN], gridsize=40)
# Slp<20
dx_coreg, dy_coreg = np.median(dX[IN]), np.median(dY[IN])

di_coreg = +1*dy_coreg/geoTransformM[5]
dj_coreg = +1*dx_coreg/geoTransformM[1]

Sc_coreg = bilinear_interpolation(Sc, dj_coreg, di_coreg)
Si_coreg = bilinear_interpolation(Si, dj_coreg, di_coreg)
Ri_coreg = bilinear_interpolation(Ri, dj_coreg, di_coreg)
RGB_coreg = bilinear_interpolation(RGB, dj_coreg, di_coreg)

#fout = os.path.join(s2path, 'shadows_coreg.tif')
#make_geo_im(Sc_coreg, geoTransformM, targetprjI, fout)
#fout = os.path.join(s2path, 'shadow_coreg.tif')
#make_geo_im(Si_coreg, geoTransformM, targetprjI, fout)
#fout = os.path.join(s2path, 'albedo_coreg.tif')
#make_geo_im(Ri_coreg, geoTransformM, targetprjI, fout)
#fout = os.path.join(s2path, 'rgb_clean.tif')
#make_geo_im(RGB_coreg, geoTransformM, targetprjI, fout)
#fout = os.path.join(s2path, 'shading.tif')
#make_geo_im(Shd, geoTransformM, targetprjI, fout)
#fout = os.path.join(s2path, 'shadowing.tif')
#make_geo_im(Shw, geoTransformM, targetprjI, fout)


Rho, Phi = np.sqrt(dY**2+dX**2), np.arctan2(dX,dY)

match_score[match_score==0] = np.nan
output_image(resize_image(match_score, Z.shape, method='nearest'),
             'Red-disp-score-w'+str(2*window_size)+'.jpg', cmap='viridis')
output_image(resize_image(Phi, Z.shape, method='nearest'),
             'Red-disp-dir-w'+str(2*window_size)+'.jpg', cmap='twilight')
output_image(resize_image(Rho, Z.shape, method='nearest'),
             'Red-disp-mag-w'+str(2*window_size)+'.jpg', cmap='magma')

dY_coreg, dX_coreg = dY-dy_coreg, dX-dx_coreg
Rho_coreg, Phi_coreg = np.sqrt(dY_coreg**2+dX_coreg**2), \
                       np.arctan2(dX_coreg,dY_coreg)
output_image(resize_image(Phi_coreg, Z.shape, method='nearest'),
             'Red-disp-dir-coreg-w'+str(2*window_size)+'.jpg', cmap='twilight')
output_image(resize_image(Rho_coreg, Z.shape, method='nearest'),
             'Red-disp-mag-coreg-w'+str(2*window_size)+'.jpg', cmap='magma')




#
#from sklearn import linear_model
#lr = linear_model.LinearRegression()
#lr.fit(x, y)
#ransac = linear_model.RANSACRegressor(max_trials=1000,min_samples=300)
#ransac.fit(Si[Stable].reshape(-1, 1), Shd[Stable].reshape(-1, 1))
#Sr = ransac.predict(Si.reshape(-1, 1)).reshape(Si.shape)

#plt.scatter(asp_val,asp_med,s=1,c='black',marker='.')
#plt.scatter(asp_val,asp_num,s=1,c='black',marker='.')
#IN = asp_num>np.quantile(asp_num, .5)
#plt.scatter(asp_val[IN],asp_num[IN],s=1,c='blue',marker='.')

#Sm = match_histograms(-Si[Stable],Shd[Stable])

sample_I, sample_J = get_coordinates_of_template_centers(Zn, window_size)

Slp,Asp = get_template_aspect_slope(Z,
                                    sample_I,sample_J,
                                    t_size)

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(Phi[Slp<20], Rho[Slp<20],2,match_score[Slp<20])
ax.set_rmax(40)


dI = Shd-Si #Shd + 4*Si#-L
tan_Slp = np.tan(np.radians(Slp))
dD = np.divide(dI,tan_Slp,
               out=np.zeros_like(tan_Slp), where=tan_Slp!=0)

Curv = terrain_curvature(Z)
Z_az = ridge_orientation(Z)

Ridge = Curv>1.





#plt.imshow(dI, cmap=plt.cm.RbBu)

Asp_Shd,dD_Shd = hough_sinus(np.radians(Asp[Stable]), Shd[Stable],
                             max_amp=1, sample_fraction=5000)
Asp_Si,dD_Si = hough_sinus(np.radians(Asp[Stable]), Si[Stable],
                           max_amp=1, sample_fraction=5000)


fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

plt.scatter(Curv[Ridge], Z_az[Ridge])

#di, dj = pol2cart(dD_H, Asp_H)

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp.flatten(), Si.flatten(),
           bins=90, range=[[-180, +180], [0, 1]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp[Stable], Si[Stable],
           bins=90, range=[[-180, +180], [0, 1]],
           cmap=plt.cm.gist_heat_r)
plt.show()

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.hist2d(Asp[Stable], Shd[Stable],
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)
ax1.hist2d(Asp[Stable], Si[Stable],
           bins=90, range=[[-180, +180], [-.5, .5]],
           cmap=plt.cm.gist_heat_r)
#ax1.scatter(asp_val,asp_med,s=1,c='black',marker='.')
plt.show()

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.imshow(-Si, cmap=plt.cm.bone)
ax1.imshow(Shd, cmap=plt.cm.bone)
plt.show()

fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
ax0.imshow(F, cmap=plt.cm.bone)
ax1.imshow(N, cmap=plt.cm.bone)
plt.show()

plt.hist2d(np.divide(Blue-Green,Blue+Green)[Stable],
           np.divide(Red-Near,Red+Near)[Stable],
           bins=100, range=[[-1, 1], [-1, 1]],
           cmap=plt.cm.gist_heat_r)
plt.show()

plt.hist2d(Asp[Stable], dI[Stable],
           bins=90, range=[[-180, +180], [-1, +1]],
           cmap=plt.cm.gist_heat_r)
plt.show()

plt.hist(Asp[Stable])
plt.show()

dY,dX = sample_Y-match_Y, sample_X-match_X
#plt.imshow(dY, vmin=-20, vmax=+20), plt.show()
#plt.imshow(dX, vmin=-20, vmax=+20), plt.show()

Rho, Phi = np.sqrt(dY**2,dX**2), np.arctan2(dY,dX)

dD = np.divide(Rho,np.tan(np.radians(Slp)))
dD = np.multiply(Rho,np.tan(np.radians(Slp)))

IN = dD!=0
# IN = (dX!=0) & (dY!=0)
plt.hist2d(Asp[IN], dD[IN], bins=90, range=[[-180, +180], [0, .4]], cmap=plt.cm.jet)
plt.show()

plt.scatter(Phi, np.divide(Rho,np.tan(np.radians(Slp))))
plt.show()

