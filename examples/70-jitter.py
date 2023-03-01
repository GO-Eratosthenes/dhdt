import os
import sys
import glob
import numpy as np
from tqdm import tqdm

from osgeo import gdal

from dhdt.generic.attitude_tools import quat_2_euler, ecef2ocs
from dhdt.generic.orbit_tools import transform_rpy_xyz2ocs
from dhdt.generic.mapping_io import read_geo_image, make_geo_im, read_geo_info
from dhdt.generic.mapping_tools import pol2xyz, pix2map
from dhdt.generic.handler_sentinel2 import get_s2_dict
from dhdt.generic.handler_im import rotated_sobel, conv_2Dfilter
from dhdt.auxilary.handler_copernicusdem import make_copDEM_mgrs_tile
from dhdt.input.read_sentinel2 import \
    read_sensing_time_s2, get_flight_bearing_from_gnss_s2, \
    read_detector_time_s2, read_detector_mask, get_flight_orientation_s2, \
    read_stack_s2, get_flight_path_s2, read_view_angles_s2, \
    get_timing_mask
from dhdt.preprocessing.image_transforms import gamma_adjustment, mat_to_gray
from dhdt.processing.network_tools import get_network_indices
from dhdt.processing.matching_tools import prepare_grids, \
    get_coordinates_of_template_centers, get_value_at_template_centers
from dhdt.processing.matching_tools_frequency_filters import perdecomp
from dhdt.processing.matching_tools_frequency_correlators import \
    amplitude_comp_corr, robust_corr
from dhdt.processing.matching_tools_spatial_subpixel import \
    get_top_centroid, get_top_esinc
from dhdt.processing.matching_tools_frequency_subpixel import \
    phase_svd, phase_pca
from dhdt.input.read_sentinel2 import list_central_wavelength_msi
from dhdt.generic.handler_sentinel2 import \
    get_s2_image_locations, get_s2_granule_id, get_s2_dict
from dhdt.presentation.image_io import output_image

ds = 2**5

dat_dir = '/Users/Alten005/jitter'

#import morphsnakes as ms
#ms.morphological_chan_vese
sname = 'S2A_MSIL1C_20220616T103631_N0400_R008_T31UFT_20220616T173756.SAFE' # the netherlands
#sname = 'S2A_MSIL1C_20220821T143741_N0400_R096_T19KFT_20220821T194815.SAFE' # saltflat bolivia
#sname = 'S2A_MSIL1C_20220819T083611_N0400_R064_T33JWM_20220819T140121.SAFE'
fname = 'MTD_MSIL1C.xml'
full_path = os.path.join(dat_dir, sname, fname)
s2_df = list_central_wavelength_msi()
resolution_of_interest = 10
boi_df = s2_df['gsd']==resolution_of_interest
s2_df = s2_df[boi_df]

s2_df, datastrip_id = get_s2_image_locations(full_path,s2_df)
s2_dict = get_s2_dict(s2_df)

# look at bearings: s2_dict.keys()
s2_dict = get_flight_path_s2(s2_dict['MTD_DS_path'], s2_dict=s2_dict)
s2_dict = get_flight_orientation_s2(s2_dict['MTD_DS_path'], s2_dict=s2_dict)

from dhdt.generic.orbit_tools import estimate_inclination_via_xyz_uvw

#psi = estimate_inclination_via_xyz_uvw(s2_dict['gps_xyz'], s2_dict['gps_uvw'])

R_ocs = ecef2ocs(s2_dict['gps_xyz'][0,:], s2_dict['gps_uvw'][0,:])

s2_quat = s2_dict['imu_quat']
roll,pitch,yaw = quat_2_euler(s2_quat[:,0], s2_quat[:,1], s2_quat[:,2],
                              s2_quat[:,3])

imu_t = (s2_dict['imu_time'] - np.datetime64(s2_dict['date'][1:])
         ).astype('timedelta64[ms]').astype(float)
gps_t = (s2_dict['gps_tim'] - np.datetime64(s2_dict['date'][1:])
         ).astype('timedelta64[ms]').astype(float)

r,p,y = transform_rpy_xyz2ocs(s2_dict['gps_xyz'], s2_dict['gps_uvw'],
                              roll, pitch, yaw, gps_t, imu_t)

timestamp = (s2_dict['imu_time'] -
             np.datetime64(s2_dict['date'][1:])).astype('timedelta64[ms]').astype(float)
f_r = np.poly1d(np.polyfit(timestamp, roll, deg=3))
f_y = np.poly1d(np.polyfit(timestamp, yaw, deg=3))
f_p = np.poly1d(np.polyfit(timestamp, pitch, deg=3))

f_r2 = np.poly1d(np.polyfit(timestamp, roll-f_r(timestamp), deg=3))
f_y2 = np.poly1d(np.polyfit(timestamp, yaw-f_y(timestamp), deg=3))
f_p2 = np.poly1d(np.polyfit(timestamp, pitch-f_p(timestamp), deg=3))

# look at view angles
from dhdt.input.read_sentinel2 import get_view_angles_s2_from_root, \
    get_crs_s2_from_root, read_detector_mask, \
    get_geotransform_s2_from_root
from dhdt.generic.handler_xml import get_root_of_table

root = get_root_of_table(s2_dict['MTD_TL_path'], 'MTD_TL.xml')
Az_grd, Zn_grd, bnd, det, grdTransform = get_view_angles_s2_from_root(root)
crs = get_crs_s2_from_root(root)

from dhdt.generic.handler_sentinel2 import list_platform_metadata_s2a

from dhdt.generic.orbit_tools import calculate_correct_mapping, \
    remap_observation_angles, get_absolute_timing
# for debugging: sort so first is first...
idx = np.lexsort((det, bnd))
# restructure
bnd, det = bnd[idx], det[idx]
bnd_list = np.asarray(s2_df['bandid']) - 1
Az_grd, Zn_grd  = Az_grd[...,idx], Zn_grd[...,idx]

geoTransform = get_geotransform_s2_from_root(root,
                                             resolution=resolution_of_interest)
det_stack = read_detector_mask(s2_dict['QI_DATA_path'], s2_df, geoTransform)

# create matching grid
tsize = 32
I_grd,J_grd = get_coordinates_of_template_centers(geoTransform, tsize)
det_grd = get_value_at_template_centers(det_stack, tsize)
X_grd,Y_grd = pix2map(geoTransform, I_grd, J_grd)


Ltime, lat, lon, radius, inclination, period, time_para, combos = \
    calculate_correct_mapping(Zn_grd, Az_grd, bnd, det, grdTransform, crs,
                              sat_dict=s2_dict)

Zn, Az, T = remap_observation_angles(Ltime, lat, lon, radius, inclination, period,
                             time_para, combos, X_grd, Y_grd, det_grd,
                             bnd_list, geoTransform, crs)

T_bias = get_absolute_timing(lat,lon,s2_dict)
T_utc = (T*1E9).astype('timedelta64[ns]')+T_bias

cop_dem_mgrs = os.path.join(dat_dir, 'COP-DEM-'+s2_dict['tile_code']+'.tif')
Z, spatialRef, geoTransform, targetprj = read_geo_image(cop_dem_mgrs)

from dhdt.generic.mapping_io import make_nc_image

make_nc_image(Z+1j*Z, geoTransform,spatialRef, 'test.nc')
#from scipy.interpolate import CubicSpline
#s_r = CubicSpline(dummy, roll-f_r(dummy))
from scipy import signal

# performing an spectral analysis on the PC-timeseries
f, Pxx = signal.periodogram(roll-f_r(timestamp))

import matplotlib.pyplot as plt
plt.figure(), plt.plot(f,Pxx), plt.yscale("log")

plt.figure()
plt.plot((timestamp-timestamp[0])/1E3, roll-f_r(timestamp))





plt.figure()
plt.plot(s2_dict['imu_time'], roll-f_r(dummy))
plt.plot(s2_dict['imu_time'], s_r(dummy))

plt.plot(s2_dict['imu_time'], yaw-f_y(dummy))
plt.plot(s2_dict['imu_time'], pitch-f_p(dummy))
plt.plot(s2_dict['gps_tim'], s2_dict['gps_err'][:,0]/1E2)
plt.plot(s2_dict['gps_tim'], s2_dict['gps_err'][:,1]/1E2)
plt.plot(s2_dict['gps_tim'], s2_dict['gps_err'][:,2]/1E2)

print('create detector angles')
spatialRef, geoTransform, targetprj,rows, cols,_= read_geo_info(
    s2_df['filepath'][1]+'.jp2')
dT, Across, Phi, s2_dict = get_timing_mask(s2_df, geoTransform, spatialRef)

# reduce to matching grids
dT = dT[ds//2:-ds//2:ds, ds//2:-ds//2:ds,:]
Across = Across[ds//2:-ds//2:ds, ds//2:-ds//2:ds]
Phi = Phi[ds//2:-ds//2:ds, ds//2:-ds//2:ds,:]

boi = ['red']
det_df = list_central_wavelength_msi()
boi_df = det_df[det_df['common_name'].isin(boi)]
d_grd = read_detector_mask(s2_dict['QI_DATA_path'], boi_df, geoTransform)
d_grd = np.squeeze(d_grd)

# get DEM
print('collect DEM')
dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM_GLO-30/'

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM_GLO-30/'
s2_path = '/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles'

sso, pw = 'basalt', '!Q2w3e4r5t6y7u8i9o0p'

cop_dem_mgrs = os.path.join(dat_dir, 'COP-DEM-'+s2_dict['tile_code']+'.tif')
if not os.path.exists(cop_dem_mgrs):
    cop_dem_mgrs = make_copDEM_mgrs_tile(s2_dict['tile_code'],
         os.path.dirname(s2_df['filepath'][0]),
         s2_df['filepath'][0].split('/')[-1]+'.jp2',
         s2_path, dem_path,
         tile_name='sentinel2_tiles_world.shp', cop_name='mapping.csv',
         map_name='DGED-30.geojson', sso=sso, pw=pw, out_path=dat_dir)

couple_list = get_network_indices(s2_df.shape[0]).T

im_stack = read_stack_s2(s2_df)[0]
im_stack = mat_to_gray(im_stack, im_stack==0)
im_stack, I_grd, J_grd = prepare_grids(im_stack, ds)

shape_grd = I_grd.shape
shape_mtc = shape_grd + (couple_list.shape[0],)
#I_grd, J_grd = I_grd.flatten(), J_grd.flatten()
Ddi, Ddj = np.zeros(shape_mtc, dtype=float), \
           np.zeros(shape_mtc, dtype=float)
print('start matching')

for i,j in np.ndindex(I_grd.shape):
    i_b, i_e = I_grd[i,j], I_grd[i,j]+ds # row begining and ending
    j_b, j_e = J_grd[i,j], J_grd[i,j]+ds # collumn begining and ending

    sub_stack = im_stack[i_b:i_e,j_b:j_e,:]
    if np.ptp(sub_stack)!=0:
        for k, couple in enumerate(couple_list):
            I1 = perdecomp(np.squeeze(sub_stack[:,:,couple[0]]))[0]
            I2 = perdecomp(np.squeeze(sub_stack[:,:,couple[1]]))[0]
            Q12 = amplitude_comp_corr(I1, I2)
            #Q12 = robust_corr(I1,I2)
            ddi,ddj = phase_pca(Q12)
            #C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
            #ddi,ddj, i_int,j_int = get_top_centroid(C)
            #ddi,ddj, i_int,j_int = get_top_esinc(C)

            # do allocation
            Ddi[i,j,k], Ddj[i,j,k] = ddi, ddj

import matplotlib.pyplot as plt

Z = read_geo_image(cop_dem_mgrs)[0]

print('.')
det_grd = d_grd[ds//2:-ds//2:ds, ds//2:-ds//2:ds].copy()
#jit_grd = o_grd[ds//2:-ds//2:ds, ds//2:-ds//2:ds].copy()
dem_grd = Z[ds//2:-ds//2:ds, ds//2:-ds//2:ds].copy()
joi = 3000
#IN = np.isclose(jit_grd-joi,0,atol=ds)

print('.')

IN = np.isclose(dT[:,:,1],-2.5,atol=.03)

IN = np.isclose(dT[:,:,1],-8.5,atol=1)
psi = Ddi[:,:,4].copy()
#psi = np.arctan2(Ddi[:,:,4], Ddj[:,:,4])
np.putmask(psi,np.invert(IN), 0)
plt.figure(), plt.imshow(psi, vmin=-.2, vmax=+.2)
#plt.figure(), plt.imshow(psi, cmap=plt.cm.twilight)

#d_along = rotated_sobel(psi,3,indexing='xy')
#slp_grd = conv_2Dfilter(dem_grd, d_along)
#slp_grd = np.pad(slp_grd, (1, 1), 'constant', constant_values=(0,0))

#import pywt
#import sys
#sys.path.insert(1, dat_dir)
#from smoothn import smoothn

#I = np.squeeze(Ddj[50:110,90:150,5].copy())
I = np.squeeze(Ddj[:,:,-1].copy())

from scipy import ndimage, fftpack
from dhdt.processing.matching_tools_frequency_filters import perc_masking

I = np.squeeze(Ddj[:,:,-1].copy()) + \
    1j*np.squeeze(Ddi[:,:,-1].copy())
I = Ddj + 1j*Ddi

np.putmask(I,np.abs(I)>.3, 0)
np.putmask(I,np.isnan(I), 0)
np.putmask(I,np.mod(det_grd,2)==0, 0)

I_c = fftpack.dctn(I)
M = perc_masking(I_c, m=.95, s=10)
I_f = fftpack.idctn(M*I_c)


#Im = np.ma.array(I, mask=np.logical_and(np.mod(det_grd,2)==0, np.abs(I)>=.3))
Msk = ~np.logical_and(np.mod(det_grd,2)==1, np.abs(I)<=.3)
Im = np.ma.array(I, mask=Msk)

#Z,W = smoothn(Im, s=0.5, rbst=True)


#coeffs = pywt.swtn(I,'db1', level=2)
#I_s = pywt.iswtn(coeffs, 'db1')

#coeffs = pywt.swtn(np.squeeze(Ddj[1:,1:,5]),'db1', level=2)
#plt.figure(), plt.imshow(np.real(coeffs[0]['aa']+coeffs[0]['da']))

#plt.figure()
#for i in range(couple_list.shape[0]):
#    slice = Ddi[:,:,i]
#    plt.plot(slice[IN])

#for i in range(couple_list.shape[0]):
#    plt.figure(), plt.imshow(Ddi[:,:,i], vmin=-.2, vmax=.2)
#plt.figure(), plt.imshow(slp_grd)

ddd = Ddi.copy()
ddd[np.mod(det_grd,2)!=1] = 0

plt.figure(), \
plt.imshow(ddd, vmin=-.2, vmax=.2), plt.colorbar()

#along = -np.cos(np.deg2rad(psi))*Ddj - np.sin(np.deg2rad(psi))*Ddi
#acros = np.sin(np.deg2rad(psi))*Ddj + np.cos(np.deg2rad(psi))*Ddi



# do matching

# get satellite frame

# estimate jitter
