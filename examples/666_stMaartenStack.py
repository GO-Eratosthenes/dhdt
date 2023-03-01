import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr

from dhdt.generic.handler_www import get_bz2_file, get_file_from_protected_www
from dhdt.generic.handler_aster import get_as_image_locations
from dhdt.generic.mapping_tools import pix2map, ll2map, get_utm_zone_simple

from dhdt.input.read_aster import list_central_wavelength_as, read_stack_as

from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.coupling_tools import match_pair

dat_url = 'https://aster.geogrid.org/ASTER/fetchL3A/'
dat_url = 'https://e4ftl01.cr.usgs.gov/ASTT/AST_L1T.003/'
dat_dir = '/Users/Alten005/ICEFLOW/FreqStack' # os.getcwd()
# dat_fol = ['1810240547361810259023', '1910110546301910129022'] # karakorum
# dat_fol = ['1906011239181906029006', '1908041239011908059001'] # iceland
dat_fol = ['2006281548062006299019', '2008311548072009019023'] # greenland
#dat_fol =['11140606421304189032', '510270552291304199003']
t_size = 2**5

#dat_fol = ['2000.11.14', '2005.10.27']
dat_hdf = ['AST_L1T_00311142000060642_20150412230241_122725.hdf',
           'AST_L1T_00310272005055229_20150511172028_80031.hdf']
as_level = 3

# ASTER has a dichroic prism

# we are only interested in the high resolution imagery
as_df = list_central_wavelength_as()
for idx, foi in enumerate(dat_fol): # loop through to see if folders of interest exist
    as_dir = os.path.join(dat_dir, foi+os.path.sep)
    if not os.path.isdir(as_dir):
        if as_level == 3:
            zip_name = 'ASTB' + foi[:12] + '.tar.bz2'
            url_full = dat_url+zip_name
            names = get_bz2_file(url_full, dump_dir=dat_dir)
        else:
            hdf_name = dat_hdf[idx]
            if not os.path.exists(os.path.join(dat_dir, hdf_name)):
                url_full = dat_url+foi+os.path.sep+hdf_name
                get_file_from_protected_www(url_full, dump_dir=dat_dir,
                                            user='icemass',
                                            password='!Q2w3e4r5t6y7u8i9o0p')
            as_dir = os.path.join(dat_dir, hdf_name)

    vn_df = as_df[np.logical_and(as_df['gsd'] == 15,
                                 np.abs(as_df['along_track_view_angle']) < 10)]
#    vn_df = as_df[np.logical_and(as_df['center_wavelength'] == 810,
#                                 np.abs(as_df['along_track_view_angle']) < 10)]

    vn_df = get_as_image_locations(as_dir, vn_df)
    if idx==0:
        I1, spatialRef1, geoTransform1, targetprj1 = read_stack_as(as_dir,
           vn_df, level=as_level)
    else:
        I2, spatialRef2, geoTransform2, targetprj2 = read_stack_as(as_dir,
           vn_df, level=as_level)
print('.')

# create grid
grd_i,grd_j = get_coordinates_of_template_centers(I1, t_size) # //2
grd_lon,grd_lat = pix2map(geoTransform1,grd_i,grd_j)

#main processing
grd_lon2,grd_lat2, match_metric = match_pair(
    I1, I2, None, None, geoTransform1, geoTransform2, grd_lon, grd_lat,
    temp_radius=t_size, search_radius=t_size,
    correlator='OC', subpix='svd', metric='phase_sup')

# 'ampl_comp' 'phas_only', 'symm_phas', 'orie_corr', 'grad_corr', 'grad_norm'
# 'gaus_phas'

Mask = np.logical_and(grd_lon2==0, grd_lat2==0)
np.putmask(grd_lon, Mask, np.nan), np.putmask(grd_lat, Mask, np.nan)
np.putmask(grd_lon2, Mask, np.nan), np.putmask(grd_lat2, Mask, np.nan)
np.putmask(match_metric, Mask, np.nan)
match_metric = np.reshape(match_metric, grd_lon.shape)

xy_1 = np.stack((grd_lat.flatten(), grd_lon.flatten()), axis=1)
xy_2 = np.stack((grd_lat2.flatten(), grd_lon2.flatten()), axis=1)

if as_level == 3:
    # find map projection
    lon_bar, lat_bar = np.nanmean(grd_lon), np.nanmean(grd_lat)

    # transform to metric
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(int(32600+get_utm_zone_simple(lon_bar)))

    xy_1 = ll2map(xy_1, proj)
    xy_2 = ll2map(xy_2, proj)

dX, dY = np.reshape(xy_1[:,0]-xy_2[:,0], grd_lon.shape), \
         np.reshape(xy_1[:,1]-xy_2[:,1], grd_lat.shape)

print('.')
plt.figure(), plt.imshow(dY-np.nanmedian(dY.flatten()),
                         cmap=plt.cm.RdBu, vmin=-30,vmax=30)

plt.figure(), plt.imshow(dX-np.nanmedian(dX.flatten()),
                         cmap=plt.cm.RdBu, vmin=-10,vmax=10)
plt.figure(), plt.imshow(np.arctan2(dX-np.nanmedian(dX.flatten()),
                                    dY-np.nanmedian(dY.flatten())),
                         cmap=plt.cm.twilight_r)
plt.figure(), plt.imshow(np.log10(np.hypot(dX-np.nanmedian(dX.flatten()),
                                  dY-np.nanmedian(dY.flatten()))),
                         cmap=plt.cm.jet, vmin=0, vmax=2.5)


