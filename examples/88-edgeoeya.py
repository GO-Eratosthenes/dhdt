import os
import numpy as np
import matplotlib.pyplot as plt

from osgeo import osr, ogr, gdal

from dhdt.generic.mapping_io import \
    read_geo_image, read_geo_info, make_geo_im
from dhdt.generic.mapping_tools import ll2map, pix_centers, map2pix
from dhdt.generic.handler_im import bilinear_interpolation
from dhdt.generic.gis_tools import shape2raster
from dhdt.input.read_sentinel2 import get_flight_bearing_from_detector_mask_s2
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope
from dhdt.preprocessing.image_transforms import mat_to_gray, gamma_adjustment
from dhdt.processing.matching_tools_differential import hough_sinus

from dhdt.input.read_sentinel2 import read_band_s2
from dhdt.presentation.image_io import output_image


S0_dir = '/Users/Alten005/SEES/Data/DEM/2008'
S0_file = 'S0_DTM5_2010_13920_35.tif'
S0_foi = 'S0_AST_2010_13920_35.tif'
as_dir = '/Users/Alten005/SEES/Doc/Svalbard/2107071148302107089005'
as_file = 'data1.l3a.demzs.tif'
rgi_dir = '/Users/Alten005/surfdrive/Eratosthenes/Denali/GIS'
rgi_file = '07_rgi60_Svalbard.shp'
rgi_foi = 'rgi60_Svalbard.tif'
map_dir = '/Users/Alten005/SEES/Data'
map_file = 'S100_Raster_10m.jp2'
map_foi = 'S100_Raster_aster'

# make grid
spatialRef, geoTransform, targetprj, rows, cols, bands =\
    read_geo_info(os.path.join(as_dir, as_file))
Lon,Lat = pix_centers(geoTransform, rows=rows, cols=cols, make_grid=True)

if not os.path.exists(os.path.join(S0_dir, S0_foi)):
    utmRef, utmTransform,_,_,_,_ =\
        read_geo_info(os.path.join(S0_dir, S0_file))

    # transform to UTM
    XY = ll2map(np.stack((Lat.flatten(), Lon.flatten())).T, utmRef)
    X,Y = np.reshape(XY[:,0], (rows, cols)), np.reshape(XY[:,1], (rows, cols))
    del XY
    I,J = map2pix(utmTransform, X, Y)

    Z_s0 = read_geo_image(os.path.join(S0_dir, S0_file))[0]
    Z_10 = bilinear_interpolation(Z_s0, I, J)
    del Z_s0
    make_geo_im(Z_10, geoTransform, spatialRef, os.path.join(S0_dir, S0_foi))
else:
    Z_10 = read_geo_image(os.path.join(S0_dir, S0_foi))[0]

if not os.path.exists(os.path.join(as_dir, map_foi)):
    mapRef, mapTransform,_,_,_,_ =\
        read_geo_info(os.path.join(map_dir, map_file))

    # transform to UTM
    XY = ll2map(np.stack((Lat.flatten(), Lon.flatten())).T, mapRef)
    X,Y = np.reshape(XY[:,0], (rows, cols)), np.reshape(XY[:,1], (rows, cols))
    del XY
    I,J = map2pix(mapTransform, X, Y)

    RGB_s0 = read_geo_image(os.path.join(map_dir, map_file))[0]
    rgb_s0 = np.zeros((I.shape[0], I.shape[1], 3))
    for i in range(3):
        rgb_s0[...,i] = bilinear_interpolation(RGB_s0[...,i], I, J)
    del RGB_s0
    rgb_s0 = rgb_s0.astype(int)
    make_geo_im(rgb_s0.astype(int), geoTransform, spatialRef,
                os.path.join(as_dir, map_foi))
else:
    rgb_s0 = read_geo_image(os.path.join(as_dir, map_foi))[0]


(Z_as, spatialRefZ, geoTransformZ, targetprjZ) = \
    read_geo_image(os.path.join(as_dir, as_file))
Z_as = Z_as.astype(float)
np.putmask(Z_as, Z_as==-9999, np.NaN)

# get map

# fix Randolph Glacier inventory
if not os.path.exists(os.path.join(as_dir, rgi_foi)):

    shape2raster(os.path.join(rgi_dir, rgi_file),
                 os.path.join(as_dir, rgi_foi),
                 geoTransform, rows, cols, spatialRef)

RGI = read_geo_image(os.path.join(as_dir, rgi_foi))[0]

Msk = np.logical_or(np.logical_or(np.isnan(Z_10), Z_10<=0),
                    np.isnan(Z_as))

# get grids
dZ = Z_as - Z_10
np.putmask(dZ, Msk, np.NaN)

dZ_ter = dZ.copy()
np.putmask(dZ_ter, RGI, np.NaN)

grdI,grdJ = np.meshgrid(np.linspace(0,Z_10.shape[0]-1,Z_10.shape[0]),
                        np.linspace(0,Z_10.shape[1]-1,Z_10.shape[1]),
                        indexing='ij')
grdI,grdJ = grdI.astype(int), grdJ.astype(int)
tsize=3
Slp,Asp = get_template_aspect_slope(Z_10,grdI,grdJ,tsize,spac=30.)

## look at along-track bias
# create grid
az = get_flight_bearing_from_detector_mask_s2(np.invert(np.isnan(Z_as)))
grdAlong = np.cos(np.deg2rad(az))*grdJ + np.sin(np.deg2rad(az))*grdI
np.putmask(grdAlong, Msk, np.NaN)
grdAlong = np.round(4*grdAlong)

dZ_med,Z_new = np.zeros_like(dZ), Z_as.copy()
iter = 3
# do estimation
for counter in range(iter):
    for id in np.unique(grdAlong):
        if not np.isnan(id):
            IN = grdAlong==id
            med_line = np.median(dZ[IN])
    #        med_line = np.median(dZ_ter[IN])
            np.putmask(dZ_med, IN, med_line)

    dZ_corr = dZ-dZ_med

    #make_geo_im(dZ_corr, geoTransformZ, spatialRefZ,
    #            os.path.join(as_dir, 'aster_corr.tif'))

    v_G = np.divide(dZ_corr, np.cos(np.deg2rad(Slp)))

    Ter = np.logical_or(Msk, RGI)
#    plt.figure()
#    plt.hexbin(Asp[~Ter], v_G[~Ter], extent=(-180,+180,-20,+20),
#               gridsize=(48,24), bins=20, cmap=plt.cm.twilight)
#    plt.show()

    phi_h, rho_h, score_h = hough_sinus(Asp[~Ter], v_G[~Ter],
                                        max_amp=20,num_estimates=1)

    # alternating .... iterate, by translation

    dZ = Z_new - Z_10


plt.figure()
plt.hexbin(Asp[~Msk], v_G[~Msk], extent=(-180,+180,-20,+20),
           gridsize=(48,24), bins=10)
plt.show()

plt.figure()
plt.imshow(dZ, vmin=-50, vmax=0, cmap=plt.cm.RdBu)
plt.show()


#CCI_dir = '/Users/Alten005/SEES/Pres/tazio'
#old_name = 'Glaciers_CCI_IV_RGI07_Svalbard_JERS1_19930710_19980326.tif'
#new_name = 'Glaciers_CCI_IV_RGI07_Svalbard_SENT1_20210120_20210212.tif'

#V_old = read_geo_image(os.path.join(CCI_dir, old_name))[0]
#V_new = read_geo_image(os.path.join(CCI_dir, new_name))[0]

#np.putmask(V_old, V_old==0, np.nan)
#np.putmask(V_new, V_new==0, np.nan)
#np.putmask(V_old, V_old>1500, 1500.)
#np.putmask(V_new, V_new>1500, 1500.)

#outputname = 'JERS1_19930710_19980326.png'
#output_image(V_old[1001:,:4000], os.path.join(CCI_dir, outputname), cmap='jet', compress=95)
#outputname = 'SENT1_20210120_20210212.png'
#output_image(V_new[1001:,:4000], os.path.join(CCI_dir, outputname), cmap='jet', compress=95)

#s2_dir = '/Users/Alten005/SEES/Pres/S2'
#s2_name = 'T33XXG_20200803T113651_B'
#Blue = read_band_s2(os.path.join(s2_dir, s2_name+'02.jp2'))[0]
#Green = read_band_s2(os.path.join(s2_dir, s2_name+'03.jp2'))[0]
#Red = read_band_s2(os.path.join(s2_dir, s2_name+'04.jp2'))[0]
#Red, Green, Blue = Red[7001:9000,3001:7000], Green[7001:9000,3001:7000], \
#                   Blue[7001:9000,3001:7000]
#RGB = np.dstack((mat_to_gray(Red), mat_to_gray(Green), mat_to_gray(Blue)))
#RGB = gamma_adjustment(RGB, 0.5)
#output_image(RGB, os.path.join(s2_dir, 'S2_edgeoeya.jpg'), compress=95)
