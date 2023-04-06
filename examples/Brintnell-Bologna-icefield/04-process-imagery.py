""" distill individual sun-traces within the imagery, doing the following:
1) refine shadow edge position
2) estimate refraction angles
3) construct connectivity files
"""

import os

import numpy as np
import pandas as pd
import morphsnakes as ms

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import get_mean_map_location, map2pix, \
    ref_trans, ref_update, make_same_size
from dhdt.generic.handler_stac import read_stac_catalog
from dhdt.generic.gis_tools import get_mask_boundary
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, read_mean_sun_angles_s2, read_sun_angles_s2
from dhdt.preprocessing.atmospheric_geometry import get_refraction_angle
from dhdt.preprocessing.shadow_geometry import shadow_image_to_suntrace_list
from dhdt.processing.photohypsometric_image_refinement import \
    update_casted_location
from dhdt.postprocessing.photohypsometric_tools import update_caster_elevation

BBOX = [543001, 6868001, 570001, 6895001] # this is the way rasterio does it
MGRS_TILE = "09VWJ"
RESOLUTION_OF_INTEREST = 10.
ITER_COUNT = 30

DATA_DIR = os.path.join(os.getcwd(), "data") #"/project/eratosthenes/Data/"
DUMP_DIR = os.path.join(os.getcwd(), "processing")
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')
STAC_L1C_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l1c-small")

# import and create general assets
dem_dat,crs,org_aff,_ = read_geo_image(DEM_PATH)

# decrease the image size to bounding box
bbox_xy = np.array(BBOX).reshape((2,2)).T.ravel()
bbox_i, bbox_j = map2pix(org_aff, bbox_xy[0:2], np.flip(bbox_xy[2::]))
bbox_i, bbox_j = np.round(bbox_i).astype(int), np.round(bbox_j).astype(int)
new_aff = ref_trans(org_aff, int(bbox_i[0]), int(bbox_j[0]))
dem_dat = dem_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
new_aff = ref_update(new_aff, dem_dat.shape[0], dem_dat.shape[1])

h = np.linspace(0, np.round(np.max(dem_dat) / 100) * 100 + 100, 10)
x_bar, y_bar = get_mean_map_location(new_aff, crs)

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==RESOLUTION_OF_INTEREST]
central_wavelength = s2_df['center_wavelength'].to_numpy().mean()

def _get_im_dates():
    im_dates = []
    for root,directories,_ in os.walk(DUMP_DIR):
        ymd = os.path.relpath(root, DUMP_DIR)
        if ymd.count(os.sep)==2:
            im_dates.append(ymd.replace(os.sep,'-'))
    return im_dates

def _refine_shadow_polygon(shadow_dat, albedo_dat, shadow_art):
    shadow_snk = ms.morphological_chan_vese(shadow_dat, ITER_COUNT,
        albedo=albedo_dat, init_level_set=shadow_art,
        smoothing=0, lambda1=1, lambda2=1)
    # clean up
    shadow_snk = ndimage.binary_opening(shadow_snk)
    shadow_snk = ndimage.binary_closing(shadow_snk)
    return shadow_snk

def _create_conn_pd(suntraces, new_aff, org_aff, timestamp, s2_path):
    m = suntraces.shape[0]
    sun_zn, sun_az = read_sun_angles_s2(s2_path)

    if new_aff != org_aff:
        sun_zn = make_same_size(sun_zn, org_aff, new_aff)
        sun_az = make_same_size(sun_az, org_aff, new_aff)

    i, j = map2pix(new_aff, suntraces[:,0].copy(), suntraces[:,1].copy())

    interp = RegularGridInterpolator(
        (np.arange(sun_zn.shape[0]), np.arange(sun_zn.shape[1])), sun_zn)
    angles_zn = interp(np.array([i, j]).T)
    interp = RegularGridInterpolator(
        (np.arange(sun_az.shape[0]), np.arange(sun_az.shape[1])), sun_az)
    angles_az = interp(np.array([i, j]).T)

    dh = pd.DataFrame({'timestamp': np.tile(np.datetime64(timestamp), m),
                       'caster_X': suntraces[:,0], 'caster_Y': suntraces[:,1],
                       'casted_X': suntraces[:,2], 'casted_Y': suntraces[:,3],
                       'azimuth': angles_az, 'zenith': angles_zn})
    return dh

def main():
    im_dates = _get_im_dates()
    for im_date in im_dates:
        # loop through the imagery
        stac_dir = os.path.join(STAC_L1C_PATH, im_date.replace('-', os.sep))
        stac_dir = os.path.join(stac_dir, next(os.walk(stac_dir))[1][0])
        dump_dir = os.path.join(DUMP_DIR, im_date.replace('-', os.sep))

        # load data
        shadow_dat = read_geo_image(os.path.join(dump_dir, 'shadow.tif'))[0]
        albedo_dat = read_geo_image(os.path.join(dump_dir, 'albedo.tif'))[0]
        shadow_art = read_geo_image(os.path.join(dump_dir,
                                                 'shadow_artificial.tif'))[0]

        shadow_snk = _refine_shadow_polygon(shadow_dat, albedo_dat, shadow_art)

        sun_zn_mean, sun_az_mean = read_mean_sun_angles_s2(stac_dir)
        suntraces = shadow_image_to_suntrace_list(shadow_snk, new_aff,
                                                  sun_az_mean)
        dh = _create_conn_pd(suntraces, new_aff, org_aff, im_date, stac_dir)

        # refine shadow edge position
        print('refine')
        dh = (dh
              .pipe(update_casted_location, shadow_dat, shadow_snk, new_aff)
              .pipe(update_caster_elevation, dem_dat, new_aff)
              .pipe(get_refraction_angle, x_bar, y_bar, crs,
                    central_wavelength, h)
              )

        # construct connectivity files
        print('.')

if __name__ == "__main__":
    main()