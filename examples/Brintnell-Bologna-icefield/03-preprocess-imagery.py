"""
preprocessing of the multi-spectral imagery, doing the following:
- emphasis the illumination component, so shadows are enhanced
- co-register the imagery to common frame
"""
import os
import pystac

import numpy as np

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import pix2map, map2pix, ref_trans, ref_update
from dhdt.generic.handler_stac import read_stac_catalog, load_bands, \
    get_items_via_id_s2, load_input_metadata_s2
from dhdt.generic.handler_sentinel2 import \
    get_s2_image_locations, get_s2_dict
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, read_sun_angles_s2
from dhdt.preprocessing.handler_multispec import \
    get_shadow_bands,read_shadow_bands
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope, compensate_ortho_offset
from dhdt.preprocessing.shadow_transforms import \
    apply_shadow_transform
from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.coupling_tools import match_pair
from dhdt.postprocessing.solar_tools import make_shadowing, make_shading

# initialization parameters
MGRS_TILE = "09VWJ"
RESOLUTION_OF_INTEREST = 10.
SHADOW_METHOD = "entropy"
MATCH_CORRELATOR, MATCH_SUBPIX, MATCH_METRIC = "phas_only","moment","peak_entr"
MATCH_WINDOW = 2**4
BBOX = [543001, 6868001, 570001, 6895001] # this is the way rasterio does it


DATA_DIR = os.path.join(os.getcwd(), 'data') #"/project/eratosthenes/Data/"
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
STAC_L1C_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l1c-small")
STAC_L2A_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l2a-small")

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==RESOLUTION_OF_INTEREST]

# import and create general assets
dem_dat,_,dem_aff,_ = read_geo_image(DEM_PATH)

# decrease the image size to bounding box
bbox_xy = np.array(BBOX).reshape((2,2)).T.ravel()
bbox_i, bbox_j = map2pix(dem_aff, bbox_xy[0:2], np.flip(bbox_xy[2::]))
bbox_i, bbox_j = np.round(bbox_i).astype(int), np.round(bbox_j).astype(int)
new_aff = ref_trans(dem_aff, int(bbox_i[0]), int(bbox_j[0]))
dem_dat = dem_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
new_aff = ref_update(new_aff, dem_dat.shape[0], dem_dat.shape[1])

sample_i, sample_j = get_coordinates_of_template_centers(new_aff, MATCH_WINDOW)
sample_x, sample_y = pix2map(new_aff, sample_i, sample_j)

rgi_dat = read_geo_image(RGI_PATH)[0]
rgi_dat = rgi_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
stable_dat = rgi_dat==0

def _compute_shadow_images(
        bands, dem_dat, sun_zn_mean, sun_az_mean
):
    """
    Compute shadow-enhanced images and DEM-based artificial images

    Parameters
    ----------
    bands: dictionary with numpy.ndarrays
        multi-spectral bands
    dem_dat : numpy.ndarray, size=(m,n)
        rasterized digital elevation model
    sun_zn_mean : float, unit=degrees
        mean zenith sun angle
    sun_az_mean : float, unit=degrees
        mean azimuth sun angle

    Returns
    -------
    shadow_dat, albedo_dat : numpy.ndarray, size=(m,n)
        estimate of the illumination and reflection component
    shadow_art, shading_art : numpy.ndarray, size=(m,n), range=0..1
        artificial shadowing and shading based on the digital elevation model
    """

    # compute shadow-enhanced image
    shadow_dat, albedo_dat = apply_shadow_transform(SHADOW_METHOD,
        bands["blue"], bands["green"], bands["red"], [], bands["nir"], []
    )
    # construct artificial images
    shadow_art = make_shadowing(dem_dat, sun_az_mean, sun_zn_mean)
    shading_art = make_shading(dem_dat, sun_az_mean, sun_zn_mean)
    return shadow_dat, albedo_dat, shadow_art, shading_art


def _coregister(
        shadow_dat, albedo_dat, shading_art, dem_dat, stable_dat,
        view_zn, view_az, shadow_aff):
    """
    Co-register shadow and albedo rasters using an artificial shading image as
    a target

    Parameters
    ----------
    shadow_dat, albedo_dat : numpy.ndarray, size=(m,n)
        estimate of the illumination and reflection component
    shading_art : numpy.ndarray, size=(m,n), range=0..1
        artificial shading based on the digital elevation model
    dem_dat : numpy.ndarray, size=(m,n)
        rasterized digital elevation model
    stable_dat : numpy.ndarray, size=(m,n)
        mask for stable terrain
    view_zn, view_az : numpy.ndarray, size=(m,n), unit=float
        observation angles
    shadow_aff: tuple
        geo-transform

    Returns
    -------
    shadow_ort, albedo_ort : numpy.ndarray, size=(m,n)
        co-registered illumination and reflection images
    """

    sample_i, sample_j = get_coordinates_of_template_centers(dem_dat,
                                                             MATCH_WINDOW)
    sample_x, sample_y, = pix2map(shadow_aff, sample_i, sample_j)

    match_x, match_y, match_score = match_pair(shading_art, shadow_dat,
       stable_dat, stable_dat, shadow_aff, shadow_aff,
       sample_x, sample_y,
       temp_radius=MATCH_WINDOW, search_radius=MATCH_WINDOW,
       correlator=MATCH_CORRELATOR, subpix=MATCH_SUBPIX, metric=MATCH_METRIC)

    offset_y, offset_x = sample_y - match_y, sample_x - match_x

    slope, aspect = get_template_aspect_slope(dem_dat, sample_i, sample_j,
                                              MATCH_WINDOW)

    pure = np.logical_and(~np.isnan(offset_x), slope < 20)
    dx_coreg, dy_coreg = np.median(offset_x[pure]), np.median(offset_y[pure])

    shadow_ort = compensate_ortho_offset(
        shadow_dat, dem_dat, dx_coreg, dy_coreg, view_az, view_zn, dem_aff
    )
    albedo_ort = compensate_ortho_offset(
        albedo_dat, dem_dat, dx_coreg, dy_coreg, view_az, view_zn, dem_aff
    )
    return shadow_ort, albedo_ort

catalog_L1C = read_stac_catalog(STAC_L1C_PATH)
catalog_L2A = read_stac_catalog(STAC_L2A_PATH)

for item in catalog_L1C.get_all_items():
    item_L1C, item_L2A = get_items_via_id_s2(catalog_L1C, catalog_L2A, item.id)

    # read imagery
    sun_zn, sun_az, view_zn, view_az = load_input_metadata_s2(item, s2_df, new_aff)

    bands, cloud_cover = load_bands(item, s2_df)
    sun_zn_mean, sun_az_mean = 90-item.properties['view:sun_elevation'], \
                               item.properties['view:sun_azimuth']
    shadow_dat, albedo_dat, shadow_art, shading_art = _compute_shadow_images(
        bands, dem_dat, sun_zn_mean, sun_az_mean)
    print('.')

    shadow_ort, albedo_ort = _coregister(shadow_dat, albedo_dat, shade_art,
        dem_dat, stable_dat, view_zn, view_az, dem_aff)

    im_time = item.properties['datetime']
    sun_angles = [sun_zn_mean, sun_az_mean]
# meta_descr='illumination enhanced imagery'
