"""
preprocessing of the multi-spectral imagery, doing the following:
- emphasis the illumination component, so shadows are enhanced
- co-register the imagery to common frame
"""
import os

import numpy as np

from dhdt.generic.unit_conversion import datetime2calender
from dhdt.generic.mapping_io import read_geo_image, make_geo_im
from dhdt.generic.mapping_tools import pix2map, map2pix, ref_trans, ref_update
from dhdt.generic.handler_stac import read_stac_catalog, load_bands, \
    get_items_via_id_s2, load_input_metadata_s2, load_cloud_cover_s2
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, dn2toa_s2
from dhdt.preprocessing.acquisition_geometry import \
    get_template_aspect_slope, compensate_ortho_offset
from dhdt.preprocessing.shadow_transforms import \
    apply_shadow_transform
from dhdt.preprocessing.image_transforms import log_adjustment
from dhdt.processing.matching_tools import get_coordinates_of_template_centers
from dhdt.processing.coupling_tools import match_pair
from dhdt.postprocessing.solar_tools import make_shadowing, make_shading

# initialization parameters
BBOX = [543001, 6868001, 570001, 6895001] # this is the way rasterio does it
MGRS_TILE = "09VWJ"
RESOLUTION_OF_INTEREST = 10.
SHADOW_METHOD = "entropy"
MATCH_CORRELATOR, MATCH_SUBPIX, MATCH_METRIC = "phas_only","moment","peak_entr"
MATCH_WINDOW = 2**4


DATA_DIR = os.path.join(os.getcwd(), "data") #"/project/eratosthenes/Data/"
DUMP_DIR = os.path.join(os.getcwd(), "processing")
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
STAC_L1C_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l1c-small")
STAC_L2A_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l2a-small")

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==RESOLUTION_OF_INTEREST]

# import and create general assets
dem_dat,crs,dem_aff,_ = read_geo_image(DEM_PATH)

# decrease the image size to bounding box
bbox_xy = np.array(BBOX).reshape((2,2)).T.ravel()
bbox_i, bbox_j = map2pix(dem_aff, bbox_xy[0:2], np.flip(bbox_xy[2::]))
bbox_i, bbox_j = np.round(bbox_i).astype(int), np.round(bbox_j).astype(int)
new_aff = ref_trans(dem_aff, int(bbox_i[0]), int(bbox_j[0]))
dem_dat = dem_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
new_aff = ref_update(new_aff, dem_dat.shape[0], dem_dat.shape[1])

sample_i, sample_j = get_coordinates_of_template_centers(new_aff, MATCH_WINDOW)
sample_x, sample_y = pix2map(new_aff, sample_i, sample_j)
slope_dat = get_template_aspect_slope(dem_dat, sample_i, sample_j,
                                      MATCH_WINDOW)[0]

rgi_dat = read_geo_image(RGI_PATH)[0]
rgi_dat = rgi_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
stable_dat = rgi_dat.data==0

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

def _filter_displacements(offset_x, offset_y, slope_dat, match_score,
                          max_slope=20., min_metric=2.):
    # filter integers
    IN_float = np.logical_and(np.mod(offset_x,1)!=0,
                              np.mod(offset_x,1)!=0)

    # filter steep slopes
    IN_flat = slope_dat <= max_slope

    # filter empty values
    IN_val = np.invert(np.logical_or(np.isnan(offset_x), np.isnan(offset_y)))

    # filter on matching metrics
    IN_metric = match_score >= min_metric

    pure = np.logical_and.reduce((IN_float, IN_flat, IN_val, IN_metric))
    return pure

def _coregister(
        shadow_dat, albedo_dat, shading_art, dem_dat, stable_dat,
        view_zn, view_az, shadow_aff, slope_dat):
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

    offset_x, offset_y = sample_x - match_x, sample_y - match_y
    pure = _filter_displacements(offset_x, offset_y, slope_dat, match_score)
    dx_coreg, dy_coreg = np.median(offset_x[pure]), np.median(offset_y[pure])

    shadow_ort = compensate_ortho_offset(
        shadow_dat, dem_dat, dx_coreg, dy_coreg, view_az, view_zn, shadow_aff
    )
    albedo_ort = compensate_ortho_offset(
        albedo_dat, dem_dat, dx_coreg, dy_coreg, view_az, view_zn, shadow_aff
    )
    return shadow_ort, albedo_ort

def _organize(shadow_dat, albedo_dat, shadow_art, shade_art,
              im_aff, crs, im_date):

    # get year, month, day
    year, month, day = datetime2calender(np.datetime64(im_date[:-1]))
    year, month, day = str(year), str(month), str(day)
    # create folder structure
    item_dir = os.path.join(DUMP_DIR, year, month, day)
    os.makedirs(item_dir, exist_ok=True)
    im_date_str = year + '-' + month.zfill(2) + '-' + day.zfill(2)

    # write outpu
    make_geo_im(shadow_dat.astype('float32'), im_aff, crs,
                os.path.join(item_dir, 'shadow.tif'),
                meta_descr='illumination enhanced imagery',
                date_created=im_date_str)
    make_geo_im(albedo_dat.astype('float32'), im_aff, crs,
                os.path.join(item_dir,'albedo.tif'),
                meta_descr='reflection enhanced imagery',
                date_created=im_date_str)
    make_geo_im(shadow_art.astype('float32'), im_aff, crs,
                os.path.join(item_dir, 'shadow_artificial.tif'),
                meta_descr='shadow calculated from CopernicusDEM',
                date_created=im_date_str)
    make_geo_im(shade_art.astype('float32'), im_aff, crs,
                os.path.join(item_dir, 'shading_artificial.tif'),
                meta_descr='shading calculated from CopernicusDEM',
                date_created=im_date_str)
    return

def main():
    catalog_L1C = read_stac_catalog(STAC_L1C_PATH)
    catalog_L2A = read_stac_catalog(STAC_L2A_PATH)

    for item in catalog_L1C.get_all_items():
        item_L1C, item_L2A = get_items_via_id_s2(catalog_L1C, catalog_L2A, item.id)

        # get year, month, day
        im_date = item.properties['datetime']
        year, month, day = datetime2calender(np.datetime64(im_date[:-1]))
        year, month, day = str(year), str(month), str(day)
        # create folder structure
        item_dir = os.path.join(DUMP_DIR, year, month, day)
        #if os.path.exists(item_dir): continue

        # read imagery
        bands = load_bands(item_L1C, s2_df, new_aff)
        for id, band in bands.items():
            bands[id] = dn2toa_s2(band)
        if item_L2A is not None:
            cloud_cover = load_cloud_cover_s2(item_L2A, new_aff)
        else:
            cloud_cover = np.zeros(new_aff[-2:], dtype=int)
        sun_zn, sun_az, view_zn, view_az = load_input_metadata_s2(
            item, s2_df, new_aff)
        if view_zn.ndim==3: view_zn = view_zn[...,-1]
        if view_az.ndim==3: view_az = view_az[...,-1]

        sun_zn_mean, sun_az_mean = 90-item.properties['view:sun_elevation'], \
                                   item.properties['view:sun_azimuth']
        shadow_dat, albedo_dat, shadow_art, shade_art = _compute_shadow_images(
            bands, dem_dat, sun_zn_mean, sun_az_mean)

        no_moving_dat = np.logical_and(stable_dat, cloud_cover!=9)
        shadow_ort, albedo_ort = _coregister(shadow_dat, albedo_dat, shade_art,
            dem_dat, no_moving_dat, view_zn, view_az, new_aff, slope_dat)

        im_date = item.properties['datetime']
        _organize(shadow_ort, albedo_ort, shadow_art, shade_art,
                  new_aff, crs, im_date)

if __name__ == "__main__":
    main()