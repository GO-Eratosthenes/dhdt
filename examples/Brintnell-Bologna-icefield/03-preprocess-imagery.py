"""
preprocessing of the multi-spectral imagery, doing the following:
- emphasis the illumination component, so shadows are enhanced
- co-register the imagery to common frame
"""
import os

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import pix2map
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
from dhdt.postprocessing.solar_tools import make_shading

# initialization parameters
MGRS_TILE = "09VWJ"
RESOLUTION_OF_INTEREST = 10.
SHADOW_METHOD = "entropy"
MATCH_CORRELATOR, MATCH_SUBPIX, MATCH_METRIC = "phas_only","moment","peak_entr"
MATCH_WINDOW = 2**4


DATA_DIR = os.path.join(os.getcwd(), 'data') #"/project/eratosthenes/Data/"
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')


s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==RESOLUTION_OF_INTEREST]
bnd_list = get_shadow_bands

# import and create general assets
dem_dat,_,dem_aff,_ = read_geo_image(DEM_PATH)[0]
sample_I, sample_J = get_coordinates_of_template_centers(dem_dat, MATCH_WINDOW)
sample_X, sample_Y = pix2map(dem_aff, sample_I, sample_J)

rgi_dat,_,rgi_aff,_ = read_geo_image(RGI_PATH)[0]
stb_dat = rgi_dat==0

# loop over assets...

# read imagery
im_df, datastrip_id = get_s2_image_locations(im_path,s2_df)
im_dict = get_s2_dict(im_df)
im_blue, im_green, im_red, im_near,_,im_aff,_,_ = read_shadow_bands()

#
ill_dat,alb_dat = apply_shadow_transform(SHADOW_METHOD, im_blue, im_green,
                                         im_red, [], im_near, [])


# create artificial shading
sun_zn, sun_az = read_sun_angles_s2(im_dict['MTD_TL_path'])
shd_dat = make_shading(dem_dat, sun_az, sun_zn)

# matching
match_X, match_Y, match_score = match_pair(shd_dat, ill_dat, stb_dat, stb_dat,
                                           dem_aff, im_aff,
                                           sample_X, sample_Y,
                                           temp_radius=MATCH_WINDOW,
                                           search_radius=MATCH_WINDOW,
                                           correlator=MATCH_CORRELATOR,
                                           subpix=MATCH_SUBPIX,
                                           metric=MATCH_METRIC)
dY, dX = sample_Y - match_Y, sample_X - match_X

IN = np.logical_and(~np.isnan(dX), match_score>=2.5)


Slp, Asp = get_template_aspect_slope(Z,
                                     sample_I, sample_J,
                                     window_size)
PURE = np.logical_and(IN, Slp<20)
dx_coreg, dy_coreg = np.median(dX[PURE]), np.median(dY[PURE])


# get time=s2_dict['date']
# get sun_angles = [65.34, 23.4]
# meta_descr='illumination enhanced imagery'
