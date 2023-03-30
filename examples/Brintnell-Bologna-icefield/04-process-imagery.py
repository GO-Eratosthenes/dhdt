""" distill individual sun-traces within the imagery, doing the following:
1) find the shadow polygons in the imagery
2) refine shadow edge position
3) estimate refraction angles
4) construct connectivity files
"""

import os

import numpy as np
import morphsnakes as ms

from scipy import ndimage

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import get_mean_map_location
from dhdt.generic.gis_tools import get_mask_boundary
from dhdt.input.read_sentinel2 import \
    list_central_wavelength_msi, read_sun_angles_s2
from dhdt.preprocessing.atmospheric_geometry import get_refraction_angle
from dhdt.preprocessing.shadow_geometry import shadow_image_to_suntrace_list
from dhdt.processing.photohypsometric_image_refinement import \
    update_casted_location
from dhdt.postprocessing.solar_tools import make_shadowing

MGRS_TILE = "09VWJ"
RESOLUTION_OF_INTEREST = 10.
ITER_COUNT = 30

DATA_DIR = os.path.join(os.getcwd(), 'data') #"/project/eratosthenes/Data/"
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')


# import and create general assets
dem_dat,crs,dem_aff,_ = read_geo_image(DEM_PATH)[0]
h = np.linspace(0, np.round(np.max(dem_dat) / 100) * 100 + 100, 10)
x_bar, y_bar = get_mean_map_location(dem_aff, crs)

s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['gsd']==RESOLUTION_OF_INTEREST]
central_wavelength = s2_df['center_wavelength'].to_numpy().mean()



# loop through the imagery

#1) find the shadow polygons in the imagery
sun_zn, sun_az = read_sun_angles_s2(im_dict['MTD_TL_path'])
shw_dat = make_shadowing(dem_dat, sun_az, sun_zn)

ill_dat,_,ill_aff,_ = read_geo_image()
alb_dat = read_geo_image()[0]

# refine shadow polygon
ill_snk = ms.morphological_chan_vese(ill_dat, ITER_COUNT,
                                     albedo=alb_dat, init_level_set=shw_dat,
                                     smoothing=0, lambda1=1, lambda2=1)
# clean up
ill_snk = ndimage.binary_opening(ill_snk)
ill_snk = ndimage.binary_closing(ill_snk)

suntrace_list = shadow_image_to_suntrace_list(ill_snk, ill_aff, sun_az)


#2) refine shadow edge position
dh = (dh
      .pipe(start_pipeline)
      .pipe(update_casted_location, ill_dat, ill_snk, ill_aff)
      )

#3) estimate refraction angles
dh = (dh
      .pipe(start_pipeline)
      .pipe(get_refraction_angle, x_bar, y_bar, crs,
            central_wavelength, h)
      )

#4) construct connectivity files
