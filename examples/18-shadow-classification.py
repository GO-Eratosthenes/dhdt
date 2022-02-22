import os

import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.filtering_statistical import \
    normalized_sampling_histogram
from eratosthenes.generic.gis_tools import get_mask_boundary

from eratosthenes.preprocessing.shadow_geometry import find_valley
from eratosthenes.preprocessing.image_transforms import \
    hyperbolic_adjustment
from eratosthenes.preprocessing.shadow_filters import \
    anistropic_diffusion_scalar
from eratosthenes.preprocessing.shadow_filters import \
    kuwahara_filter, otsu_filter, fade_shadow_cast

from eratosthenes.presentation.image_io import output_mask, output_image

s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019/'

Sc,spatialRef, geoTransform, targetprj = read_geo_image(
    os.path.join(s2path, 'shadows_coreg.tif'))
Si,spatialRef, geoTransform, targetprj = read_geo_image(
    os.path.join(s2path, 'shadow_coreg.tif'))
Ri,_,_,_ = read_geo_image(os.path.join(s2path, 'albedo_coreg.tif'))
Shd,_,_,_ = read_geo_image(os.path.join(s2path, 'shading.tif'))
Shw,_,_,_ = read_geo_image(os.path.join(s2path, 'shadowing.tif'))

# weights.... a-priori
# shadow_matting:
# compute_confidence_through_sun_angle

# pre-processing....
# shadow_geometry:
# median_filter_shadows
# mean_shift_filter
#

# dare_05: variance within a polygon can be used to distinguish between water
# and shadow (both dark...)

# classification...
# full image based..., sun trace..., polygon based
# struge... similar quantity to count of artificial....?

#plt.hist2d(Si.flatten(), Ri.flatten(), bins=100)

prior_shadow_fraction = np.sum(Shw)/Shw.size
values, base = normalized_sampling_histogram(Sc)
dips, quant = find_valley(values, base, neighbors=2)

# remap to -1 ... +1
Sip = hyperbolic_adjustment(Sc, dips)
output_image(Sip, 'Red-shadow-hyperbolic.jpg', cmap='gray')

# neighborhood enhancement
Sik = kuwahara_filter(Sip, tsize=3)
output_image(Sik, 'Red-shadow-kuwahara.jpg', cmap='gray')

# fading shadowing
zn, az = 90-21.3, 174.2 # 15 Oct 2019
Shf = fade_shadow_cast(Shw, az)
Sia = anistropic_diffusion_scalar(Sip+1j*Shf, iter=40)

from skimage.filters import threshold_sauvola, threshold_otsu

Sis = threshold_sauvola(Sip, window_size=101, k=.05, r=2)
Msk = Sip<Sis

Bnd = get_mask_boundary(Msk)
output_mask(Bnd, 'Red-threshold-sauvola.png', color=np.array([94, 0,255]))

Sio = threshold_otsu(Sip)
Msk = Sip<Sio

Bnd = get_mask_boundary(Msk)
output_mask(Bnd, 'Red-threshold-otsu.png', color=np.array([255, 119,0]))


fout = os.path.join(s2path, 'shadow_class_raw.tif')
make_geo_im(Msk, geoTransform, targetprj, fout)

Msk = Sik<Sis
fout = os.path.join(s2path, 'shadow_class_kuw.tif')
make_geo_im(Msk, geoTransform, targetprj, fout)

fout = os.path.join(s2path, 'shadow_stretch.tif')
make_geo_im(Sip, geoTransform, targetprj, fout)



Si,_,_,_ = read_geo_image(os.path.join(s2path, 'shadow_coreg.tif'))


# Sio = otsu_filter(Sik, tsize=200)

print('+')










print('+')