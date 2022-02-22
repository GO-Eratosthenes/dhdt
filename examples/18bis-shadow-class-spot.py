import os

import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.filtering_statistical import \
    normalized_sampling_histogram

from eratosthenes.preprocessing.shadow_geometry import find_valley
from eratosthenes.preprocessing.image_transforms import \
    hyperbolic_adjustment, mat_to_gray
from eratosthenes.preprocessing.shadow_filters import \
    kuwahara_filter

from eratosthenes.preprocessing.shadow_transforms import \
    entropy_shade_removal, normalized_range_shadow_index

from eratosthenes.input.read_spot5 import \
    list_central_wavelength_hrg

s5path = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/SPOT',
                      '004-005_S5_446-227-0_2012-10-11-21-08-05_HRG-2_J_MX_TT',
                      'SCENE01')

im_stack,spatialRef, geoTransform, targetprj = read_geo_image(os.path.join(s5path,'IMAGERY.TIF'))

Green, Red = im_stack[:,:,0], im_stack[:,:,1]
Near, Swir = im_stack[:,:,2], im_stack[:,:,3]

Sn = normalized_range_shadow_index(mat_to_gray(Green), mat_to_gray(Red))

Si,Ri = entropy_shade_removal(mat_to_gray(Green),
                              mat_to_gray(Red),
                              mat_to_gray(Near), a=138)

fout = os.path.join(s5path, 'shadows_coreg.tif')
make_geo_im(Sn, geoTransform, targetprj, fout)
fout = os.path.join(s5path, 'shadow_coreg.tif')
make_geo_im(Si, geoTransform, targetprj, fout)



Sc,spatialRef, geoTransform, targetprj = read_geo_image(
    os.path.join(s5path, 'shadows_coreg.tif'))

Si,spatialRef, geoTransform, targetprj = read_geo_image(
    os.path.join(s5path, 'shadow_coreg.tif'))
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

# neighborhood enhancement
Sik = kuwahara_filter(Sip, tsize=3)

from skimage.filters import threshold_sauvola

Sis = threshold_sauvola(Sip, window_size=101, k=.05, r=2)

Msk = Sip<Sis

fout = os.path.join(s2path, 'shadow_class_raw.tif')
make_geo_im(Msk, geoTransform, targetprj, fout)

Msk = Sik<Sis
fout = os.path.join(s2path, 'shadow_class_kuw.tif')
make_geo_im(Msk, geoTransform, targetprj, fout)

fout = os.path.join(s2path, 'shadow_stretch.tif')
make_geo_im(Sip, geoTransform, targetprj, fout)
