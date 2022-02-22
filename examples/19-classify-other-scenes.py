import matplotlib.pyplot as plt

import os, glob
import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im
from eratosthenes.generic.handler_sentinel2 import meta_S2string

from eratosthenes.input.read_sentinel2 import \
    read_band_s2, s2_dn2toa

from eratosthenes.preprocessing.image_transforms import mat_to_gray, s_curve
from eratosthenes.preprocessing.shadow_transforms import \
    entropy_shade_removal, normalized_range_shadow_index, \
    fractional_range_shadow_index, shadow_probabilities, shade_index

from eratosthenes.preprocessing.color_transforms import rgb2lms

s2_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.getcwd()!=s2_path:
    os.chdir(s2_path) # change to directory where script is situated


#_,spatialRefI,geoTransformI,targetprjI = read_geo_image(fpath)

im_list = glob.glob("T05VMG*.jp2")
im_tuple = tuple(im_list)

im_time = [meta_S2string(x)[0] for x in im_list]
im_band = [int(x[-6:-4]) for x in im_list]

seen = set()
[x for x in im_list if not (x[:-8] in seen or seen.add(x[:-8]))]


toa = True
for starter in seen:
    print(starter)
    Blue, spatialRef, geoTransform, targetprj = read_band_s2(
        os.path.join(s2_path, starter + '_B02.jp2'))
    Green = read_band_s2(os.path.join(s2_path, starter + '_B03.jp2'))[0]
    Red = read_band_s2(os.path.join(s2_path, starter + '_B04.jp2'))[0]
    Near = read_band_s2(os.path.join(s2_path, starter + '_B08.jp2'))[0]

    NOT = Blue == 0
    if toa:
        Red, Green = s2_dn2toa(Red), s2_dn2toa(Green)
        Blue, Near = s2_dn2toa(Blue), s2_dn2toa(Near)
    else:
        Red, Green = mat_to_gray(Red, NOT), mat_to_gray(Green, NOT)
        Blue, Near = mat_to_gray(Blue, NOT), mat_to_gray(Near, NOT)

    Sn = normalized_range_shadow_index(Blue, Green, Red)
#    Sp = shade_index(Blue, Green, Red, Near)
#    Sn4 = normalized_range_shadow_index(Blue, Green, Red, Near)
#    Sf = fractional_range_shadow_index(Blue, Green, Red)
#    Sr = shadow_probabilities(Blue, Green, Red, Near)
    Si,Ri = entropy_shade_removal(Blue, Green, Red, a=138)

#    th = .5
#    Sc = np.multiply(s_curve(1-Sn) , s_curve(Si+th))
    Sc = (s_curve(1 - Sn, a=20)+s_curve(Si, a=20, b=0))/2
    Sc[NOT] = 0
#    make_geo_im(1-Sn, geoTransform, targetprj,
#                os.path.join(s2_path, starter+'_Sn.tif'))
    make_geo_im(Sc, geoTransform, targetprj,
                os.path.join(s2_path, starter+'_Sc.tif'))
#    make_geo_im(1-Sp, geoTransform, targetprj,
#                os.path.join(s2_path, starter+'_Sp.tif'))
#    make_geo_im(-Sr, geoTransform, targetprj,
#                os.path.join(s2_path, starter+'_Sr.tif'))
#    make_geo_im(Si, geoTransform, targetprj,
#                os.path.join(s2_path, starter+'_Si.tif'))
#    make_geo_im(Ri, geoTransform, targetprj,
#                os.path.join(s2_path, starter+'_Ri.tif'))

    # how to combine...? multiply or add?

    # spatial allignment?

    #Li,Mi,Si = rgb2lms(mat_to_gray(Red), mat_to_gray(Green), mat_to_gray(Blue))
    #RGB = shadow_free_rgb(mat_to_gray(Blue), mat_to_gray(Green),
    #                      mat_to_gray(Red), mat_to_gray(Near))


for starter in seen:
    print(starter)
    Shw, spatialRef, geoTransform, targetprj = read_band_s2(
        os.path.join(s2_path, starter + '_Sc.jp2'))
    # stack

    # quantile