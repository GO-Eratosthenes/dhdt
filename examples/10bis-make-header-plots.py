import os

import numpy as np
from PIL import Image

from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.preprocessing.shadow_transforms import mat_to_gray, gamma_adjustment, log_adjustment

rgi_id = 'RGI60-01.19773' # Red Glacier
bbox = (4353, 5279, 9427, 10980) # 1000 m buffer

f2018 = "T05VMG_20180913T214531_B0"
f2020 = "T05VMG_20200912T214531_B0"

# read data
I_18_b,_,_,_ = read_geo_image(os.path.join('header', f2018+'2.jp2'))
I_18_g,_,_,_ = read_geo_image(os.path.join('header', f2018+'3.jp2'))
I_18_r,_,_,_ = read_geo_image(os.path.join('header', f2018+'4.jp2'))

I_20_b,_,_,_ = read_geo_image(os.path.join('header', f2020+'2.jp2'))
I_20_g,_,_,_ = read_geo_image(os.path.join('header', f2020+'3.jp2'))
I_20_r,_,_,_ = read_geo_image(os.path.join('header', f2020+'4.jp2'))                              

# make sub-set
I_18_r = I_18_r[bbox[0]:bbox[1],bbox[2]:bbox[3]]
I_18_g = I_18_g[bbox[0]:bbox[1],bbox[2]:bbox[3]]
I_18_b = I_18_b[bbox[0]:bbox[1],bbox[2]:bbox[3]]

I_20_r = I_20_r[bbox[0]:bbox[1],bbox[2]:bbox[3]]
I_20_g = I_20_g[bbox[0]:bbox[1],bbox[2]:bbox[3]]
I_20_b = I_20_b[bbox[0]:bbox[1],bbox[2]:bbox[3]]

# 
I_18_r = mat_to_gray(I_18_r)
I_18_g = mat_to_gray(I_18_g)
I_18_b = mat_to_gray(I_18_b)

I_20_r = mat_to_gray(I_20_r)
I_20_g = mat_to_gray(I_20_g)
I_20_b = mat_to_gray(I_20_b)

I_18_rgb = np.dstack((I_18_r, I_18_g, I_18_b))
I_18_rgb = np.uint8(255*I_18_rgb)

I_18_rgb_l = log_adjustment(I_18_rgb)

img = Image.fromarray(I_18_rgb_l)
img.save("Red-sen2-13-09-2018.jpg", quality=95)

I_20_rgb = np.dstack((I_20_r, I_20_g, I_20_b))
I_20_rgb = np.uint8(255*I_20_rgb)

I_20_rgb_l = log_adjustment(I_20_rgb)

img = Image.fromarray(I_20_rgb_l)
img.save("Red-sen2-12-09-2020.jpg", quality=95)
