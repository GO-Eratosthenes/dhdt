import os
import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import \
    map2pix, bilinear_interp_excluding_nodat
from dhdt.input.read_sentinel2 import get_local_bbox_in_s2_tile
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.preprocessing.shadow_transforms import entropy_shade_removal
from dhdt.processing.coupling_tools import couple_pair
from dhdt.postprocessing.mapping_io import dh_txt2shp
from dhdt.presentation.image_io import output_draw_color_lines_overlay

base_dir = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'


im_path = ('S2-2019-10-25',
           'S2-2019-10-15')
im_path = ('S2-2017-10-20',
           'S2-2020-10-19')

imShow = False
t_size, s_size = 3,7
soi = 'S.tif'

# get (meta)-data
im_1_name, im_2_name = im_path[0], im_path[1]
fname_1 = os.path.join(base_dir, im_1_name, soi)
fname_2 = os.path.join(base_dir, im_2_name, soi)

if imShow:
    Blue = read_geo_image(os.path.join(base_dir, im_1_name, 'B02.tif'))[0]
    Red = read_geo_image(os.path.join(base_dir, im_1_name, 'B04.tif'))[0]
    Near = read_geo_image(os.path.join(base_dir, im_1_name, 'B08.tif'))[0]

    S_1_raw = entropy_shade_removal(mat_to_gray(Blue),
                                    mat_to_gray(Red),
                                    mat_to_gray(Near), a=138)[0]
    Blue = read_geo_image(os.path.join(base_dir, im_2_name, 'B02.tif'))[0]
    Red = read_geo_image(os.path.join(base_dir, im_2_name, 'B04.tif'))[0]
    Near = read_geo_image(os.path.join(base_dir, im_2_name, 'B08.tif'))[0]

    S_2_raw = entropy_shade_removal(mat_to_gray(Blue),
                                    mat_to_gray(Red),
                                    mat_to_gray(Near), a=138)[0]

    S_1 = read_geo_image(fname_1)[0][:,:,0]
    S_2 = read_geo_image(fname_2)[0][:,:,0]

bbox_ij_s2 = get_local_bbox_in_s2_tile(fname_1,
                                       os.path.join(base_dir, im_path[0]))


post_1, post_2_new, caster, dh, score, caster_new, dh_old, post_2_old, crs = \
    couple_pair(
    fname_1, fname_2, bbox_ij_s2,
    boi=np.array([0]), rect=None, temp_radius=t_size, search_radius=s_size,
    match='norm_corr', prepro=None) #'sundir'

# look at elevation difference
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

(Z,_,geoTransform,_) = read_geo_image(os.path.join(Z_dir, Z_file))

post_1_i, post_1_j = map2pix(geoTransform, post_1[:,0], post_1[:,1])
z_1 = bilinear_interp_excluding_nodat(Z, post_1_i, post_1_j, noData=-9999)

post_2_i, post_2_j = map2pix(geoTransform, post_2_new[:,0], post_2_new[:,1])

post_2_i, post_2_j = map2pix(geoTransform, post_2_old[:,0], post_2_old[:,1])
z_2 = bilinear_interp_excluding_nodat(Z, post_2_i, post_2_j, noData=-9999)

# see difference
dz = z_2-z_1

IN = np.abs(dh_old)<100

img_name = "dh-lines-" +im_1_name+ "-" +im_2_name+ ".png"
output_draw_color_lines_overlay(post_1_i[IN], post_1_j[IN],
                                post_2_i[IN], post_2_j[IN], -dh_old[IN],
                                Z.shape, os.path.join(base_dir, img_name),
                                cmap='RdBu', vmin=-100, vmax=+100,
                                interval=1, pen=1)

shp_name = "test-" +im_1_name+ "-" +im_2_name+ ".shp"
dh_mat = np.append(post_1, np.append(post_2_new, np.append(dh[:,np.newaxis],
    score, axis=1), axis=1), axis=1)
dh_txt2shp(dh_mat, z_1, z_2, os.path.join(base_dir,shp_name), crs)

plt.figure(), plt.hexbin(dz[IN],dh_old[IN],gridsize=20,
                         extent=[-25,+25,-100,+100], bins='log')
plt.colorbar()




v, u = np.gradient(z, .2, .2)
fig, ax = plt.subplots()
ax.scatter(post_1[:,0],post_1[:,1])

ds = 5
fig, ax = plt.subplots()
ax.plot(np.stack((post_1[::ds,0], post_2_new[::ds,0]), axis=1).T,
        np.stack((post_1[::ds,1], post_2_new[::ds,1]), axis=1).T, '-k')


q = ax.quiver(post_1[:,0],post_1[:,1],
              post_2_new[:,0]-post_1[:,0],post_2_new[:,1]-post_1[:,1])
plt.show()
breakpoint