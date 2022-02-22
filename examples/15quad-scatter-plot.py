import matplotlib.pyplot as plt

import os
import numpy as np

from eratosthenes.generic.mapping_io import read_geo_image, make_geo_im

from eratosthenes.input.read_sentinel2 import \
    list_central_wavelength_msi
from eratosthenes.preprocessing.image_transforms import mat_to_gray

toi = 15

boi = ['red', 'green', 'blue', 'nir']
s2_df = list_central_wavelength_msi()
s2_df = s2_df[s2_df['common_name'].isin(boi)]

if toi==15:
    s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019/'
elif toi == 25:
    s2path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-25-10-2019/'

fpath = os.path.join(s2path, 'shadow.tif')

M_dir, M_name = os.path.split(fpath)
Z_file = "COP_DEM_red.tif"
R_file = "5VMG_RGI_red.tif"
Z_dir = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/',
                      'Cop-DEM-GLO-30')

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.getcwd()!=dir_path:
    os.chdir(dir_path) # change to directory where script is situated

(M, spatialRefM, geoTransformM, targetprjM) = read_geo_image(fpath)
M *= -1

Z = read_geo_image(os.path.join(Z_dir, Z_file))[0]
R = read_geo_image(os.path.join(Z_dir, R_file))[0]

# create observation angles
if toi==15:
    fpath = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019-full', \
                          'T05VMG_20191015T213531_B08.jp2')
elif toi==25:
    fpath = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/S2-15-10-2019-full', \
                         'T05VMG_20191015T213531_B08.jp2')

_,spatialRefI,geoTransformI,targetprjI = read_geo_image(fpath)

path_meta = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Sentinel-2/'+\
    'S2A_MSIL1C_20191015T213531_N0208_R086_T05VMG_20191015T230223.SAFE/'+\
    'GRANULE/L1C_T05VMG_A022534_20191015T213843/QI_DATA'

if toi==15:
    Det = read_geo_image(os.path.join(s2path, 'detIdBlue.tif'))[0]
    Zn = read_geo_image(os.path.join(s2path, 'viewZn.tif'))[0]
    Az = read_geo_image(os.path.join(s2path, 'viewAz.tif'))[0]

Blue = read_geo_image(os.path.join(s2path, 'B2.tif'))[0]
Green = read_geo_image(os.path.join(s2path, 'B3.tif'))[0]
Red = read_geo_image(os.path.join(s2path, 'B4.tif'))[0]
Near = read_geo_image(os.path.join(s2path, 'B8.tif'))[0]

from eratosthenes.preprocessing.shadow_transforms import \
    entropy_shade_removal, shadow_free_rgb, shade_index, \
    normalized_range_shadow_index
from eratosthenes.preprocessing.color_transforms import rgb2lms

Ia,Ib,Ic = mat_to_gray(Blue), mat_to_gray(Red), mat_to_gray(Near)

Ia[Ia == 0] = 1e-6
Ib[Ib == 0] = 1e-6
sqIc = np.power(Ic, 1. / 2)
sqIc[sqIc == 0] = 1

chi1 = np.log(np.divide(Ia, sqIc))
chi2 = np.log(np.divide(Ib, sqIc))

fig = plt.figure()
plt.hist2d(chi1.flatten(),chi2.flatten(),
           bins=100, range=[[-3,0],[-3,0]],
           cmap=plt.cm.CMRmap)
plt.axis('equal'), plt.grid()


num_classes = 8
c_p = plt.cm.Paired
for i in np.arange(num_classes):
    xy = plt.ginput(4)
    x,y = [p[0] for p in xy], [p[1] for p in xy]
    x.append(x[0]) # make polygon
    y.append(y[0])

    plt.plot(x, y, color=c_p.colors[i]) # do plotting on the figure
    plt.draw()
    if i == 0:
        X,Y = np.array(x)[:,np.newaxis], np.array(y)[:,np.newaxis]
    else:
        X = np.hstack((X, np.array(x)[:,np.newaxis]))
        Y = np.hstack((Y, np.array(y)[:,np.newaxis]))

from matplotlib.path import Path

Class = np.zeros_like(Red, dtype=int)
for i in np.arange(num_classes):
    p = Path(np.stack((X[:,i], Y[:,i]), axis=1)) # make a polygon

    IN = p.contains_points(np.stack((chi1.flatten(), chi2.flatten()), axis=1))
    Class[IN.reshape(Red.shape)] = int(i + 1)


plt.figure(), plt.imshow(Class, cmap=c_p, vmin=1, vmax=12)
#plt.colorbar(boundaries=np.linspace(0, num_classes, num_classes))



#plt.figure()
#plt.hist2d(Ia.flatten(),Ib.flatten(), bins=100,
#           range=[[0.,.1],[0.,.1]])