import os
import numpy as np
import matplotlib.pyplot as plt

#from osgeo import osr
#from dhdt.generic.mapping_tools import ll2map

from dhdt.generic.mapping_io import read_geo_image
from dhdt.preprocessing.image_transforms import mat_to_gray, gamma_adjustment
from dhdt.preprocessing.color_transforms import \
    rgb2hsi, hsi2rgb, rgb2hcv, hcv2rgb

base_dir = '/Users/Alten005/Documents/ArtiVist'

#fname = 'genshtab0500-n32-3_ll.tif.points'
#fout = 'genshtab0500-n32-3_utm.tif.points'

I1 = read_geo_image(os.path.join(base_dir, 'clip01.tif'))[0]
I1 = I1.astype(float)

i,j = np.mgrid[0:I1.shape[0], 0:I1.shape[1]]

N = 1000
IN = np.random.random_integers(0, I1.size, N)
A = np.stack((i.flatten()[IN], j.flatten()[IN], i.flatten()[IN]*j.flatten()[IN],
              i.flatten()[IN]**2, j.flatten()[IN]**2, np.ones(N))).T

x = np.linalg.lstsq(A, I1.flatten()[IN], rcond=None)[0]
I1 -= i*x[0] + j*x[1] + i*j*x[2] + i**2*x[3] + j**2*x[4] + x[5]

I1 = mat_to_gray(I1, vmin=-2E4, vmax=6E4)

# plt.figure(), plt.imshow(gamma_adjustment(I1, .5), vmin=.25, vmax=.95, cmap=plt.cm.magma), plt.colorbar()
#plt.figure(), plt.imshow(gamma_adjustment(I1, .5)), plt.colorbar()

I2 = read_geo_image(os.path.join(base_dir, 'clip02.tif'))[0]
I2 = mat_to_gray(I2)
H,S,I = rgb2hsi(I2[...,0], I2[...,1], I2[...,2])
R,G,B = hsi2rgb(H/1.25,1-S,1-I)

h,C,V = rgb2hcv(R, G, B)

R,G,B = hsi2rgb(H,1-S,I)
Rgb = 255*np.dstack((R,G,B))
plt.figure(), plt.imshow(Rgb.astype(np.uint8))

#plt.imshow(, cmap=plt.cm.twilight), plt.colorbar()

#cos, sin sin**2
h2 = np.arctan2(np.sin(h), np.cos(h))

ib = -.05*np.pi # mooie ouwe look
hb = np.mod(h+np.pi+ib,2*np.pi)-np.pi

ib = +0.8*np.pi # paars prachtig
h2 = np.arctan2(np.cos(h**2),-np.sin(h**2))
hb = np.mod(h2+np.pi+ib,2*np.pi)-np.pi
R,G,B = hcv2rgb(hb,C,V)
Rgb = 255*np.dstack((R,G,B))
plt.figure(), plt.imshow(Rgb.astype(np.uint8))

#ffull = os.path.join(base_dir, fname)

#A = np.loadtxt(ffull, delimiter=',', skiprows=1)

#proj = osr.SpatialReference()
#proj.SetWellKnownGeogCS('WGS84')
#proj.SetUTM(32, True)
#xy = ll2map(np.fliplr(A[:,:2]), proj)
#A[:,:2] = xy[:,:2]

#ffull = os.path.join(base_dir, fout)
#np.savetxt(ffull, A, delimiter=',', fmt='%0.8f',
#           header='mapX,mapY,pixelX,pixelY,enable,dX,dY,residual')
