import numpy as np

from pysolar.solar import get_azimuth, get_altitude
from datetime import datetime
from pytz import timezone

from scipy import ndimage
from scipy.spatial.transform import Rotation

from skimage import transform


spac = .5 # resolution

az = np.arange(0, 360, spac)
zn = np.flip(np.arange(-spac, +90, spac))

Az = np.repeat(az[np.newaxis,:], zn.shape[0],axis=0)
Zn = np.repeat(zn[:,np.newaxis], az.shape[0], axis=1)

Sol = np.zeros((zn.shape[0], az.shape[0]))

latitude = 51.707524
longitude = 6.244362
year = 2018
month = np.array([12, 6])
day = np.array([21, 21])

# 21 12 winter solstice - lower bound
# 21 06 summer solstice - upper bound

#datetime.utcnow()

for i in range(0,2):
    for hour in range(0, 24):
        for minu in range(0, 60):
            for sec in range(0, 60, 20):
                sun_zen = get_altitude(latitude, longitude, \
                                      datetime(year, month[i], day[i], \
                                               hour, minu, sec, \
                                               tzinfo=timezone('UTC')))
                sun_azi = get_azimuth(latitude, longitude, \
                                      datetime(year, month[i], day[i], \
                                               hour, minu, sec, \
                                               tzinfo=timezone('UTC')))
            
            az_id = (np.abs(az - sun_azi)).argmin()
            zn_id = (np.abs(zn - sun_zen)).argmin()
            if i==0:    
                Sol[zn_id,az_id] = -1
            else:
                Sol[zn_id,az_id] = +1

Sol = Sol[:-1,:]
#datetime.utcnow())

# mathematical morphology to do infilling, and extent the boundaries a bit
Sol_plu, Sol_min = Sol==+1, Sol==-1
Sol_plu = ndimage.binary_dilation(Sol_plu, np.ones((5,5))).cumsum(axis=0)==1
Sol_min = np.flipud(ndimage.binary_dilation(Sol_min, np.ones((5,5))))
Sol_min = np.flipud(Sol_min.cumsum(axis=0)==1)

Sky = np.zeros(Sol.shape)
for i in range(0,Sol.shape[1]):
    mat_idx = np.where(Sol_plu[:,i]==+1)
    if len(mat_idx[0]) > 0:
        start_idx = mat_idx[0][0]
        mat_idx = np.where(Sol_min[:,i]==1)
        if len(mat_idx[0]) > 0:
            end_idx = mat_idx[0][-1]    
        else:
            end_idx = Sol.shape[1]
        Sky[start_idx:end_idx,i] = 1   

import matplotlib.pyplot as plt
plt.imshow(Ms)
plt.show()

Z = np.loadtxt('DEM.txt', delimiter=" ", skiprows=6)

spac = 10
zn_oi = 80
az_oi = 120

Zr = ndimage.rotate(Z, az_oi, cval=-1, order=0)
Mr = ndimage.rotate(np.ones(Z.shape, dtype=bool), az_oi, cval=False, order=0)

dZ = np.tan((90-zn_oi))*spac

for i in range(1,Zr.shape[1]):
    Zr[i,:] = np.maximum(Zr[i,:], Zr[i-1,:]-dZ)

Zs = ndimage.rotate(Zr, -az_oi, cval=-1, order=0)
Ms = ndimage.rotate(Mr, -az_oi, cval=0, order=0)
i_ok, j_ok = np.where(np.any(Ms, axis=1)), np.where(np.any(Ms, axis=0))
Zs = Zs[i_ok[0][0]:i_ok[0][-1]+1, j_ok[0][0]:j_ok[0][-1]+1]

Sw = Zs>Z

plt.imshow(Sw)
plt.show()

r = Rotation.from_euler('zyx', np.array([az_oi,  0, 0]), degrees=True)
Z_hat = ndimage.affine_transform(Z, r.as_matrix())



r = Rotation.from_euler('xyz', np.array([0,  0, az_oi]), degrees=True)
#Rs = r.as_matrix()

#Rs[-1,:] = np.array([0, 0, 1])
#Rs[:,-1] = np.array([0, 0, 1])
Rs = np.array([[1, -0.5, 200],
                   [0.1, 0.9, 200],
                   [0.0015, 0.0015, 1]])
tform = transform.ProjectiveTransform(matrix=Rs)
Z_hat = transform.warp(Z, tform.inverse, cval=-1)

tform = transform.ProjectiveTransform(r.as_matrix())
Z_hat = transform.warp(Z, tform.inverse, cval=-1, order=0)

plt.imshow(Sh)
plt.show()


Z_hat = ndimage.affine_transform(Z, Rs)

# make shading
dZdx,dZdy = np.gradient(Z, spac, spac)
vx = np.dstack((dZdx, np.ones_like(dZdx), np.zeros_like(dZdx)))
vy = np.dstack((dZdy, np.ones_like(dZdy), np.zeros_like(dZdy)))

surf_norm = np.cross(vx, vy)
norm_magn = np.linalg.norm(surf_norm, axis=2)
norm_unit = np.divide( surf_norm, np.expand_dims(norm_magn, axis=2))

sun_unit = np.array([np.sin(np.radians((zn_oi))) * np.cos(np.radians((az_oi))),\
                     np.sin(np.radians((zn_oi))) * np.sin(np.radians((az_oi))),\
                     np.cos(np.radians((zn_oi)))])

Sh = norm_unit[:,:,0]*sun_unit[0] + \
    norm_unit[:,:,1]*sun_unit[1] + \
    norm_unit[:,:,2]*sun_unit[2]








# get annual range
# specify resolution
# create catalogue

# merge to daily integrations




