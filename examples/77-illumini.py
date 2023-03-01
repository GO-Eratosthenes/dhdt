import numpy as np

import matplotlib.pyplot as plt

from dhdt.testing.matching_tools import create_artificial_terrain, create_artifical_sun_angles
from dhdt.processing.matching_tools_frequency_correlators import phase_corr
from dhdt.processing.matching_tools_frequency_filters import \
    perdecomp, make_fourier_grid, cross_shading_filter
from dhdt.postprocessing.solar_tools import \
    make_shading

N = 2
m,n = 2**10, 2**10

Z, geoTransform = create_artificial_terrain(m,n)
az,zn = create_artifical_sun_angles(N, az_min=-150.0, az_max=+170.0,
                                    zn_min=60., zn_max=65.)

#az = np.array([60., 180.])
#zn = np.array([45.,  45.])

S_1 = make_shading(Z, az[0], zn[0])
S_2 = make_shading(np.roll(Z,(3,1)), az[1], zn[1])
S_1, S_2 = perdecomp(S_1)[0], perdecomp(S_2)[0]


Q = phase_corr(S_1,S_2)
W = cross_shading_filter(Q, az[0], az[1])

print('.')

plt.figure(), plt.imshow(np.angle(np.fft.fftshift(Q)), cmap=plt.cm.twilight_r)
plt.figure(), plt.imshow(W, cmap=plt.cm.twilight_r)

# make fourier grid
F_1,F_2 = make_fourier_grid(Q, indexing='ij', system='radians')
theta_r = np.rad2deg(np.angle(F_1 + 1j*F_2))
az_r = np.deg2rad(az)


Q_ill = np.ones_like(S_1)
OUT = np.logical_and((az_r[1] - (3*np.pi/2)) < theta,
                     (theta < az_r[0] - (np.pi/2)))
np.putmask(Q_ill, OUT,-1)
OUT = np.logical_and((az_r[1] - (np.pi/2)) < theta,
                     (theta < az_r[0] + (np.pi/2)))
np.putmask(Q_ill, OUT,-1)


