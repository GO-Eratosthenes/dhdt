import numpy as np

from dhdt.generic.unit_conversion import deg2compass
from dhdt.postprocessing.solar_tools import (az_to_sun_vector,
                                             sun_angles_to_vector,
                                             vector_to_sun_angles)


def test_sun_angles_to_vector():
    az = np.random.uniform(low=0., high=360., size=(1, ))[0]
    zn = np.random.uniform(low=0., high=90., size=(1, ))[0]
    coord_sys = ['xy', 'ij']
    for idx in coord_sys:
        s = sun_angles_to_vector(az, zn, indexing=idx)
        # reverse format
        az_r, zn_r = vector_to_sun_angles(s, indexing=idx)
        az_r = deg2compass(az_r)
        assert np.isclose(0, np.abs(az - az_r))
        assert np.isclose(0, np.abs(zn - zn_r))


def test_az_to_sun_vector():
    az = np.random.uniform(low=0., high=360., size=(1, ))[0]
    coord_sys = ['xy', 'ij']
    for idx in coord_sys:
        s = np.zeros((3, ))
        s[:2] = np.squeeze(az_to_sun_vector(az, indexing=idx))
        # reverse format
        az_r = vector_to_sun_angles(s, indexing=idx)[0]
        az_r = deg2compass(az_r)
        assert np.isclose(0, np.abs(az - az_r))
