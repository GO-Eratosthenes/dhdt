import numpy as np

from dhdt.generic.attitude_tools import rot_mat
from dhdt.postprocessing.adjustment_geometric_temporal import \
    rotate_disp_field, rot_covar, rotate_variance

def test_rotate_disp_field(m=40,n=30):
    # initialization
    UV = np.random.random([m,n]) + 1j* np.random.random([m,n])
    θ = np.random.uniform(low=0., high=360., size=(1,))

    # forward rotation
    UV_frw = rotate_disp_field(UV, θ)

    # whole circle rotation
    UV_whl = rotate_disp_field(UV_frw, 360-θ)
    assert np.isclose(0, np.max(np.abs(UV - UV_whl)))

    # backward rotation
    UV_bck = rotate_disp_field(UV_frw, -θ)
    assert np.isclose(0, np.max(np.abs(UV - UV_bck)))

def test_rot_covar():
    # initialization
    I = np.eye(2)
    θ = np.random.uniform(low=0., high=360., size=(1,))[0]

    # forward rotation
    I_frw = rot_covar(I, rot_mat(θ))

    # whole circle rotation
    I_whl = rot_covar(I_frw, rot_mat(360-θ))
    assert np.isclose(0, np.max(np.abs(I - I_whl)))

    # backward rotation
    I_bck = rot_covar(I_frw, np.transpose(rot_mat(θ)))
    assert np.isclose(0, np.max(np.abs(I - I_bck)))
