import numpy as np
import morphsnakes as ms

from dhdt.testing.terrain_tools import create_artificial_morphsnake_data

def test_morphsnakes(m=400,n=600,d=10):
    Shw,I,M = create_artificial_morphsnake_data(m,n, delta=d)
    M_acwe = ms.morphological_chan_vese(I[...,0], d+3, init_level_set=M,
                                        smoothing=0, lambda1=1, lambda2=1,
                                        albedo=I[...,-1])

    misfit_init = np.logical_xor(Shw,M).sum()
    misfit_snak = np.logical_xor(Shw,M_acwe).sum()
    assert misfit_init>misfit_snak, 'no optimization seems to be done'
    return