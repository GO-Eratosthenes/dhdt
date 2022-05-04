import numpy as np

from PIL import Image

from ..generic.handler_im import bilinear_interpolation
from ..generic.mapping_tools import vel2pix
from ..generic.test_tools import construct_correlation_peak

def make_seeds(Msk, n=1e2):
    idx_1,idx_2 = np.where(Msk)
    idx_rnd = np.random.randint(0, high=idx_1.size, size=int(n), dtype=int)
    # create random seed points
    i_samp, j_samp = idx_1[idx_rnd].astype(float), idx_2[idx_rnd].astype(float)
    return i_samp, j_samp

def propagate_seeds(V_i, V_j, i, j):
    i += bilinear_interpolation(V_i, i, j)
    j += bilinear_interpolation(V_j, i, j)
    return i, j

def flow_anim(V_x, V_y, geoTransform, M=np.array([]),
              speedup=1, iter=40, interval=4, fade=0.9, num_seeds=1e2,
              prefix='seeds', color=np.array([255, 0, 64])):
    im_count = 1
    (m,n) = V_x.shape
    if M.size==0: M = np.ones((m,n), dtype=bool)

    V_i, V_j = vel2pix(geoTransform, V_x, V_y)
    i,j = make_seeds(M, n=num_seeds)

    idx_msk = np.ravel_multi_index(np.where(M), (m,n))
    Seed = np.zeros((m,n))
    for counter in range(iter):
        i,j = propagate_seeds(speedup*V_i, speedup*V_j, i, j)

        # are they moving outside the frame or mask
        idx_seed = (n*np.round(i) + np.round(j)).astype(int)
        #idx_seed = np.ravel_multi_index(np.stack((np.round(i),
        #                                          np.round(j))).astype(int),
        #                                (m, n))
        # does not work with out of bound index
        OUT = np.in1d(idx_seed, idx_msk, invert=True)

        # generate new ones
        if np.any(OUT): i[OUT],j[OUT] = make_seeds(M, n=np.sum(OUT))

        Seed_new = construct_correlation_peak(Seed, i,j, origin='corner')

        Seed *= fade
        Seed = np.maximum(Seed, np.real(Seed_new))
        if np.mod(counter, interval)==0:
            # write out image
            rgb = np.dstack(((color[0].astype(float)*Seed).astype(np.uint8),
                             (color[1].astype(float)*Seed).astype(np.uint8),
                             (color[2].astype(float)*Seed).astype(np.uint8)))
            rgba = np.dstack((rgb, (255*Seed[:,:,np.newaxis]).astype(np.uint8)))
            img = Image.fromarray(rgba)
            outputname = prefix + str(im_count).zfill(4) + '.png'
            img.save(outputname)
            im_count += 1
    return