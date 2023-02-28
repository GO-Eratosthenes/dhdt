import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from ..generic.handler_im import bilinear_interpolation
from ..generic.mapping_tools import vel2pix
from ..testing.matching_tools import construct_correlation_peak
from ..generic.gis_tools import polylines2shapefile

def make_seeds(Msk, n=1E2):
    if not np.any(Msk):
        idx_1,idx_2 = np.mgrid[0:Msk.shape[0], 0:Msk.shape[1]]
        idx_1,idx_2 = idx_1.flatten(), idx_2.flatten()
    else:
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
            rgba = np.dstack((rgb, np.atleast_3d(255*Seed).astype(np.uint8)))
            img = Image.fromarray(rgba)
            outputname = prefix + str(im_count).zfill(4) + '.png'
            img.save(outputname)
            im_count += 1
    return

def streamplot2list(X,Y, U,V, density=[1, 1]):
    """

    Parameters
    ----------
    X,Y : numpy.ndarray, size=(m,n)
        map coordinates
    U,V : numpy.ndarray, size=(m,n)
        displacement in X and Y direction
    density: {float, list}
        density of streamlines

    Returns
    -------
    stream_stack : list
        list with arrays with coordinates of the streamlines
    """
    # admin
    if not np.all(np.diff(X) > 0):
        X,U,V = np.fliplr(X), np.fliplr(U), np.fliplr(V)
    if not np.all(np.diff(Y) > 0):
        Y,U,V = np.flipud(Y), np.flipud(U), np.flipud(V)

    # generate streamplot
    stream = plt.streamplot(X, Y, U, V, density=density)

    # get data from container
    paths = stream.lines.get_paths()
    previous = None
    stream_stack = []
    for i in range(len(paths)):
        vert = paths[i].vertices
        if not isinstance(previous, np.ndarray):
            previous, trace = vert.copy(), vert.copy()
            continue

        if np.all((vert[0, :] - previous[-1, :]) == 0):  # connected
            trace = np.vstack((trace, vert[1, :]))
        else:
            stream_stack.append(trace)
            trace = vert.copy()
        previous = vert.copy()
    plt.close()
    return stream_stack

def streamplot2shapefile(X,Y,U,V,spatialRef,
                         out_path,out_file='streamplot.shp',
                         density=[1, 1]):

    stream_stack = streamplot2list(X, Y, U, V, density=density)
    # write shapefile
    polylines2shapefile(stream_stack, spatialRef, out_path, out_file)
    return