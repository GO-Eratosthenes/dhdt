import os

import numpy as np
import random
import tempfile

from osgeo import osr
from skimage import data
from scipy.interpolate import griddata

from .mapping_io import make_geo_im
from .mapping_tools import get_max_pixel_spacing, map2pix
from .unit_conversion import deg2arg
from .mapping_io import read_geo_info
from .handler_im import bilinear_interpolation
from ..preprocessing.image_transforms import mat_to_gray
from ..preprocessing.shadow_geometry import shadow_image_to_list
from ..processing.matching_tools_frequency_filters import \
    normalize_power_spectrum
from ..processing.matching_tools import get_integer_peak_location
from ..processing.network_tools import get_network_indices
from ..processing.coupling_tools import couple_pair
from ..postprocessing.solar_tools import make_shadowing
from ..postprocessing.photohypsometric_tools import \
    read_conn_files_to_stack, get_casted_elevation_difference, clean_dh, \
    get_hypsometric_elevation_change

def create_sample_image_pair(d=2**7, max_range=1, integer=False, ndim=1):
    """ create an image pair with random offset

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    max_range : float
        maximum offset of the random displacement.
    integer : bool
        should the offset be whole numbers. The default is False.
    ndim : integer, {1,3}
        number of bands making up an image

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : size=(1)
        displacement in the vertical direction
         * "integer" : integer
         * otherwise : float
    random_dj : size=(1)
        displacement in the horizontal direction
         * "integer" : integer
         * otherwise : float
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.

    Notes
    -----
    The following coordinate system is used here:

        .. code-block:: text

          indexing   |
          system 'ij'|
                     |
                     |       j
             --------+-------->
                     |
                     |
          image      | i
          based      v

    """
    im1 = data.astronaut()
    if ndim==1:
        im1 = mat_to_gray(im1[...,0], im1[...,0]==0)
    else:
        ndim = np.maximum(im1.shape[-1], ndim)
    mI,nI = im1.shape[0], im1.shape[1]

    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    if integer: random_di,random_dj = np.round(random_di),np.round(random_dj)

    (grd_j1,grd_i1) = np.meshgrid(np.linspace(1, mI, mI), np.linspace(1, nI, nI))
    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = stk_1 + np.array([random_di, random_dj])
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[:,0], (mI, nI))
    grd_j2 = np.reshape(grd_2[:,1], (mI, nI))

    # do interpolation
    if ndim==1:
        im2 = griddata(stk_1, im1.flatten().T,
                       (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                        grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                       method='cubic')
        im1_same = im1[mI//2-d:mI//2+d,nI//2-d:nI//2+d]

    else:
        im2 = np.zeros((2*d,2*d,ndim))
        for i in range(ndim):
            im2[...,i] = griddata(stk_1, im1[...,i].flatten().T,
                                  (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                                   grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                                  method='linear')
        im2 = im2.astype('uint8')
        im1_same = im1[mI//2-d:mI//2+d, nI//2-d:nI//2+d,:]
    return im1_same, im2, random_di, random_dj, im1

def create_sheared_image_pair(d=2**7,sh_i=0.00, sh_j=0.00, max_range=1):
    """ create an image pair with random offset and shearing

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    sh_j : float, unit=image
        horizontal shear to apply, scalar is based on the image width, ranging
        from -1...1 .
    sh_i : float, unit=image
        vertical shear to apply, scalar is based on the image width, ranging
        from -1...1 .
    max_range : float
        maximum offset of the random displacement.

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : float, unit=pixel
        displacement in the vertical direction
    random_dj : float, unit=pixel
        displacement in the horizontal direction
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.

    Notes
    -----
    The following coordinate system is used here:

    .. code-block:: text

      indexing   |
      system 'ij'|
                 |
                 |       j
         --------+-------->
                 |
                 |
      image      | i
      based      v

    The shear is applied in the following manner:

    .. code-block:: text

            sh_j = 0         sh_j > 0
            +--------+      +--------+
            |        |     /        /
            |        |    /        /
            |        |   /        /
            |        |  /        /
            +--------+ +--------+

    The shear parameter is based upon a centered unit image domain, that is, the
    image extent spans -1...+1
    """
    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    A = np.array([[1, sh_i], [sh_j, 1]]) # transformation matrix
    (mI,nI) = im1.shape

    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI),
                                  np.linspace(-1, 1, nI),
                                  indexing='ij')

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(A, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))

    # introduce offset
    grd_i2 += random_di/mI
    grd_j2 += random_dj/nI

    # do sheared interpolation
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[mI//2-d:mI//2+d,nI//2-d:nI//2+d],
                    grd_j2[mI//2-d:mI//2+d,nI//2-d:nI//2+d]),
                   method='cubic')
    im1_same = im1[mI//2-d:mI//2+d,nI//2-d:nI//2+d]
    return im1_same, im2, random_di, random_dj, im1

def create_scaled_image_pair(d=2**7,sc_x=1.00, sc_y=1.00, max_range=1):
    """ create an image pair with random offset and scaling

    Parameters
    ----------
    d : integer, range=0...256
        radius of the template. The default is 2**7.
    sc_x : float, unit=image
        horizontal scale to apply, scalar is based on the image width, ranging
        from -1...1 .
    sc_y : float, unit=image
        vertical scale to apply, scalar is based on the image height, ranging
        from -1...1 .
    max_range : float
        maximum offset of the random displacement.

    Returns
    -------
    im1_same : np.array, size=(d,d), dtype=float
        image template.
    im2 : np.array, size=(d,d), dtype=float
        image template.
    random_di : float, unit=pixel
        displacement in the vertical direction
    random_dj : float, unit=pixel
        displacement in the horizontal direction
    im1 : np.array, size=(512,512), dtype=float
        image in full resolution.

    Notes
    -----
    .. code-block:: text

            sc_x = 1         sc_x > 1
            +--------+      +------------+
            |        |      |            |
            |        |      |            |
            |        |      |            |
            |        |      |            |
            +--------+      +------------+

    """
    scalar_mul = 2*np.minimum(d // 2, max_range)

    random_di = (np.random.random()-.5)*scalar_mul
    random_dj = (np.random.random()-.5)*scalar_mul # random tranlation

    im1 = data.astronaut()
    im1 = mat_to_gray(im1[:,:,0], im1[:,:,0]==0)
    (mI,nI) = im1.shape

    A = np.array([[1/sc_x, 0], [0, 1/sc_y]]) # transformation matrix
    (mI,nI) = im1.shape

    (grd_i1,grd_j1) = np.meshgrid(np.linspace(-1, 1, mI), np.linspace(-1, 1, nI))

    stk_1 = np.vstack( (grd_i1.flatten(), grd_j1.flatten()) ).T

    grd_2 = np.matmul(A, stk_1.T)
    # calculate new interpolation grid
    grd_i2 = np.reshape(grd_2[0,:], (mI, nI))
    grd_j2 = np.reshape(grd_2[1,:], (mI, nI))

    # introduce offset
    grd_i2 += random_di/mI
    grd_j2 += random_dj/nI

    # do shearing
    im2 = griddata(stk_1, im1.flatten().T,
                   (grd_i2[d:-d,d:-d],grd_j2[d:-d,d:-d]),
                   method='cubic')
    im1_same = im1[d:-d,d:-d]
    return im1_same, im2, random_di, random_dj, im1

def construct_correlation_peak(I, di, dj, fwhm=3., origin='center'):
    """given a displacement, create a gaussian peak

    Parameters
    ----------
    I : np.array, size=(m,n)
        image domain
    di : {float, np.array}
        displacement along the vertical axis
    dj : {float, np.array}
        displacement along the horizontal axis
    fwhm: float
        full width half maximum

    Returns
    -------
    C : np.array, size=(m,n), complex
        array with correlation peak in the form of a circular Gaussian
    """
    (m,n) = I.shape

    (I_grd, J_grd) = np.meshgrid(np.arange(0, m),
                                 np.arange(0, n),
                                 indexing='ij')
    if origin in ('center'):
        I_grd -= m//2
        J_grd -= n//2
    I_grd,J_grd = I_grd.astype('float64'), J_grd.astype('float64')

    if len(di)==1:
        I_grd -= di
        J_grd -= dj
        C = np.exp(-4*np.log(2) * (I_grd**2 + J_grd**2) / fwhm**2)
        return C

    C = np.zeros((m,n), dtype=float)
    for idx,delta_i in enumerate(di):
        delta_j = dj[idx]
        C = np.maximum(C,
                       np.real(
                       np.exp(-4*np.log(2) * ((I_grd-delta_i)**2 +
                                              (J_grd-delta_j)**2) / fwhm**2)))
    return C

def test_phase_plane_localization(Q, di, dj, tolerance=1.):
    C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    di_hat,dj_hat,_,_ = get_integer_peak_location(C)

    assert np.isclose(np.round(di), di_hat, tolerance)
    assert np.isclose(np.round(dj), dj_hat, tolerance)

    return (di-di_hat, dj-dj_hat)

def test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=.1):
    assert np.isclose(di, di_hat, tolerance)
    assert np.isclose(dj, dj_hat, tolerance)

    return (di_hat-di, dj_hat-dj)

def test_phase_direction(theta, di, dj, tolerance=5):
    tolerance = np.radians(tolerance)
    theta = np.radians(theta)
    theta_tilde = np.arctan2(di, dj)

    # convert to complex domain, so angular difference can be done
    a,b = 1j*np.sin(theta), 1j*np.sin(theta_tilde)
    a += np.cos(theta)
    b += np.cos(theta_tilde)

    assert np.isclose(a,b, tolerance)

def test_normalize_power_spectrum(Q, tolerance=.001):
    # trigonometric version
    Qn = 1j*np.sin(np.angle(Q))
    Qn += np.cos(np.angle(Q))
    Qn[Q==0] = 0

    Qd = normalize_power_spectrum(Q)
    assert np.all(np.isclose(Qd,Qn, tolerance))

def create_artificial_terrain(m,n,step_size=.01, multi_res=(2,4)):
    """ create artificail terrain, based upon Perlin noise, with a flavour of
    multi-resolution within

    Parameters
    ----------
    m,n : integer
        dimension of the elevation model to be created
    step_size : float, default=0.01
        parameter to be used to scale the frequency of the topography

    Returns
    -------
    Z : numpy.array, size=(m,n), ndim=2
        data array with elevation values
    geoTransform : tuple, size=(8,1)
        affine transformation coefficients, and dimensions of the array.
    """
    def _linreg(a, b, x):
        return a + x * (b - a)

    def _fade(t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def _gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:,:,0]*x + g[:,:,1]*y

    def _perlin(x, y):
        # permutation table
        rng = np.random.default_rng()
        p = np.arange(256, dtype=int)
        rng.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        x_i,y_i = x.astype(int), y.astype(int)
        # internal coordinates
        x_f,y_f = x-x_i, y-y_i
        # fade factors
        u,v = _fade(x_f),_fade(y_f)
        # noise components
        n00 = _gradient(p[p[x_i] +y_i], x_f, y_f)
        n01 = _gradient(p[p[x_i] +y_i+1], x_f, y_f-1)
        n11 = _gradient(p[p[x_i+1] +y_i+1], x_f-1, y_f-1)
        n10 = _gradient(p[p[x_i+1] +y_i], x_f-1, y_f)
        # combine noises
        x1, x2 = _linreg(n00, n10, u), _linreg(n01, n11, u)
        return _linreg(x1, x2, v)

    lin_m = np.arange(0, step_size*m, step_size)
    lin_n = np.arange(0, step_size*n, step_size)
    x,y = np.meshgrid(lin_n, lin_m)
    Z = _perlin(x, y)
    for f in multi_res:
        Z += mat_to_gray(Z)*(1/f)*_perlin(f*x, f*y)
    Z += 1
    Z *= 1E3
    geoTransform = (0, +10, 0, 0, 0, -10, m, n)
    return Z, geoTransform

def create_artifical_sun_angles(n):
    zn = np.random.uniform(low=35, high=85, size=n)
    az = deg2arg(np.random.uniform(low=160, high=200, size=n))
    return az, zn

#def create_artificial_glacier_mask(Z, seeds=42):
#    i,j = make_seeds(S, n=seeds)
#    R = none
#    return R

def create_shadow_caster_casted(Z, geoTransform, az, zn, out_path,
                                out_name="conn-test.txt", incl_image=False,
                                **kwargs):
    """
    Create artificial shadowing, and a connection file, for testing purposes

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
        array with elevation
    geoTransform : tuple, size={(6,1), (8,1)}
        georeference transform of an image.
    az, zn : float, unit=degrees
        sun angles
    out_path : string
        location where to write the connectivity file
    out_name : string, default="conn-test.txt"
        name of the file
    """
    spac = get_max_pixel_spacing(geoTransform)
    Shw = make_shadowing(Z, az, zn, spac=spac)

    if incl_image:
        im_name = out_name.split('.')[0]+".tif"
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(3411) # Northern Hemisphere Projection Based on Hughes 1980 Ellipsoid
        make_geo_im(Shw, geoTransform, crs.ExportToWkt(),
                    os.path.join(out_path, im_name))

    shadow_image_to_list(Shw, geoTransform, out_path,
                         Zn=zn, Az=az, out_name=out_name, **kwargs)
    return

def test_photohypsometric_coupling(N, Z_shape, tolerance=0.1):
    Z, geoTransform = create_artificial_terrain(Z_shape[0],Z_shape[1])
    az,zn = create_artifical_sun_angles(N)

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        for i in range(N):
            f_name = "conn-"+str(i).zfill(3)+".txt"
            create_shadow_caster_casted(Z, geoTransform,
                az[i], zn[i], dump_path, out_name=f_name, incl_image=False)

        conn_list = tuple("conn-"+str(i).zfill(3)+".txt" for i in range(N))
        dh = read_conn_files_to_stack(None, conn_file=conn_list,
                                      folder_path=dump_path)

    dh = clean_dh(dh)
    dxyt = get_casted_elevation_difference(dh)
    dhdt = get_hypsometric_elevation_change(dxyt, Z, geoTransform)

    assert np.isclose(0, np.quantile(dhdt['dZ_12'], 0.5), atol=tolerance)

def test_photohypsometric_refinement_by_same(Z_shape, tolerance=0.5):
    Z, geoTransform = create_artificial_terrain(Z_shape[0],Z_shape[1])
    az,zn = create_artifical_sun_angles(1)

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        create_shadow_caster_casted(Z, geoTransform,
            az[0], zn[0], dump_path, out_name="conn.txt", incl_image=True)
        # read and refine through image matching
        post_1,post_2_new,caster,dh,score,caster_new,_,_,_ = \
            couple_pair(os.path.join(dump_path, "conn.tif"), \
                        os.path.join(dump_path, "conn.tif"),
                        rect=None)
    pix_dispersion = np.nanmedian(np.hypot(post_1[:,0]-post_2_new[:,0],
                                           post_1[:,1]-post_2_new[:,1]))
    assert np.isclose(0, pix_dispersion, atol=tolerance)

def test_photohypsometric_refinement(N, Z_shape, tolerance=0.1):
    Z, geoTransform = create_artificial_terrain(Z_shape[0],Z_shape[1])
    az,zn = create_artifical_sun_angles(N)

    # create temperary directory
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        dump_path = os.path.join(os.getcwd(), tmpdir)
        for i in range(N):
            f_name = "conn-"+str(i).zfill(3)+".txt"
            create_shadow_caster_casted(Z, geoTransform,
                az[i], zn[i], dump_path, out_name=f_name, incl_image=True)
        # read and refine through image matching
        print('+')
        match_idxs = get_network_indices(N)
        for idx in range(match_idxs.shape[1]):
            fname_1 = "conn-"+str(match_idxs[0][idx]).zfill(3)+".tif"
            fname_2 = "conn-"+str(match_idxs[1][idx]).zfill(3)+".tif"
            file_1 = os.path.join(dump_path,fname_1)
            file_2 = os.path.join(dump_path,fname_2)
            post_1,post_2_new,caster,dh,score,caster_new,_,_,_ = \
                couple_pair(file_1, file_2)
            geoTransform = read_geo_info(file_1)[1]
            i_1,j_1 = map2pix(geoTransform, post_1[:,0], post_1[:,1])
            i_2,j_2 = map2pix(geoTransform, post_2_new[:,0], post_2_new[:,1])
            h_1 = bilinear_interpolation(Z, i_1, j_1)
            h_2 = bilinear_interpolation(Z, i_2, j_2)
            # dh

            print('.')
    return
