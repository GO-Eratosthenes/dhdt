import os

import numpy as np
import random
import tempfile

from osgeo import osr
from skimage import data
from scipy.interpolate import griddata

from .mapping_io import make_geo_im
from .mapping_tools import get_max_pixel_spacing, map2pix
from .unit_conversion import deg2arg, celsius2kelvin
from .mapping_io import read_geo_info
from .handler_im import bilinear_interpolation
from ..preprocessing.image_transforms import mat_to_gray
from ..preprocessing.atmospheric_geometry import \
    refractive_index_visible, refractive_index_broadband, \
    get_sat_vapor_press, get_water_vapor_enhancement
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

def test_water_vapour_fraction():
    """ test if the calculations comply with the original work, using values
    of Table2 in [1]

    References
    ----------
    .. [1] Giacomo, "Equation for the determination of the density of moist air"
           Metrologica, vol.18, pp.33-40, 1982.
    """
    t = np.arange(0, 27.1, 0.1)
    svp_tilde = np.array([611.2, 615.7, 620.2, 624.7, 629.2, 633.8, 638.4,
                          643.0, 647.7, 652.4, 657.1, 661.8, 666.6, 671.4,
                          676.2, 681.1, 686.0, 691.0, 695.9, 700.9, 705.9,
                          711.0, 716.1, 721.2, 726.4, 731.6, 736.8, 742.1,
                          747.3, 752.7, 758.0, 763.4, 768.8, 774.3, 779.8,
                          785.3, 790.9, 796.5, 802.1, 807.8, 813.5, 819.2,
                          825.0, 830.8, 836.6, 842.5, 848.4, 854.4, 860.4,
                          866.4, 872.5, 878.6, 884.7, 890.9, 897.1, 903.4,
                          909.7, 916.0, 922.4, 928.8, 935.2, 941.7, 948.2,
                          954.8, 961.4, 968.1, 974.8, 981.5, 988.3, 995.1,
                          1001.9, 1008.8, 1015.8, 1022.7, 1029.8, 1036.8,
                          1043.9, 1051.1, 1058.3, 1065.5, 1072.8, 1080.1,
                          1087.5, 1094.9, 1102.4, 1109.9, 1117.4, 1125.0,
                          1132.6, 1140.3, 1148.1, 1155.8, 1163.7, 1171.5,
                          1179.4, 1187.4, 1195.4, 1203.5, 1211.6, 1219.7,
                          1227.9, 1236.2, 1244.5, 1252.8, 1261.2, 1269.7,
                          1278.2, 1286.7, 1295.3, 1304.0, 1312.7, 1321.4,
                          1330.2, 1339.1, 1348.0, 1356.9, 1366.0, 1375.0,
                          1384.1, 1393.3, 1402.5, 1411.8, 1421.1, 1430.5,
                          1439.9, 1449.4, 1459.0, 1468.6, 1478.2, 1488.0,
                          1497.7, 1507.5, 1517.4, 1527.4, 1537.4, 1547.4,
                          1557.5, 1567.7, 1577.9, 1588.2, 1598.6, 1609.0,
                          1619.4, 1630.0, 1640.5, 1651.2, 1661.9, 1672.6,
                          1683.5, 1694.4, 1705.3, 1716.3, 1727.4, 1738.5,
                          1749.8, 1761.0, 1772.3, 1783.7, 1795.2, 1806.7,
                          1818.3, 1829.9, 1841.7, 1853.4, 1865.3, 1877.2,
                          1889.2, 1901.2, 1913.3, 1925.5, 1937.8, 1950.1,
                          1962.5, 1974.9, 1987.5, 2000.1, 2012.7, 2025.5,
                          2038.3, 2051.1, 2064.1, 2077.1, 2090.2, 2103.4,
                          2116.6, 2129.9, 2143.3, 2156.8, 2170.3, 2183.9,
                          2197.6, 2211.3, 2225.2, 2239.1, 2253.0, 2267.1,
                          2281.2, 2295.4, 2309.7, 2324.1, 2338.5, 2353.1,
                          2367.7, 2382.4, 2397.1, 2412.0, 2426.9, 2441.9,
                          2456.9, 2472.1, 2487.4, 2502.7, 2518.1, 2533.6,
                          2549.2, 2564.8, 2580.6, 2596.4, 2612.3, 2628.3,
                          2644.4, 2660.6, 2676.8, 2693.2, 2709.6, 2726.1,
                          2742.8, 2759.4, 2776.2, 2793.1, 2810.1, 2827.1,
                          2844.3, 2861.5, 2878.8, 2896.2, 2913.7, 2931.3,
                          2949.0, 2966.8, 2984.7, 3002.7, 3020.7, 3038.9,
                          3057.2, 3075.5, 3094.0, 3112.5, 3131.2, 3149.9,
                          3168.7, 3187.7, 3206.7, 3225.8, 3245.1, 3264.4,
                          3283.8, 3303.4, 3323.0, 3342.8, 3362.6, 3382.5,
                          3402.6, 3422.7, 3443.0, 3463.3, 3483.8, 3504.4,
                          3525.0, 3545.8, 3566.7])
    T = celsius2kelvin(t)
    svp = get_sat_vapor_press(T)
    assert np.all(np.isclose(svp, svp_tilde, atol=1E2))
    return

def test_water_vapor_enhancement():
    """ test if the calculations comply with the original work, using values
    of Table3 in [1]

    References
    ----------
    .. [1] Giacomo, "Equation for the determination of the density of moist air"
           Metrologica, vol.18, pp.33-40, 1982.
    """
    f_hat = np.array([[1.0024, 1.0025, 1.0025, 1.0026, 1.0028, 1.0029, 1.0031],
                    [1.0026, 1.0026, 1.0027, 1.0028, 1.0029, 1.0031, 1.0032],
                    [1.0028, 1.0028, 1.0029, 1.0029, 1.0031, 1.0032, 1.0034],
                    [1.0029, 1.0030, 1.0030, 1.0031, 1.0032, 1.0034, 1.0035],
                    [1.0031, 1.0031, 1.0032, 1.0033, 1.0034, 1.0035, 1.0037],
                    [1.0033, 1.0033, 1.0033, 1.0034, 1.0035, 1.0036, 1.0038],
                    [1.0035, 1.0035, 1.0035, 1.0036, 1.0037, 1.0038, 1.0039],
                    [1.0036, 1.0036, 1.0037, 1.0037, 1.0038, 1.0039, 1.0041],
                    [1.0038, 1.0038, 1.0038, 1.0039, 1.0040, 1.0041, 1.0042],
                    [1.0040, 1.0040, 1.0040, 1.0040, 1.0041, 1.0042, 1.0044],
                    [1.0042, 1.0041, 1.0041, 1.0042, 1.0042, 1.0044, 1.0045]])
    p, t = np.mgrid[60E3:115E3:5E3, 0:35:5]
    f = get_water_vapor_enhancement(t.ravel(), p.ravel())
    assert np.all(np.isclose(f, f_hat.ravel(), atol=1E-3))
    return

def test_refraction_calculation():
    """ follow the examples in the paper [1], to see if they are correct

    References
    ----------
    .. [1] Birch and Jones, "Correction to the updated Edlen equation for the
       refractive index of air", Metrologica, vol.31(4) pp.315-316, 1994.
    .. [2] Ciddor, "Refractive index of air: new equations for the visible and
       near infrared", Applied optics, vol.35(9) pp.1566-1573, 1996.
    """

    # from Table 3 in [1], or Table 1 in [2]
    sigma = 633.                                # central wavelength in [nm]
    T = np.array([20., 20., 20., 10., 30.])     # temperature in [Celsius]
    P = np.array([80., 100., 120., 100., 100.]) # atm. pressure in [kiloPascal]
    P *= 1E3
    n_0 = refractive_index_visible(sigma, T, P)

    n_tilde = np.array([21458.0, 26824.4, 32191.6, 27774.7, 25937.2])
    n_tilde *= 1E-8 # convert to refraction
    n_tilde += 1
    assert np.all(np.isclose(n_0, n_tilde, atol=1E-8))

    frac_Hum = np.array([0., 0., 0., 0., 0.])
    n_0 = refractive_index_broadband(sigma, T, P, frac_Hum, LorentzLorenz=False)


    # from Table.4 in [1], for ambient air
    T = np.array([19.526, 19.517, 19.173, 19.173, 19.188,
                  19.189, 19.532, 19.534, 19.534])
                                            # temperature in [Celsius]
    P = np.array([102094.8, 102096.8, 102993.0, 103006.0, 102918.8,
                  102927.8, 103603.2, 103596.2, 103599.2])
                                            # atmospheric pressure in [Pa]
    p_w = np.array([1065., 1065., 641., 642., 706., 708., 986., 962., 951.])
                                            # vapor pressure of water in [Pa]
    CO2 = np.array([510., 510., 450., 440., 450., 440., 600., 600., 610.])
                                            # carbondioxide concentration, [ppm]
    n_0 = refractive_index_visible(sigma, T, P, p_w, CO2=CO2)

    n_tilde = np.array([27392.3, 27394.0, 27683.4, 27686.9, 27659.1,
                      27661.4, 27802.1, 27800.3, 27801.8])  # measured refraction
    n_tilde *= 1E-8
    n_tilde += 1
    assert np.all(np.isclose(n_0, n_tilde, atol=1E-8))

    return