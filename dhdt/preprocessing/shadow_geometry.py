import os
import glob
import numpy as np

from scipy import ndimage  # for image filtering
from scipy import special  # for trigonometric functions

from skimage import measure
from skimage import segmentation  # for superpixels
from skimage import color  # for labeling image
from skimage.morphology import remove_small_objects # opening, disk, erosion, closing

from rasterio.features import shapes  # for raster to polygon

from shapely.geometry import shape
from shapely.geometry import Point, LineString
from shapely.geos import TopologicalError  # for troubleshooting

from ..generic.mapping_io import read_geo_image, read_geo_info
from ..generic.mapping_tools import \
    cast_orientation, make_same_size, pix_centers, map2pix, pix2map
from ..generic.filtering_statistical import normalized_sampling_histogram
from ..input.read_sentinel2 import \
    read_sun_angles_s2, read_mean_sun_angles_s2, read_sensing_time_s2

def vector_arr_2_unit(vec_arr):
    n = np.linalg.norm(vec_arr, axis=2)
    vec_arr[..., 0] /= n
    vec_arr[..., 1] /= n
    vec_arr[..., 2] /= n
    return vec_arr

def estimate_surface_normals(Z, spac=10.):
    """ given an array with elevation values, and a given spacing, estimate the surface normals

    Parameters
    ----------
    Z : np.array, size=(m,n)
        array with elevation values
    spac : float
        spacing between posts, (isotropic spacing is assumed, that is a square grid)

    Returns
    -------
    normal : np.array, size=(m,n,3)
        array with unit vectors of the surface normal
    """
    dy, dx = np.gradient(Z, spac)

    normal = np.dstack((-dx, dy, np.ones_like(Z)))
    normal = vector_arr_2_unit(normal)
    return normal

def make_shadowing(dem_path, dem_file, im_path, im_name,
                   Zn=45., Az=-45., nodata=-9999, dtype=bool ):

    (dem_mask, crs_dem, geoTransform_dem, targetprj_dem) = read_geo_image(
        os.path.join(dem_path, dem_file))
    if 'MSIL1C' in im_name: # for Sentinel-2
        fname = os.path.join(im_path, im_name, '*B08.jp2')
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            glob.glob(fname)[0])
    else:
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            os.path.join(im_path, im_name))

    dem = make_same_size(dem_mask, geoTransform_dem, geoTransform_im,
                         rows, cols)
    msk = dem==nodata

    if 'MSIL1C' in im_name: # for Sentinel-2
        Zn, Az = read_mean_sun_angles_s2(os.path.join(im_path, im_name))

    # turn towards the sun
    dem_rot = ndimage.rotate(dem, Az,
                             axes=(1, 0), reshape=True, output=None,
                             order=3, mode='constant', cval=nodata)
    msk_rot = ndimage.rotate(msk, Az,
                             axes=(1, 0), reshape=True, output=None,
                             order=0, mode='constant', cval=True)
    del dem
    shw_rot = np.zeros_like(dem_rot, dtype=bool)

    dZ = special.tandg(90-Zn) * np.sqrt(1 +
                                        min(special.tandg(Az)**2,
                                            special.cotdg(Az)**2)) * geoTransform_dem[1]

    cls_rot = np.bitwise_not(ndimage.binary_dilation(msk_rot,
                                                    structure=np.ones((7,7)))
                             ).astype(int)
    # sweep over matrix
    for swp in range(1,dem_rot.shape[0]):
        shw_rot[swp,:] = (dem_rot[swp,:] < dem_rot[swp-1,:]-dZ) & ( cls_rot[swp,:]==1 )
        dem_rot[swp,:] = np.maximum((dem_rot[swp,:]*cls_rot[swp,:]),
                                    (dem_rot[swp-1,:]*cls_rot[swp-1,:]) - dZ*cls_rot[swp,:])
    del dem_rot, cls_rot
    msk_new = ndimage.rotate(msk_rot, -Az,
                             axes=(1, 0), reshape=True, output=None,
                             order=0, mode='constant', cval=True)

    i_min = int(np.floor((msk_new.shape[0] - msk.shape[0])/2))
    i_max = int(np.floor((msk_new.shape[0] + msk.shape[0])/2))
    j_min = int(np.floor((msk_new.shape[1] - msk.shape[1])/2))
    j_max = int(np.floor((msk_new.shape[1] + msk.shape[1])/2))


    if isinstance(dtype, (int, float)):
        shw_rot.astype(dtype)
        shw_new = ndimage.rotate(shw_rot, -Az,
                             axes=(1, 0), reshape=True, output=None,
                             order=1, mode='constant', cval=False)
    else:
        shw_new = ndimage.rotate(shw_rot, -Az,
                             axes=(1, 0), reshape=True, output=None,
                             order=0, mode='constant', cval=False)

    Shw = shw_new[i_min:i_max, j_min:j_max]
    return Shw

def make_shading(dem_path, dem_file, im_path, im_name,
                 Zn=np.radians(45.), Az=np.radians(-45.), nodata=-9999,):
    """
    make hillshading from elevation model and meta data of image file

    """
    (dem_mask, crs_dem, geoTransform_dem, targetprj_dem) = read_geo_image(
        os.path.join(dem_path, dem_file))
    if 'MSIL1C' in im_name: # for Sentinel-2
        fname = os.path.join(im_path, im_name, '*B08.jp2')
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            glob.glob(fname)[0])
    else:
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            os.path.join(im_path, im_name))

    dem = make_same_size(dem_mask, geoTransform_dem, geoTransform_im,
                         rows, cols)

    if 'MSIL1C' in im_name: # for Sentinel-2
        Zn, Az = read_mean_sun_angles_s2(os.path.join(im_path, im_name))
        Zn = np.radians(Zn)
        Az = np.radians(Az)

    # convert to unit vectors
    sun = np.dstack((np.sin(Az), np.cos(Az), np.tan(Zn)))
    sun = vector_arr_2_unit(sun)

    normal = estimate_surface_normals(dem, geoTransform_im[1])

    Shd = normal[:,:,0]*sun[:,:,0] + normal[:,:,1]*sun[:,:,1] + normal[:,:,2]*sun[:,:,2]
    return Shd

# geometric functions
def get_shadow_polygon(M, t_siz):
    """ use superpixels to group and label an image

    Parameters
    ----------
    M : numpy.array, size=(m,n)
        grid with intensity values
    t_size : integer
        window size of the kernel

    Returns
    -------
    labels : numpy.array, size=(m,n)
        grid with numbered labels
    super_pix : numpy.array, size=(m,n)
        grid with numbered superpixels
    """
    mn = np.ceil(np.divide(np.nanprod(M.shape), t_size))
    super_pix = segmentation.slic(M, sigma=1,
                               n_segments=mn,
                               compactness=0.010)  # create super pixels

    mean_im = color.label2rgb(super_pix, M, kind='avg')
    labels = sturge(mean_im)[0]
    return labels, super_pix

# threshholding and classification functions
def sturge(M):
    """ Transform intensity array to labelled image

    Parameters
    ----------
    M : np.array, size=(m,n), dtype=float
        array with intensity values

    Returns
    -------
    labels : np.array, size=(m,n), dtype=integer
        array with numbered labels
    """
    values, base = normalized_sampling_histogram(M)
    dips = find_valley(values, base, 2)
    val = max(dips)
    imSeparation = M > val # or <

    # remove "salt and pepper" noise
    imSeparation = remove_small_objects(imSeparation, min_size=10)

    labels, num_polygons = ndimage.label(imSeparation)
    return labels, val

def find_valley(values, base, neighbors=2):
    """ A valley is a point which has "n" consequative high values on both sides

    Parameters
    ----------
    values : np.array, size=(m,1), dtype=float
        vector with number of occurances
    base : np.array, size=(m,1), dtype=float
        vector with central values
    neighbors : integer, default=2
        number of neighbors needed in order to be a valley

    Returns
    -------
    dips : np.array, size=(k,1), dtype=float
        array with valley locations
    quantile : np.array, size=(k,1), dtype=float
        array with amount of data under the value set by "dips"

    See Also
    --------
    normalized_sampling_histogram
    """
    for i in range(neighbors):
        if i == 0:
            wall_plu = np.roll(values, +(i + 1))
            wall_min = np.roll(values, -(i + 1))
        else:
            wall_plu = np.vstack((wall_plu, np.roll(values, +(i + 1))))
            wall_min = np.vstack((wall_min, np.roll(values, -(i + 1))))
    wall_plu = np.vstack((values, wall_plu))
    wall_min = np.vstack((values, wall_min))

    if neighbors > 1:
        concav_plu = np.all(np.sign(np.diff(wall_plu, n=1, axis=0))==+1, axis=0)
        concav_min = np.all(np.sign(np.diff(wall_min, n=1, axis=0))==+1, axis=0)

    # select the dips
    selec = np.all(np.vstack((concav_plu, concav_min)), axis=0)
    selec = selec[neighbors - 1:-(neighbors + 1)]
    idx = base[neighbors - 1:-(neighbors + 1)]
    dips = idx[selec]

    # estimate quantiles of the valleys
    cumsum_norm = np.cumsum(values)/np.sum(values)
    cumsum_norm = cumsum_norm[neighbors - 1:-(neighbors + 1)]
    quantiles = cumsum_norm[selec]
    return dips, quantiles

def shadow_image_to_suntrace_list(M, geoTransform, Az, method='nearest'):
    """ given an mask, compute the suntraces with caster and casted locations

    Parameters
    ----------
    M : np.array, size=(m,n), dtype={integer,boolean}
        grid with a shadow classification
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients
    Az : float, unit=degrees
        argument of the illumination
    method : string, default='nearest'
        interpolation of the, the following are implemented:
        - 'nearest' using nearest neighbor, so the coordinates are at the
          resolution of the given mask, thus precisely falling at the pixel
          center
        - 'spline' using cubic spline, so the coordinates are floats, though
          the position might not be much more precise

    Returns
    -------
    suntrace_list : numpy.array, size=(k,4)
        list of coordinates with x,y coordinates of caster and casted
    """
    assert M.ndim>1, ('please provide a multi-dimensional array')
    if method in ('nearest','nearest_neighbor','nn',):
        spl_order = 0
    else:
        spl_order = 3
    # turn towards the sun
    M_rot = ndimage.rotate(M.astype(float), 180-Az,
                           axes=(1, 0), reshape=True, output=None,
                           order=spl_order, mode='constant', cval=-9999)

    # neither ndimage or transform.warp gives the new coordinate system.
    # hence, clumsy interpolation of the grid is applied here
    X,Y = pix_centers(geoTransform, M.shape[0], M.shape[1], make_grid=True)
    X_rot = ndimage.rotate(X, 180-Az,
                           axes=(1, 0), reshape=True, output=None,
                           order=spl_order, mode='constant', cval=X[0,0])
    Y_rot = ndimage.rotate(Y, 180-Az,
                           axes=(1, 0), reshape=True, output=None,
                           order=spl_order, mode='constant', cval=-9999)

    suntrace_list = np.empty((0, 4))
    for k in range(M_rot.shape[1]):
        M_trace = M_rot[:,k] # line of intensities along a sun trace

        shade_class = M_trace>=0.5 # take the upper threshold
        shade_class = shade_class.astype(int)
        # remove shadowtraces that touch the boundary
        trace = np.logical_or(M_trace==-9999, shade_class)
        start_idx = np.argmin(np.cumsum(trace) == \
                              np.linspace(1, trace.size, trace.size))
        end_idx = np.argmin(np.linspace(1, trace.size,trace.size) == \
                            np.cumsum(np.flipud(trace)))
        shade_class[:start_idx] = 0
        shade_class[-end_idx:] = 0

        shade_exten = np.pad(shade_class, (1,1), 'constant',
                             constant_values=0)
        shade_node = np.roll(shade_exten, 1)-shade_exten

        # (col_idx, ) = np.where(IN)
        if -90<Az<90:
            (shade_beg, ) = np.where(shade_node[1::]==-1)
            (shade_end, ) = np.where(shade_node[2::]==+1)
        else:
            (shade_beg, ) = np.where(shade_node[2::]==+1)
            (shade_end, ) = np.where(shade_node[1::]==-1)

        if len(shade_beg)==0: continue

        # coordinate transform, not image transform (loss of points)
        col_idx = k*np.ones(shade_beg.size, dtype=np.int32)
        # index seems shifted...?
        x_beg, x_end = X_rot[shade_beg, col_idx], X_rot[shade_end, col_idx]
        y_beg, y_end = Y_rot[shade_beg, col_idx], Y_rot[shade_end, col_idx]

        suntrace_list = np.vstack((suntrace_list ,
                                   np.transpose(np.vstack(
                                       (x_beg[np.newaxis], y_beg[np.newaxis],
                                        x_end[np.newaxis], y_end[np.newaxis] ))
                                               ) ))

    # remove out of image sun traces
    OUT = np.any(suntrace_list==-9999, axis=1)
    suntrace_list = suntrace_list[~OUT,:]
    return suntrace_list

def shadow_image_to_list(M, geoTransform, s2_path,
                         Zn=None, Az=None, timestamp=None, crs=None, **kwargs):
    """
    Turns the image towards the sun, and looks along each solar ray, hereafter
    the caster and casted locations are written to a txt-file

    Parameters
    ----------
    M : np.array, size=(m,n), dtype={integer,boolean}
        shadow classification
    geoTransform : tuple, size={(1,6), (1,8)}
        affine transformation coefficients
    s2_path : string
        working directory path with metadata
    Zn : float, unit=degrees
        elevation angle of the illumination direction
    Az : float, unit=degrees
        argument of the illumination
    timestamp : string
        the date of acquistion
    crs : {osr.SpatialReference, string}
        the coordinate reference system via GDAL SpatialReference describtion
    """
    if (Zn is None) or (Az is None):
        Zn, Az = read_mean_sun_angles_s2(s2_path)
    if 'out_name' in kwargs:
        out_name = kwargs.get('out_name')
    else:
        out_name = "conn.txt"
    if timestamp is not None:
        assert isinstance(timestamp, str), \
            ('please provide a string for the timestamp')
    if timestamp is not None:
        if not isinstance(crs, str):
            crs = crs.ExportToWkt()

    suntrace_list = shadow_image_to_suntrace_list(M, geoTransform, Az)

    # find azimuth and elevation at cast location
    (sunZn,sunAz) = read_sun_angles_s2(s2_path)

    # if timestamp of imagery is not given, extract it from metadata
    if timestamp is None:
        timing = read_sensing_time_s2(s2_path)
        timestamp = np.datetime_as_string(timing, unit='D')

    if (sunZn is None) or (sunAz is None):
        sunZn, sunAz = Zn*np.ones_like(M), Az*np.ones_like(M)
    else:
        if 'bbox' in kwargs:
            sunZn = sunZn[kwargs['bbox'][0]:kwargs['bbox'][1],
                          kwargs['bbox'][2]:kwargs['bbox'][3]]
            sunAz = sunAz[kwargs['bbox'][0]:kwargs['bbox'][1],
                          kwargs['bbox'][2]:kwargs['bbox'][3]]

    (iC,jC) = map2pix(geoTransform,
                      suntrace_list[:,0].copy(), suntrace_list[:,1].copy())
    iC, jC = np.round(iC).astype(int), np.round(jC).astype(int)
    np.putmask(iC, iC<=0, 0) # move into the array range
    np.putmask(jC, jC<=0, 0)
    np.putmask(iC, iC<=sunZn.shape[0], sunZn.shape[0]-1)
    np.putmask(jC, jC<=sunZn.shape[1], sunZn.shape[1]-1)

    sun_angles = np.transpose(np.vstack((sunAz[iC,jC], sunZn[iC,jC])))

    # write to file
    f = open(os.path.join(s2_path, out_name), 'w')
    # add meta-data if given
    if timestamp is not None:
        print('# time: '+timestamp, file=f)
    if crs is not None:
        print('# proj: '+crs, file=f)

    # add header
    print('# caster_X '+'caster_Y '+'casted_X '+'casted_Y '+\
          'azimuth '+'zenith', file=f)
    # write suntrace list to text file
    for k in range(suntrace_list.shape[0]):
        line = '{:+8.2f}'.format(suntrace_list[k,0]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,1]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,2]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,3]) + ' '
        line += '{:+3.4f}'.format(sun_angles[k,0]) + ' '
        line += '{:+3.4f}'.format(sun_angles[k,1])

        print(line, file=f)
    f.close()
    return

def label_occluder_and_casted(labeling, sunAz):
    """ Find along the edge, the casting and casted pixels of a polygon

    Parameters
    ----------
    labeling : numpy.array, size=(m,n), dtype=integer
        array with labelled polygons
    sunAz : numpy.array, size=(m,n), dtype=integer
        band of azimuth values

    Returns
    -------
    shadowIdx : numpy.array, size=(m,n), dtype=signed integers
        array with numbered pairs, where the caster is the positive number the
        casted is the negative number
    """
    labeling = labeling.astype(np.int64)
    msk = labeling >= 1
    mL, nL = labeling.shape
    shadowIdx = np.zeros((mL, nL), dtype=np.int16)

    inner = ndimage.morphology.binary_erosion(msk)

    bndOrient = cast_orientation(inner.astype(np.float), sunAz)
    del mL, nL, inner

    labList = np.unique(labeling)
    labList = labList[labList != 0]
    for i in labList:
        selec = labeling == i
        labIdx = np.nonzero(selec)
        labImin = np.min(labIdx[0])
        labImax = np.max(labIdx[0])
        labJmin = np.min(labIdx[1])
        labJmax = np.max(labIdx[1])
        subMsk = selec[labImin:labImax, labJmin:labJmax]

        subOrient = np.sign(bndOrient[labImin:labImax,
                            labJmin:labJmax])

        subBound = subMsk ^ ndimage.morphology.binary_erosion(subMsk)
        subOrient[~subBound] = 0  # remove other boundaries

        subAz = sunAz[labImin:labImax,
                      labJmin:labJmax]  # [loc] # subAz = sunAz[loc]

        subWhe = np.nonzero(subMsk)
        ridgIdx = subOrient[subWhe[0], subWhe[1]] == 1

        ridgeI = subWhe[0][ridgIdx]
        ridgeJ = subWhe[1][ridgIdx]


        # boundary of the polygon that receives cast shadow
        cast = subOrient == -1

        m, n = subMsk.shape
        print("For shadowpolygon #%s: Its size is %s by %s," +
              " connecting %s pixels in total" % (i, m, n, len(ridgeI)))

        for x in range(len(ridgeI)):  # loop through all occluders
            sunDir = subAz[ridgeI[x]][ridgeJ[x]]  # degrees [-180 180]

            # Bresenham's line algorithm
            dI = -np.cos(np.radians(subAz[ridgeI[x]][ridgeJ[x]]))  # -cos # flip axis to get from world into image coords
            dJ = -np.sin(np.radians(subAz[ridgeI[x]][ridgeJ[x]]))  # -sin #
            brd = 3 # add aditional cast borders to the suntrace
            if abs(sunDir) > 90:  # northern hemisphere
                if dI > dJ:
                    rr = np.arange(start=0, stop=ridgeI[x]+brd, step=1)
                    cc = np.flip(np.round(rr * dJ), axis=0) + ridgeJ[x]
                else:
                    cc = np.arange(start=0, stop=ridgeJ[x]+brd, step=1)
                    rr = np.flip(np.round(cc * dI), axis=0) + ridgeI[x]
            else:  # southern hemisphere
                if dI > dJ:
                    rr = np.arange(start=ridgeI[x]-brd, stop=m, step=1)
                    cc = np.round(rr * dJ) + ridgeJ[x]
                else:
                    cc = np.arange(start=ridgeJ[x]-brd, stop=n, step=1)
                    rr = np.round(cc * dI) + ridgeI[x]
                    # generate cast line in sub-image
            rr, cc = rr.astype(np.int64), cc.astype(np.int64)
            IN = (cc >= 0) & (cc <= n) & (rr >= 0) & (
                            rr <= m)  # inside sub-image
            if not IN.any(): continue

            rr = rr[IN]
            cc = cc[IN]

            subCast = np.zeros((m, n), dtype=np.uint8)
            try:
                subCast[rr, cc] = 1
            except IndexError:
                continue

                # find closest casted
            castedHit = cast & subCast
            (castedIdx) = np.where(castedHit[subWhe[0], subWhe[1]] == 1)
            castedI = subWhe[0][castedIdx[0]]
            castedJ = subWhe[1][castedIdx[0]]
            del IN, castedIdx, castedHit, subCast, rr, cc, dI, dJ, sunDir

            if len(castedI) > 1:
                # do selection of the closest casted
                dist = np.hypot(castedI-ridgeI[x], castedJ-ridgeJ[x])
                idx = np.where(dist == np.amin(dist))
                castedI = castedI[idx[0]]
                castedJ = castedJ[idx[0]]
            elif len(castedI) > 0:
                # write out
                shadowIdx[ridgeI[x] + labImin][ridgeJ[x] + labJmin] = +(x+1)
                shadowIdx[castedI[0] + labImin][castedJ[0] + labJmin] = -(x+1)

        print("polygon done")
    return shadowIdx

def list_occluder_and_casted(labels, sunZn, sunAz, geoTransform):
    """ Find along the edge, the casting and casted pixels of a polygon

    Parameters
    ----------
    labels : np.array, size=(m,n)
        array with labelled polygons
    sunAz : np.array, size=(m,n)
        band of azimuth values
    sunZn : np.array, size=(m,n)
        band of zenith values
    geoTransform : tuple, size=(1,6)
        affine transformation coefficients

    Returns
    -------
    castList : list, size=(k,6)
        array with image coordinates of caster and casted with the sun angles
        of the caster
    """
    msk = labels > 1
    labels = labels.astype(np.int32)
    mskOrient = cast_orientation(msk.astype(np.float), sunAz)
    mskOrient = np.sign(mskOrient)

    castList = []
    for shp, val in shapes(labels, mask=msk, connectivity=8):
        if val == 0: continue

        # get ridge coordinates
        polygoon = shape(shp)
        polyRast = labels == val  # select the polygon
        polyInnr = ndimage.binary_erosion(polyRast,
                                          np.ones((3, 3),
                                                  dtype=bool))
        polyBoun = np.logical_xor(polyRast, polyInnr)
        polyWhe = np.nonzero(polyBoun)
        ridgIdx = mskOrient[polyWhe[0], polyWhe[1]] == 1
        ridgeI = polyWhe[0][ridgIdx]
        ridgeJ = polyWhe[1][ridgIdx]
        del polyRast, polyInnr, polyBoun, polyWhe, ridgIdx

        for idx,_ in enumerate(ridgeI):  # ridgeI:
            castLine = find_polygon_intersect(ridgeI[idx],ridgeJ[idx], polygoon,
                                              sunAz[ridgeI[idx]][ridgeJ[idx]],
                                              sunZn[ridgeI[idx]][ridgeJ[idx]],
                                              geoTransform)
            if castLine is not None: castList.append(castLine)
            del castLine
    return castList

def find_polygon_intersect(ridge_i,ridge_j,polygoon,sun_az,sun_zn,
                                     geoTransform):
    """

    Parameters
    ----------
    ridge_i : integer
        collumn coordinate of a location within the image that is a caster
    ridge_j : integer
        row coordinate of a location within the image that is a caster
    polygoon : rasterio shape
        polgon of a not illuminated region
    sun_az : float, unit=degrees
        argument of the illumination
    sun_zn : float, unit=degrees
        elevation angle of the illumination direction
    geoTransform : tuple, size=(1,6)
        affine transformation coefficients

    Returns
    -------
    castLine : np.array, size=(,6)
        edge locations of caster and casted, together with the sun angles
    """
    try:
        castLine = LineString([[ridge_j, ridge_i],
                               [ridge_j - (np.sin(np.radians(sun_az)) * 1e4),
                                ridge_i + (np.cos(np.radians(sun_az)) * 1e4)]])
    except IndexError:
        return
    try:
        castEnd = polygoon.intersection(castLine)
    except TopologicalError:
        # somehow the exterior of the polygon crosses or touches
        # itself, making it a LinearRing
        polygoon = polygoon.buffer(0)
        castEnd = polygoon.intersection(castLine)

    if castEnd.geom_type == 'LineString':
        castEnd = castEnd.coords[:]
    elif castEnd.geom_type == 'MultiLineString':
        cEnd = []
        for m in list(castEnd):
            cEnd += m.coords[:]
        castEnd = cEnd
        del m, cEnd
    elif castEnd.geom_type == 'GeometryCollection':
        cEnd = []
        for m in range(len(castEnd)):
            cEnd += castEnd[m].coords[:]
        castEnd = cEnd
        del m, cEnd
    elif castEnd.geom_type == 'Point':
        castEnd = []
    else:
        print('something went wrong?')

    if len(castEnd) <= 1: return

    # find closest intersection
    occluder = Point(ridge_j, ridge_i)
    dists = [Point(c).distance(occluder) for c in castEnd]
    dists = [float('Inf') if i == 0 else i for i in dists]
    castIdx = dists.index(min(dists))
    casted = castEnd[castIdx]

    # transform to UTM and append to array
    ridge_x,ridge_y = pix2map(geoTransform, ridge_i, ridge_j)
    cast_x,cast_y = pix2map(geoTransform, casted[1], casted[0])

    castLine = np.array([ridge_x, ridge_y, cast_x, cast_y,
                         sun_az, sun_zn])
    return castLine

