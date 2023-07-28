import os
import pystac
import fnmatch

from scipy import ndimage

from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import map2pix, pix_centers, make_same_size
from dhdt.generic.handler_xml import get_root_of_table
from dhdt.generic.handler_sentinel2 import list_platform_metadata_s2a, \
    list_platform_metadata_s2b
from dhdt.generic.orbit_tools import calculate_correct_mapping, \
    remap_observation_angles
from dhdt.input.read_sentinel2 import get_view_angles_s2_from_root, \
    get_crs_s2_from_root, read_detector_mask, \
    get_geotransform_s2_from_root, read_sun_angles_s2


def read_stac_catalog(urlpath, stac_io=None):
    """
    Read STAC catalog from URL/path

    Parameters
    ----------
    urlpath : string
        URL/path to the catalog root
    stac_io : STAC IO instance
        to read the catalog

    Returns
    -------
    PySTAC Catalog object
    """
    urlpath = urlpath \
        if urlpath.endswith("catalog.json") \
        else f"{urlpath}/catalog.json"
    catalog = pystac.Catalog.from_file(urlpath, stac_io=stac_io)
    return catalog


def load_bands(item, sat_df, geoTransform):
    """
    Load input raster files

    Parameters
    ----------
    item : dictionary
        with items to local input files
    sat_df : pandas.DataFrame
        information about the bands of interest to use

    Returns
    -------
    bands : list with numpy.ndarrays, size=(m,n)
    """
    bands = {}
    for _, v in sat_df["common_name"].items():
        full_path = item.assets[v].get_absolute_href()
        band, _, bandTransform, _ = read_geo_image(full_path)
        if bandTransform != geoTransform:
            band = make_same_size(band, bandTransform, geoTransform)
        bands[v] = band
    return bands


def load_cloud_cover_s2(item, geoTransform):
    full_path = item.assets['scl'].get_absolute_href()
    scl, _, sclTransform, _ = read_geo_image(full_path)

    if sclTransform != geoTransform:
        x_grd, y_grd = pix_centers(geoTransform)
        i_grd, j_grd = map2pix(sclTransform, x_grd, y_grd)
        scl = ndimage.map_coordinates(scl, [i_grd, j_grd],
                                      order=0,
                                      mode='nearest')
    return scl


def get_items_via_id_s2(catalog_L1C, catalog_L2A, item_id_L1C):
    item_L1C = catalog_L1C.get_item(item_id_L1C, recursive=True)
    item_L2A = None

    platform = item_L1C.properties["platform"][-2:].upper()
    datetime = item_L1C.datetime.strftime("%Y%m%dT%H%M%S")
    tile_id = item_L1C.properties["s2:mgrs_tile"]
    item_L2A_id_match = f"S{platform}_MSIL2A_{datetime}_*_T{tile_id}_*"
    for item in catalog_L2A.get_all_items():
        if fnmatch.fnmatch(item.id, item_L2A_id_match):
            item_L2A = item
            break
    return item_L1C, item_L2A


def load_input_metadata_s2(item, s2_df, geoTransform):
    """
    Load and preprocess spatial metadata at pixel level

    Parameters
    ----------
    item :
    :param bbox_idx: crop the metadata using the provided array indices
    :param transform: geo-transform
    :return: zenith and azimuth sun angles, mean zenith and azimuth sun angles,
        zenith and azimuth view angles
    """

    # get sun angles
    if item.properties['s2:product_uri'].startswith('S2A'):
        s2_dict = list_platform_metadata_s2a()
    else:
        s2_dict = list_platform_metadata_s2b()

    # estimate orbital flightpath from rough observation angles
    root = get_root_of_table(
        item.assets['granule_metadata'].get_absolute_href())
    view_az_grd, view_zn_grd, bnd, det, grdTransform = \
        get_view_angles_s2_from_root(root)
    crs = get_crs_s2_from_root(root)

    Ltime, lat, lon, radius, inclination, period, time_para, combos = \
        calculate_correct_mapping(view_zn_grd, view_az_grd, bnd, det,
                                  grdTransform, crs, sat_dict=s2_dict)

    # map to domain of interest
    im_x, im_y = pix_centers(geoTransform, make_grid=True)
    # todo
    full_href = item.assets['granule_metadata'].get_absolute_href()
    qi_path = os.path.dirname(full_href)

    det_stack = read_detector_mask(qi_path, s2_df, geoTransform)
    orgTransform = get_geotransform_s2_from_root(root)

    view_zn, view_az, _ = remap_observation_angles(Ltime, lat, lon, radius,
                                                   inclination, period,
                                                   time_para, combos, im_x,
                                                   im_y, det_stack, s2_df,
                                                   orgTransform, crs)

    sun_zn, sun_az = read_sun_angles_s2(os.path.dirname(full_href),
                                        fname=os.path.basename(full_href))

    sun_zn = make_same_size(sun_zn, orgTransform, geoTransform)
    sun_az = make_same_size(sun_az, orgTransform, geoTransform)

    return sun_zn, sun_az, view_zn, view_az
