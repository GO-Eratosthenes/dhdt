import os
import glob

import pandas as pd

from dhdt.generic.handler_xml import get_root_of_table

def _get_file_meta(theia, vn_df, dat_dir, mtype='Image'):

    if mtype in ('Image'):
        noi = 'Reflectance'# nature of interest
    else:
        noi = 'Aberrant_Pixels'

    im_listing = None
    for child in theia:
        if child.tag == mtype+'_List':
            im_listing = child
    assert (im_listing is not None), ('metadata not in xml file')

    im_list = None
    for child in im_listing:
        if child.tag == mtype:
            if child[0][0].text == noi:
                im_list = child
    assert (im_list is not None), ('metadata not in xml file')

    file_list = None
    for child in im_list:
        if child.tag == mtype+'_File_List':
            file_list = child
    assert (file_list is not None), ('metadata not in xml file')

    im_paths, band_id = [], []
    for im_loc in file_list:
        full_path = os.path.join(dat_dir,
                                 os.path.split(im_loc.text)[1])
        im_paths.append(full_path)
        band_id.append(im_loc.get('band_id'))

    band_path = pd.Series(data=im_paths, index=band_id,
                          name=mtype.lower()+"path")
    vn_df_new = pd.concat([vn_df, band_path], axis=1, join="inner")
    return vn_df_new

def get_vn_image_locations(fname,vn_df):
    """
    VENµS imagery are placed within a folder structure, this function finds the
    paths to the metadata

    Parameters
    ----------
    fname : string
        path string to the VENµS folder
    vn_df : pandas.dataframe
        index of the bands of interest

    Returns
    -------
    vn_df : pandas.dataframe
        dataframe series with relative folder and file locations of the bands
    """
    if os.path.isdir(fname): # only directory is given, search for metadata
        fname = glob.glob(os.path.join(fname,'*MTD*.xml'))[0]
    assert os.path.exists(fname), 'metafile does not seem to be present'

    root = get_root_of_table(fname)

    prod_org = None
    for child in root:
        if child.tag == 'Product_Organisation':
            prod_org = child
    assert(prod_org is not None), ('metadata not in xml file')

    theia = None
    for child in prod_org:
        if child.tag == 'Muscate_Product':
            theia = child
    assert(theia is not None), ('metadata not in xml file')

    dat_dir = os.path.split(fname)[0]
    vn_df_new = _get_file_meta(theia, vn_df, dat_dir, mtype='Image')
    vn_df_new = _get_file_meta(theia, vn_df_new, dat_dir, mtype='Mask')
    return vn_df_new

def get_vn_mean_view_angles(fname,vn_df):
    """
    VENµS imagery are placed within a folder structure, this function finds the
    paths to the metadata

    Parameters
    ----------
    fname : string
        path string to the VENµS folder
    vn_df : pandas.dataframe
        index of the bands of interest

    Returns
    -------
    vn_df : pandas.dataframe
        dataframe series with relative folder and mean view of each detector.
        given by "zenith_mean" and "azimuth_mean"
    """
    if os.path.isdir(fname): # only directory is given, search for metadata
        fname = glob.glob(os.path.join(fname,'*MTD*.xml'))[0]
    assert os.path.exists(fname), 'metafile does not seem to be present'

    root = get_root_of_table(fname)

    geom_info = None
    for child in root:
        if child.tag == 'Geometric_Informations':
            geom_info = child
    assert(geom_info is not None), ('metadata not in xml file')

    mean_list = None
    for child in geom_info:
        if child.tag == 'Mean_Value_List':
            mean_list = child
    assert (mean_list is not None), ('metadata not in xml file')

    view_list = None
    for child in mean_list:
        if child.tag == 'Mean_Viewing_Incidence_Angle_List':
            view_list = child
    assert (view_list is not None), ('metadata not in xml file')

    zn, az, det_id = [], [], []
    for view_child in view_list:
        zn.append(float(view_child[0].text))
        az.append(float(view_child[1].text))
        det_id.append(int(view_child.get('detector_id')))

    az_view = pd.Series(data=az, index=det_id, name="azimuth_mean")
    vn_df_new = pd.concat([vn_df, az_view], axis=1, join="inner")

    return