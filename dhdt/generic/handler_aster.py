import os
import warnings
import pandas as pd

from dhdt.generic.handler_dat import get_list_files

def list_platform_metadata_tr():
    as_dict = {
        'COSPAR': '1999-068A',
        'NORAD': 25994,
        'instruments': {'ASTER', 'CERES', 'MISR', 'MODIS', 'MOPITT'},
        'constellation': 'terra',
        'launch_date': '+1999-12-18',
        'orbit': 'sso',
        'mass': 4864., # [kg]
        'inclination': 98.1054, # https://www.n2yo.com/satellite/?s=25994
        'equatorial_crossing_time': '10:30'
        }
    return as_dict

def in_which_spectral_range(wavelength):
    if wavelength<1000:
        return 'vnir'
    if wavelength<3000:
        return 'swir'
    else:
        return 'tir'

def get_as_image_locations(f_path, as_df):
    """

    Parameters
    ----------
    f_path
    as_df

    Returns
    -------

    Notes
    -----
    The file structure of ASTER can be in different forms. For L1A in tif-format
    this looks like:

    .. code-block:: text

        * AST_L1A_XX..._XXX..._XXX
        ├ AST_L1A_XX..._XXX..._XXX.Ancillary_Data.txt
        ├ AST_L1A_XX..._XXX..._XXX.VNIR_BandY.AttitudeAngle.txt
        ├ AST_L1A_XX..._XXX..._XXX.VNIR_BandY.AttitudeRate.txt
        ├ AST_L1A_XX..._XXX..._XXX.VNIR_BandY.AttitudeLatitude.txt
        ├ AST_L1A_XX..._XXX..._XXX.VNIR_Band1.ImageData.tif
        └ AST_L1A_XX..._XXX..._XXX.SWIR_Band4.ImageData.tif

    For L1A in hdf-format this looks like:

    .. code-block:: text

        * AST_L1A_XX..._XXX..._XXX.hdf

    The file structure of ASTER can be in different forms. For L3 in tif-format
    this looks like:

    .. code-block:: text

        * XXXXXXX
        ├ data1.l3a.demsh
        ├ data1.l3a.vnirsh
        ├ data1.l3a.demzs.tif
        ├ data1.l3a.swirX.tif
        ├ data1.l3a.tirX.tif
        ├ data1.l3a.vnirX.tif
        └ data1.l3a.vnir3n.tif

    """
    if os.path.isdir(f_path): # level-3 data is in a folder
        file_list = get_list_files(f_path, '.tif')
        if file_list[0].startswith('data1.l3a'):
            as_df_new = get_as_image_locations_l3(f_path, as_df)
    as_df_new = get_as_image_locations_l1(f_path, as_df)
    return as_df_new

def get_as_image_locations_l1(f_path, as_df):
    assert os.path.isdir(f_path), 'directory does not seem to exist'

    im_paths, band_id = [], []
    if f_path.endswith('.hdf'):
        for i,_ in as_df.iterrows():
            im_paths.append(f_path)
            band_id.append(i)
    else:
        for i,_ in as_df.iterrows():
            ending = 'Band' + i[1:].lstrip('0') + '.ImageData.tif'
            as_list = get_list_files(f_path, ending)
            if len(as_list)==1:
                im_paths.append(os.path.join(f_path, as_list[0]))
                band_id.append(i)
            else:
                print('file does not seem to be present')

    band_path = pd.Series(data=im_paths, index=band_id, name="filepath")
    as_df_new = pd.concat([as_df, band_path], axis=1, join="inner")
    return as_df_new

def get_as_image_locations_l3(f_path, as_df):
    """
    The ASTER Level3 imagery are placed within a folder structure, these are
    searched for and placed within the dataframe.

    Parameters
    ----------
    f_path : string
        path string to the ASTER folder
    as_df : pandas.dataframe
        dataframe structure with the different bands of the instrument

    Returns
    -------
    as_df_new : pandas.dataframe
        dataframe series with relative folder and file locations of the bands

    Notes
    -----
    The folder structure for Level3 ASTER data is the following:

    .. code-block:: text

        * ASTB...tar.bz2
        * [YYMMDDHH]...        <- folder name depends on the acquisition date
        ├ data1.l3a.demsh
        ├ data1.l3a.gh
        ├ data1.l3a.swirsh
        ├ data1.l3a.tirsh
        ├ data1.l3a.vnirsh
        ├ data1.l3a.demzs.tif
        ├ data1.l3a.swir4.tif
        ├ data1.l3a.swir5.tif
        ├ data1.l3a.swir6.tif
        ├ data1.l3a.swir7.tif
        ├ data1.l3a.swir8.tif
        ├ data1.l3a.swir9.tif
        ├ data1.l3a.tir10.tif
        ├ data1.l3a.tir11.tif
        ├ data1.l3a.tir12.tif
        ├ data1.l3a.tir13.tif
        ├ data1.l3a.tir14.tif
        ├ data1.l3a.vnir1.tif
        ├ data1.l3a.vnir2.tif
        └ data1.l3a.vnir3n.tif
    """
    assert os.path.isdir(f_path)

    im_paths, band_id = [], []
    for i, row in as_df.iterrows():
        spec_type = in_which_spectral_range(row['center_wavelength'])
        bid = i[1:]
        if bid[-1] in ('N',):
            f_name = 'data1.l3a.'+spec_type+'3n.tif'
        else:
            f_name = 'data1.l3a.'+spec_type+str(int(bid))+'.tif'

        f_full = os.path.join(f_path, f_name)
        assert os.path.isfile(f_full), \
            ('something is wrong, please look if file structure has changed')
        im_paths.append(f_full)
        band_id.append(i)

    band_path = pd.Series(data=im_paths, index=band_id, name="filepath")
    as_df_new = pd.concat([as_df, band_path], axis=1, join="inner")
    return as_df_new