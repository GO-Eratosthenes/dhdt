import os
import warnings
import pandas as pd

def in_which_spectral_range(wavelength):
    if wavelength<1000:
        return 'vnir'
    if wavelength<3000:
        return 'swir'
    else:
        return 'tir'

def get_as_image_locations(f_path, as_df):
    if os.path.isdir(f_path): # level-3 data is in a folder
        as_df_new = get_as_image_locations_l3(f_path, as_df)
    else:
        as_df_new = get_as_image_locations_l1(f_path, as_df)
    return as_df_new

def get_as_image_locations_l1(f_path, as_df):
    assert os.path.exists(f_path)
    if not f_path[-4:].lower()=='.hdf':
        warnings.warn("no hdf-file given, but this might work")

    im_paths, band_id = [], []
    for i, row in as_df.iterrows():
        im_paths.append(f_path)
        band_id.append(i)
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
        assert os.path.exists(f_full), \
            ('something is wrong, please look if file structure has changed')
        im_paths.append(f_full)
        band_id.append(i)

    band_path = pd.Series(data=im_paths, index=band_id, name="filepath")
    as_df_new = pd.concat([as_df, band_path], axis=1, join="inner")
    return as_df_new