# generic libraries
import glob
import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from ..generic.handler_xml import get_root_of_table
from ..generic.mapping_io import read_geo_image


# dove-C
def list_central_wavelength_dc():
    center_wavelength = {"B1": 485., "B2": 545., "B3": 630., "B4": 820.}
    # convert from nm to µm
    center_wavelength = {k: v / 1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {"B1": 60., "B2": 90., "B3": 80., "B4": 80.}
    # convert from nm to µm
    full_width_half_max = {k: v / 1E3 for k, v in full_width_half_max.items()}
    gsd = {"B1": 3., "B2": 3., "B3": 3., "B4": 3.}
    bandid = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
    acquisition_order = {"B1": 1, "B2": 1, "B3": 1, "B4": 2}
    # along_track_view_angle =
    # field_of_view =
    common_name = {"B1": 'blue', "B2": 'green', "B3": 'red', "B4": 'nir'}
    d = {
        "center_wavelength":
        pd.Series(center_wavelength, dtype=np.dtype('float')),
        "full_width_half_max":
        pd.Series(full_width_half_max, dtype=np.dtype('float')),
        "gsd":
        pd.Series(gsd, dtype=np.dtype('float')),
        "common_name":
        pd.Series(common_name, dtype='str'),
        "acquisition_order":
        pd.Series(acquisition_order, dtype=np.dtype('int64')),
        "bandid":
        pd.Series(bandid, dtype=np.dtype('int64'))
    }
    df = pd.DataFrame(d)
    return df


# dove-R
def list_central_wavelength_dr():
    center_wavelength = {"B1": 485., "B2": 545., "B3": 630., "B4": 820.}
    # convert from nm to µm
    center_wavelength = {k: v / 1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {"B1": 60., "B2": 90., "B3": 80., "B4": 80.}
    # convert from nm to µm
    full_width_half_max = {k: v / 1E3 for k, v in full_width_half_max.items()}
    gsd = {"B1": 3., "B2": 3., "B3": 3., "B4": 3.}
    bandid = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
    # along_track_view_angle =
    field_of_view = {"B1": 3., "B2": 3., "B3": 3., "B4": 3.}
    acquisition_order = {"B1": 4, "B2": 1, "B3": 2, "B4": 3}
    common_name = {"B1": 'blue', "B2": 'green', "B3": 'red', "B4": 'nir'}
    d = {
        "center_wavelength":
        pd.Series(center_wavelength, dtype=np.dtype('float')),
        "full_width_half_max":
        pd.Series(full_width_half_max, dtype=np.dtype('float')),
        "gsd":
        pd.Series(gsd, dtype=np.dtype('float')),
        "common_name":
        pd.Series(common_name, dtype='str'),
        "field_of_view":
        pd.Series(field_of_view, dtype=np.dtype('float')),
        "acquisition_order":
        pd.Series(acquisition_order, dtype=np.dtype('int64')),
        "bandid":
        pd.Series(bandid, dtype=np.dtype('int64'))
    }
    df = pd.DataFrame(d)
    return df


def list_central_wavelength_sd(num_bands=4):
    center_wavelength = {
        "B1": 443.,
        "B2": 490.,
        "B3": 531.,
        "B4": 565.,
        "B5": 665.,
        "B6": 610.,
        "B10": 705.,
        "B13": 865.
    }
    # convert from nm to µm
    center_wavelength = {k: v / 1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {
        "B1": 20.,
        "B2": 50.,
        "B3": 36.,
        "B4": 36.,
        "B5": 31.,
        "B6": 20.,
        "B10": 15.,
        "B13": 40.
    }
    # convert from nm to µm
    full_width_half_max = {k: v / 1E3 for k, v in full_width_half_max.items()}
    gsd = {
        "B1": 12.,
        "B2": 3.,
        "B3": 3.,
        "B4": 3.,
        "B5": 3.,
        "B6": 6.,
        "B10": 6.,
        "B13": 6.,
    }
    if num_bands == 4:
        bandid = {"B2": 0, "B4": 1, "B5": 2, "B13": 3}
    else:
        bandid = {
            "B1": 0,
            "B2": 1,
            "B3": 2,
            "B4": 3,
            "B5": 4,
            "B6": 5,
            "B10": 6,
            "B13": 7,
        }
    # along_track_view_angle =
    field_of_view = {
        "B1": 4.,
        "B2": 4.,
        "B3": 4.,
        "B4": 4.,
        "B5": 4.,
        "B6": 4.,
        "B10": 4.,
        "B13": 4.
    }
    acquisition_order = {
        "B1": 8,
        "B2": 1,
        "B3": 3,
        "B4": 4,
        "B5": 2,
        "B6": 5,
        "B10": 6,
        "B13": 7
    }
    common_name = {
        "B1": 'coastalblue',
        "B2": 'blue',
        "B3": 'green I',
        "B4": 'green II',
        "B5": 'red',
        "B6": 'yellow',
        "B10": 'rededge I',
        "B13": 'nir',
    }
    d = {
        "center_wavelength":
        pd.Series(center_wavelength, dtype=np.dtype('float')),
        "full_width_half_max":
        pd.Series(full_width_half_max, dtype=np.dtype('float')),
        "gsd":
        pd.Series(gsd, dtype=np.dtype('float')),
        "common_name":
        pd.Series(common_name, dtype='str'),
        "field_of_view":
        pd.Series(field_of_view, dtype=np.dtype('float')),
        "acquisition_order":
        pd.Series(acquisition_order, dtype=np.dtype('int64')),
        "bandid":
        pd.Series(bandid, dtype=np.dtype('int64'))
    }
    df = pd.DataFrame(d)
    return df


def read_band_ps(dir_path, fname, no_dat=None):
    """
    This function takes as input the PlanetScope file name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array

    Parameters
    ----------
    dir_path : string
        path of the folder
    fname : string
        PlanetScope image name

    Returns
    -------
    data : np.array, size=(_,_)
        array of the band image
    spatialRef : string
        projection
    geoTransform : tuple
        affine transformation coefficients
    targetprj : osr.SpatialReference object
        spatial reference

    See Also
    --------
    list_central_wavelength_ps

    """
    if fname is not None:
        fname = os.path.join(dir_path, fname)
    else:
        fname = dir_path

    assert len(glob.glob(fname)) != 0, ('file does not seem to be present')

    data, spatialRef, geoTransform, targetprj = \
        read_geo_image(glob.glob(fname)[0], no_dat=no_dat)
    return data, spatialRef, geoTransform, targetprj


def read_data_ps(f_full, df):  # todo: doctstring
    data, spatialRef, geoTransform, targetprj = read_band_ps(f_full, None)
    data = data[..., df['bandid'].to_numpy()]
    return data, spatialRef, geoTransform, targetprj


def read_detector_type_ps(path, fname='*_metadata.xml'):
    # todo: elementree does not work nicely,
    # while "from xml.dom import minidom" seems to work better
    """
    Notes
    -----
    Dove Pilot has the following sensor array:

        .. code-block:: text

            <---- 4032 pixels ---->
          ┌------------------------┐ ^
          |                        | |
          |                        | | 2672 pixels
          |         RGB            | |
          |     (Bayer pattern)    | |
          |                        | |
          └------------------------┘ v

    PlanetScope (PS Instrument) has the following sensor array:

        .. code-block:: text

            <---- 6600 pixels ---->
          ┌------------------------┐ ^
          |                        | |
          |         RGB            | | 4400 pixels
          |     (Bayer pattern)    | |
          ├------------------------┤ |
          |                        | |
          |         NIR            | |
          |                        | |
          └------------------------┘ v

    Dove-R (PS2.SD Instrument) has the following sensor array:

        .. code-block:: text

            <---- 6600 pixels ---->
          ┌------------------------┐ ^
          |         Green II (B04) | |
          ├------------------------┤ | 4400 pixels
          |         Red      (B05) | |
          ├------------------------┤ |
          |         NIR      (B13) | |
          ├------------------------┤ | ^
          |         Blue     (B02) | | | 1100 pixels
          └------------------------┘ v v

    Superdove(PSB.SD Instrument) has the following bands:
    Coastal Blue, Blue, Green, Green I, Green II, Yellow, Red, Rededge, NIR

        .. code-block:: text

            <---- 8800 pixels ---->
          ┌------------------------+ ^
          |         Blue   (B02)   | |
          ├------------------------┤ | 5280 pixels
          |         Red    (B05)   | |
          ├------------------------┤ |
          |         Green  (B03)   | |
          ├------------------------┤ |
          |         GreenII(B04)   | |
          ├------------------------┤ |
          |         Yellow (B06)   | |
          ├------------------------┤ |
          |         Rededge(B10)   | |
          ├------------------------┤ |
          |         NIR    (B13)   | |
          ├------------------------┤ | ^
          |         Coastal Blue   | | | 660 pixels
          └------------------------┘ v v

    """
    root = get_root_of_table(path, fname)
    dtyp = root[2][0][1][0][0].text
    return dtyp


def planetscope_noradid():
    url_status = 'https://ephemerides.planet-labs.com/operational_status.txt'
    df = pd.read_csv(url_status,
                     escapechar='#',
                     delimiter="\t",
                     delim_whitespace=False,
                     skiprows=5,
                     error_bad_lines=False)
    df = df.rename(columns=lambda s: s.strip())
    # test: list(df.columns.values)
    df = df.rename(columns={'Planet_Name': 'platform', 'CATID': 'NORADID'})
    return df
