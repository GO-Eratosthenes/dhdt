import pandas as pd


def list_platform_metadata_s4():
    s4_dict = {
        'COSPAR': '1998-017A',
        'NORAD': 25260,
        'full_name': 'Satellite Pour l’Observation de la Terre',
        'instruments': {'Végétation', 'HRVIR', 'DORIS'},
        'constellation': 'SPOT',
        'launch': '1998-03-24',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:30'
    }
    return s4_dict


def list_central_wavelength_s4():
    """ create dataframe with metadata about SPOT4

    Returns
    -------
    df : datafram
        metadata and general multispectral information about the MSI
        instrument that is onboard SPOT4, having the following collumns:

            * center_wavelength, unit=µm : central wavelength of the band
            * full_width_half_max, unit=µm : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * common_name : general name of the band, if applicable

    Examples
    --------
    make a selection by name:

    >>> boi = ['red', 'green', 'blue', 'near infrared']
    >>> s4_df = list_central_wavelength_s2()
    >>> s4_df = s4_df[s4_df['name'].isin(boi)]
    >>> s4_df
             wavelength  bandwidth  resolution       name  bandid
    B02         492         66          10           blue       1
    B03         560         36          10          green       2
    B04         665         31          10            red       3
    B08         833        106          10  near infrared       7

    """
    center_wavelength = {
        "B01": 545,
        "B02": 640,
        "B03": 740,
        "B04": 1665,
    }
    # convert from nm to µm
    center_wavelength = {k: v / 1E3 for k, v in center_wavelength.items()}
    full_width_half_max = {
        "B01": 90,
        "B02": 70,
        "B03": 100,
        "B04": 170,
    }
    # convert from nm to µm
    full_width_half_max = {k: v / 1E3 for k, v in full_width_half_max.items()}
    bandid = {
        "B01": 0,
        "B02": 1,
        "B03": 2,
        "B04": 3,
    }
    resolution = {
        "B01": 20,
        "B02": 20,
        "B03": 20,
        "B04": 20,
    }
    common_name = {
        "B01": 'green',
        "B02": 'red',
        "B03": 'near infrared',
        "B04": 'shortwave infrared',
    }
    d = {
        "center_wavelength": pd.Series(center_wavelength),
        "full_width_half_max": pd.Series(full_width_half_max),
        "resolution": pd.Series(resolution),
        "common_name": pd.Series(name),
        "bandid": pd.Series(bandid)
    }
    df = pd.DataFrame(d)
    return df
