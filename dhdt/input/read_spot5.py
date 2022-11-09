import pandas as pd

def list_platform_metadata_s5():
    s5_dict = {
        'COSPAR': '2002-021A',
        'NORAD': 27421,
        'full_name': 'Satellite Pour l’Observation de la Terre',
        'instruments': {'HRS', 'HRG', 'Végétation', 'DORIS'},
        'constellation': 'SPOT',
        'launch': '2002-05-04',
        'orbit': 'sso',
        'equatorial_crossing_time': '10:30'}
    return s5_dict

def list_central_wavelength_hrg():
    """ create dataframe with metadata about SPOT5

    Returns
    -------
    df : datafram
        metadata and general multispectral information about the MSI
        instrument that is onboard SPOT5, having the following collumns:

            * center_wavelength, unit=µm : central wavelength of the band
            * full_width_half_max, unit=µm : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * common_name : general name of the band, if applicable

    Example
    -------
    make a selection by name:

    >>> boi = ['blue', 'green', 'red', 'nir']
    >>> s5_df = list_central_wavelength_hrg()
    >>> s5_df = s4_df[s4_df['common_name'].isin(boi)]
    >>> s5_df
             wavelength  bandwidth  resolution       name  bandid
    B02         492         66          10           blue       1
    B03         560         36          10          green       2
    B04         665         31          10            red       3
    B08         833        106          10  near infrared       7

    """
    center_wavelength = {"B01": 550, "B02": 650, "B03": 840, "B04": 1670,
                  }
    full_width_half_max = {"B01": 90, "B02": 70, "B03": 110, "B04": 170,
                 }
    bandid = {"B01": 0, "B02": 1, "B03": 2, "B04": 3,
                  }
    gsd = {"B01": 10, "B02": 10, "B03": 10, "B04": 20}
    field_of_view = {"B01": 4.0, "B02": 4.0, "B03": 4.0, "B04": 4.0,}
    common_name = {"B01": 'green', "B02" : 'red',
                   "B03" : 'nir',  "B04" : 'swir16',
            }
    d = {
         "center_wavelength": pd.Series(center_wavelength),
         "full_width_half_max": pd.Series(full_width_half_max),
         "gsd": pd.Series(gsd),
         "field_of_view": pd.Series(field_of_view),
         "common_name": pd.Series(common_name),
         "bandid": pd.Series(bandid)
         }
    df = pd.DataFrame(d)
    return df
