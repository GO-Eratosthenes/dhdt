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
        'equatorial_crossing_time': '10:30'}
    return s4_dict

def list_central_wavelength_s4():
    """ create dataframe with metadata about SPOT4

    Returns
    -------
    df : datafram
        metadata and general multispectral information about the MSI
        instrument that is onboard SPOT4, having the following collumns:

            * wavelength : central wavelength of the band
            * bandwidth : extent of the spectral sensativity
            * bandid : number for identification in the meta data
            * resolution : spatial resolution of a pixel
            * name : general name of the band, if applicable

    Example
    -------
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
    wavelength = {"B01": 545, "B02": 640, "B03": 740, "B04": 1665,
                  }
    bandwidth = {"B01": 90, "B02": 70, "B03": 100, "B04": 170,
                 }
    bandid = {"B01": 0, "B02": 1, "B03": 2, "B04": 3,
                  }
    resolution = {"B01": 20, "B02": 20, "B03": 20, "B04": 20,
                  }
    name = {"B01": 'green', "B02" : 'red', "B03" : 'near infrared',
            "B04" : 'shortwave infrared',
            }
    d = {
         "wavelength": pd.Series(wavelength),
         "bandwidth": pd.Series(bandwidth),
         "resolution": pd.Series(resolution),
         "name": pd.Series(name),
         "bandid": pd.Series(bandid)
         }
    df = pd.DataFrame(d)
    return df
