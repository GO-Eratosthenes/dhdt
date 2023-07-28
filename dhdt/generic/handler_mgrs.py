import re

import numpy as np

from osgeo import osr

from dhdt.generic.unit_check import check_mgrs_code
from dhdt.generic.gis_tools import get_utm_zone
from dhdt.generic.mapping_tools import ll2map


def get_mgrs_tile(ϕ, λ):
    """ return a military grid reference system zone designation string.
    This zoning is used in the tiling of Sentinel-2.

    Parameters
    ----------
    ϕ : float, unit=degrees, range=-90...+90
        latitude
    λ : float, unit=degrees
        longitude

    Returns
    -------
    mgrs_code : string
    """

    # numbering goes with the alphabet, excluding "O" and "I"
    mgrsABC = [
        chr(i) for i in list(range(65, 73)) + list(range(74, 79)) +
        list(range(80, 91))
    ]
    tile_size = 100.  # [km]
    tile_size *= 1E3

    utm_zone = get_utm_zone(ϕ, λ)
    utm_no = int(utm_zone[:-1])

    # transform to UTM coordinates
    proj = osr.SpatialReference()
    NH = True if ϕ > 0 else False
    proj.SetUTM(utm_no, NH)
    falseEasting = proj.GetProjParm(osr.SRS_PP_FALSE_EASTING)

    xyz = np.squeeze(ll2map(np.array([[ϕ, λ]]), proj))

    # λ letter
    shift_letter = np.floor(np.divide(xyz[0] - falseEasting,
                                      tile_size)).astype(int)
    center_letter = int(np.mod(utm_no - 1, 3) * 8) + 4
    λ_letter = mgrsABC[center_letter + shift_letter]

    # ϕ letter
    tile_shift = np.fix(xyz[1])
    ϕ_num = np.mod(tile_shift, 20)
    if np.mod(utm_no, 2) == 0:  # even get an additional five letter shift
        if np.all((xyz[1] < 0, np.abs(tile_shift) > 5)):
            ϕ_num -= 4  # counts up to "R"
        else:
            ϕ_num += 5
    else:
        if xyz[1] < 0:  # a leap hole is present in the southern hemisphere
            ϕ_num -= 4  # counts up to "R"
    ϕ_idx = np.mod(ϕ_num, 20)
    ϕ_idx -= 1
    ϕ_letter = mgrsABC[ϕ_idx]

    mgrs_code = utm_zone + λ_letter + ϕ_letter
    return mgrs_code
