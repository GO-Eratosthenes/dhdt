import pathlib

import numpy as np

from scipy.interpolate import griddata  # for grid interpolation
from xml.etree import ElementTree

from ..generic.handler_s2 import get_array_from_xml


def read_mean_sun_angles_s2(path):  # processing
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts the
    mean sun angles.
    """
    fname = pathlib.Path(path)/'MTD_TL.xml'
    dom = ElementTree.parse(fname)
    root = dom.getroot()

    Zn = float(root[1][1][1][0].text)
    Az = float(root[1][1][1][1].text)
    return Zn, Az


def read_view_angles_s2(path):  # processing
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    """
    fname = pathlib.Path(path)/'MTD_TL.xml'
    dom = ElementTree.parse(fname)
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)
    # get coarse grids
    for grd in root.iter('Viewing_Incidence_Angles_Grids'):
        bid = float(grd.get('bandId'))
        if bid == 4:  # take band 4
            Zarray = get_array_from_xml(grd[0][2])
            Aarray = get_array_from_xml(grd[1][2])
            if 'Zn' in locals():
                Zn = np.nanmean(np.stack((Zn, Zarray), axis=2), axis=2)
                Az = np.nanmean(np.stack((Az, Aarray), axis=2), axis=2)
            else:
                Zn = Zarray
                Az = Aarray
            del Aarray, Zarray

    # upscale to 10 meter resolution
    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)
    Zok = ~np.isnan(Zn)
    Zn = griddata(Zij[Zok.reshape(-1), :], Zn[Zok], (Igrd, Jgrd),
                  method="linear")

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj

    Aok = ~np.isnan(Az)  # remove NaN values from interpolation
    Az = griddata(Aij[Aok.reshape(-1), :], Az[Aok], (Igrd, Jgrd),
                  method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az
