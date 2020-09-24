import pathlib

import numpy as np

from scipy.interpolate import griddata
from xml.etree import ElementTree

from eratosthenes.generic.handler_s2 import get_array_from_xml


def read_sun_angles_s2(path):  # pre-processing
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    input:   path           string            path where xml-file of
                                              Sentinel-2 is situated
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimtuh angles

    """
    path = pathlib.Path(path)
    fname = path/'MTD_TL.xml'
    dom = ElementTree.parse(fname)
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)), 0)
    znSpac = np.divide(znSpac, 10)  # transform from meters to pixels

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj, znSpac

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)), 0)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj, azSpac

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az
