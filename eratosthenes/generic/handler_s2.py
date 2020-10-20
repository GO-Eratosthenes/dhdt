import numpy as np


def get_array_from_xml(treeStruc):  # generic
    """
    Arrays within a xml structure are given line per line
    Output is an array
    """
    for i in range(0, len(treeStruc)):
        Trow = [float(s) for s in treeStruc[i].text.split(' ')]
        if i == 0:
            Tn = Trow
        elif i == 1:
            Trow = np.stack((Trow, Trow), 0)
            Tn = np.stack((Tn, Trow[1, :]), 0)
        else:
            # Trow = np.stack((Trow, Trow),0)
            # Tn = np.concatenate((Tn, [Trow[1,:]]),0)
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn


def meta_S2string(S2str):  # generic
    """
    get meta data of the Sentinel-2 file name
    input:   S2str          string            filename of the L1C data
    output:  S2time         string            date "YYYYMMDD"
             S2orbit        string            relative orbit "RXXX"
             S2tile         string            tile code "TXXXXX"
    """
    S2split = S2str.split('_')
    S2time = S2split[2][0:8]
    S2orbit = S2split[4]
    S2tile = S2split[5]
    return S2time, S2orbit, S2tile
