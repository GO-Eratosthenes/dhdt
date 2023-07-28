import glob
import os
from xml.etree import ElementTree

import numpy as np


def get_array_from_xml(treeStruc):
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
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn


def get_root_of_table(path, fname=None):
    if fname is None:
        full_name = path
    else:
        full_name = os.path.join(path, fname)
    if not '*' in full_name:
        assert os.path.isfile(full_name), \
            ('please provide correct path and file name')
    dom = ElementTree.parse(glob.glob(full_name)[0])
    root = dom.getroot()
    return root
