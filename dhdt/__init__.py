# -*- coding: utf-8 -*-

import logging
import numpy as np

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Bas Altena"
__email__ = 'b.altena@uu.nl'

np.set_printoptions(precision=3, suppress=True)
