import os

from pathlib import Path
from skimage.filters import threshold_otsu

from dhdt.generic.mapping_io import read_geo_image
from dhdt.preprocessing.image_transforms import mat_to_gray
from dhdt.preprocessing.shadow_transforms import (apply_shadow_transform,
                                                  _get_shadow_transforms)
from dhdt.testing.classification_tools import calculate_accuracy

TEST_DATA = os.path.join(Path(__file__).parents[2],
                         'testdata/ShadowImages/shadow-5cvx')


def _get_shadow_test_imagery():
    Img = read_geo_image(TEST_DATA + '.jpg')[0]
    Img = mat_to_gray(Img)
    Msk = read_geo_image(TEST_DATA + '.png')[0]
    return Img, Msk


# testing functions, based on imagery from:
# https://www3.cs.stonybrook.edu/~minhhoai/projects/shadow.html

def test_shadow_transforms():
    methods = list(_get_shadow_transforms().keys())
    methods = list(methods[i] for i in [0, 5, 6, 7, 8, 10, 11, 12])

    Img, Msk = _get_shadow_test_imagery()
    for method in methods:
        Scn = apply_shadow_transform(method,
                                     Img[..., 2], Img[..., 1], Img[..., 0],
                                     None, None, None)
        if method in ['sdi', 'c3']:
            Class = Scn < threshold_otsu(Scn)
        else:
            Class = Scn > threshold_otsu(Scn)

        # estimate classification metrics
        acc = calculate_accuracy(Msk, Class != 0)
        assert acc > .7, 'accuracy of classification seems to low'
    return
