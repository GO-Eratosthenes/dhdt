import numpy as np

# raster/image libraries
from skimage.measure import ransac
from skimage.transform import AffineTransform

def findCoherentPairs(src,dst):
    '''
    
    '''
    # estimate affine transform model using all coordinates
    model = AffineTransform()
    model.estimate(src, dst)
    
    # estimate affine transformation model with RANSAC
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
    outliers = inliers == False
    return inliers
