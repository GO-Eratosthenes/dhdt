import math

import numpy as np

def sigma_filtering(y,thres=3):
    """
    3-sigma filtering

    """
    mean_y = np.mean(y)
    std_y = np.std(y)
    
    IN = (y > (mean_y - thres*std_y)) & \
        (y < mean_y + thres*std_y)
    return IN

def weighted_sigma_filtering(y,w,thres=3):
    """
    weighted 3-sigma filtering

    """
    mean_y = np.average(y,weights=w)
    std_y = math.sqrt(np.average((y-mean_y)**2, weights=w))

    IN = (y > (mean_y - thres*std_y)) & \
        (y < mean_y + thres*std_y)
    return IN

def mad_filtering(y,thres=3):
    """
    robust 3-sigma filtering

    """
    med_y = np.median(y)
    mad_y = np.median(np.abs(med_y - med_y))

    IN = (y > (med_y - thres*mad_y)) & \
        (y < med_y + thres*mad_y)
    return IN

def normalize_variance(y):
    mean_y = np.mean(y)
    std_y = np.std(y)
    r = (y-mean_y)/std_y
    return r

def normalize_variance_robust(y):
    med_y = np.median(y)
    mad_y = np.median(np.abs(med_y - med_y))
    r = (y-med_y)/mad_y
    return r

def huber_filtering(y,thres=1.345,preproc='normal'):    
    if preproc == 'normal':
        r = normalize_variance(y)
    else:
        r = normalize_variance_robust(y)
    w = 1 / max(1, abs(r))
    IN = w<thres
    return IN

def cauchy_filtering(y,thres=2.385,preproc='normal'):
    if preproc == 'normal':
        r = normalize_variance(y)
    else:
        r = normalize_variance_robust(y)    
    w = 1 / (1 + r**2)
    IN = w<thres
    return IN
    

