import numpy as np

def get_hypsometry(Z,dZ,interval=100): # todo

    L = np.round(Z/interval)
    f =  lambda x: np.quantile(x, 0.5)
    label,hyspometry = get_stats_from_labelled_array(L,dz,f)
    return label, hypsometry

def get_stats_from_labelled_array(L,I,func): #todo is tuple: multiple functions?
    """ get properties of a labelled array

    Parameters
    ----------
    L : np.array, size=(m,n), {bool,integer,float}
        array with unique labels
    I : np.array, size=(m,n)
        data array of interest
    func : function
        group operation, that extracts information about the labeled group

    Returns
    -------
    labels : np.array, size=(k,1), {bool,integer,float}
        array with the given labels
    result : np.array, size=(k,1)
        array with the group property, as given by func

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.filters import threshold_otsu

    >>> image = data.coins()
    >>> thresh = threshold_otsu(image)
    >>> L = image > thresh
    >>> func = lambda x: np.quantile(x, 0.7)
    >>> label,quant = get_stats_from_labelled_array(L,image,func)
    False: 74
    True: 172
    """
    L, I = L.flatten(), I.flatten()
    labels, indices, counts = np.unique(L, return_counts=True, return_inverse=True)
    arr_split = np.split(I[indices.argsort()], counts.cumsum()[:-1])

    result = np.array(list(map(func, arr_split)))
    return labels, result
