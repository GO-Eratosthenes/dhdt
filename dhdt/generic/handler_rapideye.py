def meta_REstring(REstr):  # generic
    """ get meta information of the RapidEye file name

    Parameters
    ----------
    REstr : string
        filename of the rapideye data

    Returns
    -------
    REtime : string
        date "+YYYY-MM-DD"
    REtile : string
        tile code "TXXXXX"
    REsat : string
        which RapedEye satellite

    Example
    -------
    >>> REstr = '568117_2012-09-10_RE2_3A_Analytic.tif'
    >>> REtime, REtile, REsat = meta_REstring(REstr)
    >>> REtime
    '+2012-09-10'
    >>> REtile
    '568117'
    >>> REsat
    'RE2'
    """
    assert type(REstr)==str, ("please provide a string")
    
    REsplit = REstr.split('_')
    REtime = '+' + REsplit[1][0:4] +'-'+ REsplit[1][5:7] +'-'+ REsplit[1][8:10]
    REsat = REsplit[2]
    REtile = REsplit[0]
    return REtime, REtile, REsat
