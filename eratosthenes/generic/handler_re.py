def meta_REstring(REstr):  # generic
    """
    get meta data of the RapidEye file name
    input:   REstr          string            filename of the rapideye data
    output:  REtime         string            date "+YYYY-MM-DD"
             REsat          string            which RapedEye satellite
             REtile         string            tile code "TXXXXX"
    """
    REsplit = REstr.split('_')
    REtime = '+' + REsplit[1][0:4] + REsplit[1][5:7] + REsplit[1][8:10]
    REsat = REsplit[2]
    REtile = REsplit[0]
    return REtime, REtile, REsat, 