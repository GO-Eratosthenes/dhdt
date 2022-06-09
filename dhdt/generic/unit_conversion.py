import numpy as np

# time conversions
def datetime2doy(dt):
    """ Convert array of datetime64 to a day of year.

    Parameters
    ----------
    dt : np.datetime64
        times of interest

    Returns
    -------
    year : integer
        calender year
    doy : integer
        day of year
    """
    year = dt.astype('datetime64[Y]').astype(int)+1970
    doy = (dt.astype('datetime64[D]') -
          dt.astype('datetime64[Y]') + 1).astype('float64')
    return year, doy

def datetime2calender(dt):
    """ Convert array of datetime64 to a calendar year, month, day.

    Parameters
    ----------
    dt : np.datetime64
        times of interest

    Returns
    -------
    cal : numpy array
        calendar array with last axis representing year, month, day
    """

    # decompose calendar floors
    Y, M, D = [dt.astype(f"M8[{x}]") for x in "YMD"]
    year = (Y + 1970).astype('timedelta64[Y]').astype(int)
    month = ((M - Y) + 1).astype('timedelta64[M]').astype(int)
    day = ((D - M) + 1).astype('timedelta64[D]').astype(int)
    return year, month, day

# atmospheric scales and formats
def kelvin2celsius(T):
    T -= 273.15
    return T

def celsius2kelvin(T):
    T += 273.15
    return T

def hpa2pascal(P):
    mbar2pascal(P)
    return P

def mbar2pascal(P):
    P *= 100
    return P

def pascal2mbar(P):
    P /= 100
    return P

def hpa2pascal(P):
    mbar2pascal(P)
    return P

def pascal2hpa(P):
    pascal2mbar(P)
    return P

def mbar2torr(P):
    P *= 0.750061683
    return P

def torr2mbar(P):
    P /= 0.750061683
    return P

# geometric and angular scales and formats
def deg2dms(ang):
    """ convert decimal degrees to degree minutes seconds format

    Parameters
    ----------
    ang : {float,np.array}, unit=decimal degrees
        angle(s) of interest

    Returns
    -------
    deg : {integer,np.array}
        degrees
    min : {integer,np.array}, range=0...60
        angular minutes
    sec : {float,np.array}, range=0...60
        angular seconds
    """
    min,sec = np.divmod(ang*3600, 60)
    deg,min = np.divmod(min, 60)
    deg,min = deg.astype(int), min.astype(int)
    return deg,min,sec

def dms2deg(deg,min,sec):
    """ convert degree minutes seconds format to decimal degrees

    Parameters
    ----------
    deg : {integer,np.array}
        degrees
    min : {integer,np.array}, range=0...60
        angular minutes
    sec : {float,np.array}, range=0...60
        angular seconds

    Returns
    -------
    ang : {float,np.array}, unit=decimal degrees
        angle(s) of interest
    """
    ang = deg.astype(float) + (min.astype(float)/60) + (sec.astype(float)/3600)
    return ang

def deg2gon(ang):
    """ convert from gon to degrees

    Parameters
    ----------
    ang : unit=degrees, range=0...360

    Returns
    -------
    ang : unit=gon, range=0...400

    See Also
    --------
    gon2deg, deg2compass
    """
    ang *= 400/360
    return ang

def gon2deg(ang):
    """ convert from gon to degrees

    Parameters
    ----------
    ang : unit=gon, range=0...400

    Returns
    -------
    ang : unit=degrees, range=0...360

    See Also
    --------
    deg2gon, deg2compass
    """
    ang *= 360/400
    return ang

def deg2compass(theta):
    """ adjust angle to be in bounds of a positive argument angle,like a compass

    Parameters
    ----------
    theta : unit=degrees

    Returns
    -------
    theta : unit=degrees, range=0...+360

    See Also
    --------
    deg2arg

    Notes
    -----
    The angle is declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x
    """
    return theta % 360

def deg2arg(theta):
    """ adjust angle to be in bounds of an argument angle

    Parameters
    ----------
    theta : unit=degrees

    Returns
    -------
    theta : unit=degrees, range=-180...+180

    See Also
    --------
    deg2compass

    Notes
    -----
    The angle is declared in the following coordinate frame:

        .. code-block:: text

                 ^ North & y
                 |
            - <--|--> +
                 |
                 +----> East & x
    """
    return ((theta + 180) % 360) -180