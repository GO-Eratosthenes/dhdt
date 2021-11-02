import numpy as np

def project_along_flow(dX_raw,dY_raw,dX_prio,dY_prio,e_perp):
    """

    Parameters
    ----------
    dX_raw : np.array, size=(m,n), dtype=float
        raw horizontal displacement with mixed signal
    dY_raw : np.array, size=(m,n), dtype=float
        raw vertical displacement with mixed signal
    dX_prio : np.array, size=(m,n), dtype=float
        reference of horizontal displacement (a-priori knowledge)
    dY_prio : np.array, size=(m,n), dtype=float
        reference of vertical displacement (a-priori knowledge)
    e_perp : np.array, size=(2,1), float
        vector in the perpendicular direction to the flightline (bearing).

    Returns
    -------
    dX_proj : np.array, size=(m,n), dtype=float
        projected horizontal displacement in the same direction as reference.
    dY_proj : np.array, size=(m,n), dtype=float
        projected vertical displacement in the same direction as reference.
    
    Notes
    ----- 
    The projection function is as follows:

    .. math:: P = ({d_{x}}e^{\perp}_{x} - {d_{y}}e^{\perp}_{y}) / ({\hat{d}_{x}}e^{\perp}_{x} - {\hat{d}_{y}}e^{\perp}_{y})

    See also Equation 10 and Figure 2 in [1].

    Furthermore, two different coordinate system are used here: 
        
        .. code-block:: text

          indexing   |           indexing    ^ y
          system 'ij'|           system 'xy' |
                     |                       |
                     |       i               |       x 
             --------+-------->      --------+-------->
                     |                       |
                     |                       |
          image      | j         map         |
          based      v           based       |


    References
    ----------
    .. [1] Altena & Kääb. "Elevation change and improved velocity retrieval 
       using orthorectified optical satellite data from different orbits" 
       Remote Sensing vol.9(3) pp.300 2017.        
    """
    # e_{\para} = bearing satellite...
    assert(dX_raw.size == dY_raw.size) # all should be of the same size
    assert(dX_prio.size == dY_prio.size)
    assert(dX_raw.size == dX_prio.size)
    
    d_proj = ((dX_raw*e_perp[0])-(dY_raw*e_perp[1])) /\
        ((dX_prio*e_perp[0])-(dY_prio*e_perp[1]))

    dX_proj = d_proj * dX_raw
    dY_proj = d_proj * dY_raw 
    return dX_proj,dY_proj 
 
# geometric_configuration
# temporal_configuration  