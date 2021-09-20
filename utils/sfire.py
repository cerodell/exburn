import context
import numpy as np

import config

# find zi
def get_zi(T0, dz):
    r"""
    Retrieve the height of boundary layer top $z_i$, based on gradient of potential temperature lapse rate.
    ...
    Parameters:
    -----------
    T0 : ndarray
        1D array representing potential temperature sounding [K]
    dz : float
        vertical grid spacing [m]
    Returns:
    --------
    $z_i$ : float
        boundary layer height [m]
    """
    dT = T0[1:] - T0[0:-1]
    gradT = dT[1:] - dT[0:-1]
    surface = round(200 / dz)
    zi_idx = np.argmax(gradT[surface:]) + surface  # vertical level index of BL top
    zi = dz * zi_idx
    return zi
