# Standard imports
import numpy as np

# Functions that characterize the soil are stored here

# Firmness factor
def calc_firmness_factor(gravity, max_deceleration, total_penetration_time, impact_velocity):
    """
    Calculate the firmness factor (FF) of the soil.

    Parameters
    ----------

    gravity : float
        Acceleration due to gravity [m/s^2].
    max_deceleration : float
        Maximum deceleration of the probe [m/s^2].
    total_penetration_time : float
        Total time of penetration [s].
    impact_velocity : float
        Impact velocity of the probe [m/s].

    Returns
    -------

    float
        Firmness factor (FF).

    Notes
    -----

    The firmness factor is calculated using the following equation:
    
    .. math::
        FF = \\frac{a_{max}}{g \\cdot t_{p} \\cdot v_{i}}
    
    where:
        - :math:`FF`      : Firmness factor.
        - :math:`a_{max}` : Maximum deceleration.
        - :math:`g`       : Acceleration due to gravity.
        - :math:`t_{p}`   : Total penetration time.
        - :math:`v_{i}`   : Impact velocity.

    The firmness factor is not a dimensionless parameter.

    For more information, refer to the paper: [Springer article](https://link.springer.com/article/10.1007/s11001-011-9116-2).
    
    """

    return max_deceleration/(gravity * total_penetration_time * impact_velocity)                                           

if __name__ == "__main__":
    # Add some testing here
    pass