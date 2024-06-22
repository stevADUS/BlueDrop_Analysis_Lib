# Standard imports
import numpy as np

# Functions that characterize the soil are stored here

# Firmness factor
def calc_firmness_factor(gravity, max_deceleration, total_penetration_time, impact_velocity):
    """
    Purpose: Calc the firmness factor (FF) of the soil
    Eqn.:
        FF = a_{max}/(  g * t_{p} * v_{i}) 
    
    Link to paper: https://link.springer.com/article/10.1007/s11001-011-9116-2

    NOTE: Not a dimensionless parameter
    """

    return max_deceleration/(gravity * total_penetration_time * impact_velocity)                                           

if __name__ == "__main__":
    # Add some testing here
    pass