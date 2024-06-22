# Standard imports 
import numpy as np

# Libary imports
from lib.mechanics_functions.general_geotech_funcs import calc_cambridge_mean_eff_stress
from lib.general_functions.global_constants import DEBUG

def  calc_Albatal_rel_density(max_deceleration):
    """
    Purpose: Calc the relative density following the correlation presented by Albatal (2019)
    Link to paper: https://cdnsciencepub.com/doi/pdf/10.1139/cgj-2018-0267 (Eqn at bottom right of page 26)

    where:
        max_deceleration: The maximum deceleration from the BlueDrop recording in g's
    """
    return -2.18 * 1e-4 * max_deceleration**3  + 1.29 * 1e-2 * max_deceleration**2 + 1.61 * max_deceleration - 13.09

# def calc_Jamiolkowski_relative_density(qNet_dry, depth, soil_unit_wt = 17.81, water_unit_wt = 9.81, C0 = 300, C1 = 0.46, C2 = 2.96, k0 = 0.5):
def calc_Jamiolkowski_relative_density(qNet_dry, depth, soil_unit_wt = 17.81, water_unit_wt = 9.81, C0 = 300, C1 = 0.46, C2 = 2.96, k0 = 0.5):

    """"
    Calc the relative density (I_d) using the equation from Jamiolkowski et al. (2003)
    
    Input:
        qNet_dry       : Net Drained bearing resistance
        depth        : Depth below the seabed
        soil_unit_wt : Total Soil unit weight [kN/m^3]
        water_unit_wt: Water unit weight (rho * g) [kN/m^3]

    Eqn:
        I_{d} = 1/C_{2} * ln(\frac{ q_{net, d} }{ C_{0} * p'_{0}^{C_{1}} })

    where:
        C_{0}: Dimensionless Coefficient
        C_{1}: Coefficient
        C_{2}: Coefficient
        q_{net, d}: Net Drained bearing resistance
    """

    # Calc the vertical effective stress
    sigma_1 = (soil_unit_wt - water_unit_wt) * depth

    # Calc the horizontal effective stress
    sigma_3  = k0 * sigma_1 

    # Calc the mean effective stress
    mean_eff_stress = calc_cambridge_mean_eff_stress(sigma_1, sigma_3, sigma_3)

    # Calc the insid of the parenthesis
    inside = qNet_dry/(C0 * mean_eff_stress**C1)

    if DEBUG:
        print("Inside the parenthesis: {}".format(inside))
        print("sigma_1: {:.2f}".format(sigma_1))
        print("Mean effective stress: {:.2f}".format(mean_eff_stress))

    # Calc the relative density
    return 1/C2 * np.log(inside)


if __name__ == "__main__":
    # Add some testing here
    pass