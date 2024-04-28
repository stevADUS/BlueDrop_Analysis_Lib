# Standard imports
import numpy as np

# Functions that are related to calculating friction angle

def calc_Duncan_friction_angle(relative_density, unit_weight, max_depth, coeff = [34, 10, 3, 2], atmospheric_pressure = 101.325):
    # Purpose: Calc the friction angle (phi) using the Duncan correlation.
    # Link to paper: #TODO: Add link to paper here
    # Eqn: #TODO: type the equation here

    # NOTE: If alternate values of the eqn. coefficients pass them into this function
     
    # Variables
    # relative_density: Relative density of the soil
    # unit_weight: Unit weight of the soil (TODO: Figure out which unit weight it should be) Elise used 1120 kg/m^3 for the assumed value
    # max_depth: Max penetration depth of the FFP
    # coeff: [A, B, C, D] coefficients from the Duncan correlation
    # atmospheric_pressure: Atmospheric pressure, if no value is provided assumed to be 101.325 kPa

    
    # Coefficients A, B, C, and D from the Duncan correlation
    A = coeff[0]
    B = coeff[1]
    C = coeff[2]
    D = coeff[3]

    sigma = unit_weight  * max_depth

    phi = A * B  * relative_density/100 - (C + D * relative_density/100) *  np.log(sigma/atmospheric_pressure)

    return phi


def calc_Alabatal_friction_angle(relative_density, unit_weight, max_depth, atmospheric_pressure = 101.325):
    # Purpose: Calc the friction angle (phi) using the modified Duncan correlation coefficients Albatal used
    # This function acts as a wrapper for the Duncan friction angle eqn.
    # Link to paper: #TODO: Add link to paper here
    # Eqn: #TODO: type the equation here

    # List of coefficients that Albatal used for his fit
    Alabatal_coeff = [34, 10, 2, 5]

    # Calc phi (friction angle)
    phi = calc_Duncan_friction_angle(relative_density, unit_weight, max_depth, Alabatal_coeff, atmospheric_pressure)

    return phi

if __name__ == "__main__":
    # Add some testing here
    pass