# Standard imports 
import numpy as np

def  calc_Albatal_rel_density(max_deceleration):
    # Purpose: Calc the relative density following the correlation presented by Albatal (2019)
    # Link to paper: https://cdnsciencepub.com/doi/pdf/10.1139/cgj-2018-0267 (Eqn at bottom right of page 26)

    # Variables
    # max_deceleration: The maximum deceleration from the BlueDrop recording

    return -2.18 * 1e-4 * max_deceleration**3  + 1.29 * 1e-2 * max_deceleration**2 + 1.61 * max_deceleration - 13.09

if __name__ == "__main__":
    # Add some testing here
    pass