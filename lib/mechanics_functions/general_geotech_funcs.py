# Standard imports
import numpy as np

# Lib imports
from lib.general_functions.global_constants import GRAVITY_CONST

def calc_consolidation_coeff(diameter, t_50, T_50 = 0.6):
    """"
    Calc the consolidation coeff (c_{h})
    
    Paper: White, D. J., et al. "Free fall penetrometer tests in sand: Determining the equivalent static resistance.
    Eqn:
        c_{h} = \frac{D^{2} T_{50}}{t_{50}}
    
    where:
        D: Diameter of the penetrating object
        t_50: Time to 50% of pore pressure dissipation
        T_50: Dimensionless time
    """

    return diameter**2 * T_50/t_50

def calc_dimensionless_velocity(v, D, coeff_consolidation):
    """
    Calc the dimensionless velocity (V)
    
    Eqn:
        V = v D/c_{h}
        
    where:
        V    : Dimensionless velocity
        v    : Probe velocity
        D    : Diameter of the probe
        c_{h}: Consolidation coefficient
    """

    return v * D/coeff_consolidation

def calc_cambridge_mean_eff_stress(sigma_1, sigma_2, sigma_3):

    """
    Calc cambridge mean effective stress

    Eqn:
        p' = (\sigma'_{1} + \sigma'_{2} + \sigma'_{3})/3

    where:
        sigma_{1}: Effective stress 1
        sigma_{2}: Effective stress 2
        sigma_{3}: Effective stress 3
    """

    return (sigma_1 +sigma_2 + sigma_3)/3

def calc_white_failure_mean_eff_stress(relative_density, Q = 10):
    """
    Calc the mean effective stress at failure.
    
    Eqn:
        p'_{f} = e^{ Q - 1/I_{D} }
    
    where:
        p'_{f}: Mean effective stress at failure [kPa]
        Q     : Crushing strength parameter, Commonly taken as 10
        I_{D} : relative density

    
    NOTE: This equation was taken from 
          I_{R} = I_{D} (D - ln(p')) - 1 (Bolton (1986))
          assuming relative dilatancy is zero 0 (I_{R}= 0) at undrained failure
    """

    return np.exp( Q - 1/relative_density)

def calc_mohr_coulomb_su(failure_mean_eff_stress, phi_cv = 32):
    """
    Calc the undrained strength (su) assuming a mohr-coulomb failure envelope
    
    inputs:
        failure_mean_eff_stress: Mean effective stress at failure
        phi_cv                 : Friction angle at constant volume

    Eqn:
        s_{u} = 1/2 p'_{f} \frac{ 6 sin(\phi_{cv}) }{3 - sin(\phi_{cv}) }
    
    where:
        s_{u}   : Undrained Strength
        p'_{f}  : Mean effective stress at failure 
        phi_{cv}: Friction angle at constant volume
    """

    # Convert phi to radians
    phi_cv = phi_cv * np.pi/180

    # Calc the inside of the parenthesis
    inside = 6 * np.sin(phi_cv)/( 3 - np.sin(phi_cv))

    return 0.5 * failure_mean_eff_stress * inside



if __name__ == "__main__":
    # Add some testing here
    pass