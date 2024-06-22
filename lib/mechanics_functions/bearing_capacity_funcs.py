# Standard imports
import numpy as np

# Lib imports
from lib.general_functions.global_constants import GRAVITY_CONST, DEBUG

def calc_air_drop_dyn_bearing(pffp_accel, pffp_mass, contact_area, gravity):
    """
    ## Purpose: calc the soil dynamic bearing capacity for air drops

    Units are assumed to be standard SI ie. m/s^2, Newtons, kg
    Reference frame convention: The towards the sky direction is assumed to be positive. 
    NOTE: DO NOT pass raw accelerometer data to this function, the acceleration passed to this function should 
    be the sensor offset by 1g down
    ie. pffp_accel= raw_sensor_accel - 1g

    Eqn 1: $F_{Br} = m_{p} a_{p} + m_{p} g$
    Eqn 2: $q_{Dyn} = F_{Br}/A$

    where:
        F_{Br} : Force of soil bearing resistance
        m_{p}  : Mass of the portabble free fall penetrometer (pffp)
        a_{p}  : Acceleration of the pffp during impact
        g      : Gravity Constant
    """

    # Calc the force of gravity
    force_gravity = pffp_mass * gravity
    
    # Calc the bearing force
    force_bearing = pffp_mass * pffp_accel + force_gravity

    # Calc the dyn
    qDyn = force_bearing/contact_area

    return  qDyn

def calc_water_drop_dyn_bearing(pffp_accel, pffp_mass, contact_area, pffp_volume, gravity, rho_w):
    """
    ## Purpose: calc the soil dynamic bearing capacity for water drops

    Units are assumed to be standard SI ie. m/s^2, Newtons, kg
    Reference frame convention: The towards the sky direction is assumed to be positive. 
    NOTE: DO NOT pass raw accelerometer data to this function, the acceleration passed to this function should 
    be the sensor offset by 1g down
    ie. pffp_accel= raw_sensor_accel - 1g

    Eqn 1: $F_{Br} = m_{p} a_{p} + m_{p} g - rho_{f} V_{p} g$
    Eqn 2: $q_{Dyn} = F_{Br}/A$

    where:
        F_{Br} : Force of soil bearing resistance
        m_{p}  : Mass of the portabble free fall penetrometer (pffp)
        a_{p}  : Acceleration of the pffp during impact
        g      : Gravity Constant
        rho_{f}: Density of fluid
        V_{p}  : Volume of the pffp 
    """
     
    # Calc the force of gravity
    force_gravity = pffp_mass * gravity 

    # Calc the buyant force assuming water pressure in soil doesn't change # NOTE: This is a pretty big assumption
    force_buoyant = rho_w * gravity * pffp_volume
    
    # Calc the bearing capacity force 
    force_bearing = pffp_mass * pffp_accel + force_gravity - force_buoyant 

    # Need to convert force_bearing to array to over the stored indices in it's series
    # Calc the dynamic bearing capacity
    qDyn = np.array(force_bearing)/contact_area

    return qDyn

def calc_dyn_bearing_capacity(pffp_accel, pffp_mass, contact_area, pffp_volume, water_drop, gravity = GRAVITY_CONST, rho_w = 1020):
    """
    Purpose: Acts as a wrapper for the water and air drop soil bearing capacity calculations
    NOTE: USES PFFP acceleration NOT raw sensor acceleration

    Inputs
        water_drop  : True for water drops, False for air drops
        accleration: deaccleration of the PFFP
        pffp_mass: Mass of the Free Fall Penetrometer (FFP)
        contact_area: Array of contact areas between the FFP and the soil
        rho_w: Density of water (If no val provided assumed to be 1020 [kg/m^3] which is a good estimate for ocean water)
        pffp_volume: Volume of the FFP
    """
    
    if water_drop:
        # Calc the dynamic bearing capacity
        qDyn = calc_water_drop_dyn_bearing(pffp_accel, pffp_mass= pffp_mass, contact_area=contact_area, 
                                           pffp_volume = pffp_volume, gravity=gravity, rho_w=rho_w)

    elif not water_drop:
        # Otherwise do the calc using the air drop formula
        qDyn = calc_air_drop_dyn_bearing(pffp_accel=pffp_accel, pffp_mass=pffp_mass, contact_area=contact_area, gravity=gravity)

    return qDyn

def calc_qs_bearing_capacity(velocity, strainrateCorrectionType, qDyn, k_factor = 0.1, ref_velocity = 0.02):
    """
    Purpose: calculate the quasi-static bearing capacity value
    velocity: Array of FFP velocity values [m/s]
    k_factor: k factor to be used

    Local Variables: 
    f_SR: Strain Rate factor
    maxVelocity: Max velocity of the descent
    """
    maxVelocity = np.max(velocity)
    
    # make sure that velocity isn't a series
    velocity = np.array(velocity)

    # Selct with strain rate factor should be used
    match strainrateCorrectionType:
        case "log":

            # Force the log val to not become undefined
            log_val = np.log10(velocity/ref_velocity)
            log_val = np.nan_to_num(log_val, nan=0.0)
            
            f_SR = 1 + k_factor * log_val

        case "Brilli":
            # Strain rate correction factor following Brilli et al. (20??) Commonly used for air drops
            
            # The value used by Brilli was 0.31
            if k_factor != 0.31:
                # Print a warning if a non-standard value is used for the brilli type calculation
                print("Warning: Value used for Brilli type strain rate correction is not 0.31")
            
            log_val = np.log10(velocity/ref_velocity)
            f_SR = 1 + k_factor * maxVelocity * log_val

        case "invHyperSin":
            # Inv hyperbolic sin correction factor following Stephan (2015) and Randolph (2004)
            f_SR = 1 + k_factor/np.log(10) * np.arcsinh(velocity/ref_velocity)

        case _:
            ValueError("strainrateCorrectionType must be one of the folliwng: log, Brilli, invHyperSin.")

    # calc Quasi Static Bearing capacity value
    qsbc = qDyn/f_SR
    return qsbc

if __name__ == "__main__":
    # Add some testing here
    pass