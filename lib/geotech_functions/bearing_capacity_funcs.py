# Standard imports
import numpy as np

def calcDynamicBearingCapacity(acceleration, massFFP, contactArea, volumeFFP, waterDrop, gravity = 9.80665, rho_w = 1020 ):
    # Function calculates the dynamic bearing capacity of the soil

    # Inputs
        # dropType (S: "air" or "water" drop
        # accleration: deaccleration array
        # massFFP: Mass of the Free Fall Penetrometer (FFP)
        # contactArea: Array of contact areas between the FFP and the soil
        # rho_w: Density of water (If no val provided assumed to be 1020 [kg/m^3] which is a good estimate for ocean water)
        # volumeFFP: Volume of the FFP

    if waterDrop:
        # TODO: This equation is super wrong
        # FIXME: There should be a gravity force, a buoyant force, and the bearing capacity force at a minimum

        # # Calc mass of water displaced by the FFP
        # displacedWaterMass = rho_w * volumeFFP

        # # Calc Buoyant mass of the FFP
        # buoyantMass = massFFP - displacedWaterMass

        # # Calc impact force
        # force = buoyantMass * acceleration

        # Calc the force of gravity
        force_gravity = massFFP * gravity

        # Calc the buyant force assuming water pressure in soil doesn't change # TODO: This is a pretty big assumption
        force_buoyant = rho_w * gravity * volumeFFP

        # Calc the bearing capacity force
        force_bearing = massFFP * acceleration - force_buoyant + force_gravity

    # Check for air drop
    elif not waterDrop:
        force_bearing = massFFP * acceleration + force_gravity

    # Dynamic bearing capacity
    qDyn = force_bearing/contactArea

    return qDyn

def calcQuasiStaticBearingCapacity(velocity, strainrateCorrectionType, qDyn, K_Factor = 0.1, refVelocity = 0.02, BrilliCoeff = 0.31):
    # Function calculates the quasi-static bearing capacity value
    # velocity: Array of FFP velocity values [m/s]
    # K_Factors: Array of K factors to be used

    # Local Variables: 
    # f_SR: Strain Rate factor
    # maxVelocity: Max velocity of the descent

    maxVelocity = np.max(velocity)

    if strainrateCorrectionType == "log":
        f_SR = 1 + K_Factor * np.log10(velocity/refVelocity)

    elif strainrateCorrectionType == "Brilli":
        # Strain rate correction factor following Brilli et al. (20??) Commonly used for air drops
        f_SR = 1 + BrilliCoeff * maxVelocity * np.log10(velocity/refVelocity)

    elif strainrateCorrectionType == "invHyperSin":
        # Inv hyperbolic sin correction factor following Stephan (2015) and Randolph (2004)
        f_SR = 1 + K_Factor/np.log(10) * np.arcsinh(velocity/refVelocity)

    else:
        ValueError("strainrateCorrectionType must be one of the folliwng: log, Brilli, invHyperSin.")

    # calc Quasi Static Bearing capacity value
    qsbc = qDyn/f_SR
    return qsbc