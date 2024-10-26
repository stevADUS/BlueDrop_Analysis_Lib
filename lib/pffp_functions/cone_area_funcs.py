#Standard imports 
import numpy as np
import pandas as pd

# From local library
from lib.pffp_functions.basic_geometry_funcs import calcCircleArea, calcConeLateralSurfaceArea, calcCylinderSurfaceArea, calcParabolaSurfaceArea
from lib.general_functions.global_constants import ALLOWED_TIP_TYPES_LIST


def calc_pffp_contact_area(penetrationDepth, areaCalcType, tipType, tipProps, tip_val_col):
    """
    Purpose: Wrapper for the contact area functions for all cones
    NOTE: Assumes a consistent set of units
    where:
        penetrationDepth: Array of depths the FFP has penetrated, 
        areaCalcType: "mantle or projected"
        tipType: blunt or cone tip type       
        tipProps: Holds the tip properties that are required for the area calculation
    """

    if not isinstance(tipProps, pd.DataFrame):
        raise TypeError("Tip props should be a dataframe of the tip properties") 
    
    # Unpack the general tip properties
    tipHeight = tipProps.loc[tipProps["Properties"] == "tip_height"][tip_val_col].iloc[0]
    baseRadius = tipProps.loc[tipProps["Properties"] == "base_radius"][tip_val_col].iloc[0] 

    # Calc area for cone tip
    match tipType:
        case "cone":
            # Unpack the cone properties
            coneTipRadius = tipProps.loc[tipProps["Properties"] == "tip_radius"][tip_val_col].iloc[0] 

            area = calcFFPConeContactArea(penetrationDepth, tipHeight, baseRadius, coneTipRadius,  areaCalcType)

        # Calc area for blunt tip
        case "blunt":
            area = calcFFPBluntContactArea(penetrationDepth, tipHeight, baseRadius, areaCalcType)
        
        case "parabola":
            # Unpack the radius coefficient
            radius_coeff = tipProps.loc[tipProps["Properties"] == "radius_coeff"][tip_val_col].iloc[0]

            area = calcParabolicContactArea(penetrationDepth, tipHeight, areaCalcType, radius_coeff)
        case _: 
            raise ValueError("{} is not a valid contact area calculation type\n \
                             Valid options are {}".format(tipType, ALLOWED_TIP_TYPES_LIST))
    return area 

#TODO: Could use a mask to calc the values for the whole array that is below the tipHeight and init array for the part that is above

def calcFFPConeContactArea(penetrationDepth, tipHeight, baseRadius, coneTipRadius, areaCalcType):
    """ 
    Calculates the contact area for the cone top of the FFP.

    Parameters
    ----------
    penetrationDepth: np.ndarray
        Array of penetration depth values.
    baseRadius: float
        Radius of the Cone at the base (Circular side).
    coneTipRadius: float
        Radius of the cone tip (is not zero).
    tipHeight: float
        Height of the cone when measured normal to the circular base.
    areaCalcType: str
        Selection of "mantle" or "projected" for area calculation type.

    Returns
    -------
    np.ndarray
        Array of calculated contact area values for each penetration depth.

    Raises
    ------
    ValueError
        If `areaCalcType` is not "mantle" or "projected".
    """

    if areaCalcType not in ["mantle", "projected"]:
        raise ValueError("areaCalcType must be 'mantle' or 'projected'. Currently is: {}".format(areaCalcType))

    # Init array to store area
    area = np.ones(len(penetrationDepth))

    # Calc cone side slope
    coneSideSlope = (baseRadius - coneTipRadius) / tipHeight

    for i, depth in enumerate(penetrationDepth):
        
        if depth >= tipHeight:
            # set the depth used in the calculation to the tipHeight
            depth = tipHeight
            radius = baseRadius
        else:
            # Calculate the radius at the current depth
            radius = coneTipRadius + depth * coneSideSlope

        # Check area selection and calculate accordingly
        if areaCalcType == "mantle":
            area[i] = calcConeLateralSurfaceArea(radius, depth)
        elif areaCalcType == "projected":
            area[i] = calcCircleArea(radius)  # Using circle area formula directly

    return area


def calcFFPBluntContactArea(penetrationDepth, tipHeight, baseRadius, areaCalcType):
    """
    Calculates the contact area for the blunt FFP tip (cylindrical shape).
    
    Parameters
    ----------
    penetrationDepth: np.ndarray
        Array of penetration depth values.
    tipHeight: float
        Height of the tip measured normal to the circular base.
    baseRadius: float
        Radius of the base of the blunt cylinder.
    areaCalcType: str
        Selection of "mantle" or "projected" for area calculation type.
    
    Returns
    -------
    np.ndarray
        Array of calculated contact area values for each penetration depth.

    Raises
    ------
    ValueError
        If `areaCalcType` is not "mantle" or "projected".
    """

    if areaCalcType not in ["mantle", "projected"]:
        raise ValueError("areaCalcType must be 'mantle' or 'projected'. Currently is: {}".format(areaCalcType))

    # Init array to store area calcs
    area = np.zeros(len(penetrationDepth))

    for i, depth in enumerate(penetrationDepth):
        if depth >= tipHeight:
            depth = tipHeight
        
        # radius is constant for cylinder
        radius = baseRadius

        # Check area selection
        if areaCalcType == "mantle":
            area[i] = calcCylinderSurfaceArea(radius, depth)  # Cylinder mantle area (without top and bottom)
        elif areaCalcType == "projected":
            area[i] = calcCircleArea(radius)  # Circle area for the projection

    return area

def calcParabolicContactArea(penetrationDepth, tipHeight, areaCalcType, radius_coeff = 2.4184):
    """
    Purpose: Calc the contact area for the parabolic type of  the bluedrop

    where:
    penetrationDepth: Array of penetration depth values
    tipHeight: Height of the tip measured normal to the circular base
    areaCalcType: Selection of mantle or projected
    radius_coeff: The coefficient required to calc the radius of the parabola from the depth of penetration
                  radius = \sqrt{depth/a} where a is from the parabola eqn. depth = a * radius^{2}
    """
    # init array to store the area calcs
    area = np.zeros(len(penetrationDepth))

    for i, depth in enumerate(penetrationDepth):
        #if the depth  is greater than the max length of the parabolic  cylinder
        if depth  >= tipHeight:
            # decrease the depth used in the calculatiion to the tipHeight
            depth = tipHeight

            # calc the radius
            radius = np.sqrt(radius_coeff * depth) 

            # Check area selection
            if areaCalcType == "mantle":
                # if selected  calc mantle area (Surface area of a parabola with the form y = f(x) = a*x^{2})
                area[i] = calcParabolaSurfaceArea(depth, radius_coeff)
            
            elif areaCalcType == "projected":
                #if selected calc projected area
                area[i] = calcCircleArea(radius)

        elif depth < tipHeight:

            # calc the radius
            radius = np.sqrt(radius_coeff * depth) 

            # Check area selection
            if areaCalcType == "mantle":
                # if selected  calc mantle area (Surface area of a parabola with the form y = f(x) = a*x^{2})
                area[i] = calcParabolaSurfaceArea(depth, radius_coeff)
            
            elif areaCalcType == "projected":
                #if selected calc projected area
                area[i] = calcCircleArea(radius)

    return area