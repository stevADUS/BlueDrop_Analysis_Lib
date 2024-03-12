#Standard imports 
import numpy as np

# From local library
from basic_geometry_funcs import calcCircleArea, calcConeLateralSurfaceArea, calcCylinderSurfaceArea, calcParabolaSurfaceArea

def calcFFPContactArea(penetrationDepth,  maxLength, areaCalcType, tipType, baseRadius = 4.375, coneTipRadius = 0.22, coneTipHeight = 7.55, radius_coeff = 2.4184):
    # Function calls the required functions to calculate the generated area for a cone
    # Inputs:
        # penetrationDepth: Array of depths the FFP has penetrated, [m]
        # maxLength: max length of penetration before radius become constant, [cm]
        # areaCalcType: "mantle or projected"
        # tipType: blunt or cone tip type       
        # baseRadius: Base radius of the cone and blunt tip, [cm]
        # coneTipRadius: Tip radius of the cone??, [cm]
        # coneTipHeight: height of the cone, [cm]
        # radius_coeff: coefficient from the parabola fit the parabolic cone tip
        
    # Convert penetration depth to centimeters [cm]
    penetrationDepth = penetrationDepth * 100 #[cm]

    # Calc area for cone tip
    if tipType == "cone":
        area = calcFFPConeContactArea(penetrationDepth, maxLength, baseRadius, coneTipRadius, coneTipHeight, areaCalcType)

     # Calc area for blunt tip
    elif tipType == "blunt":
        area = calcFFPBluntContactArea(penetrationDepth, maxLength, baseRadius, areaCalcType)
    
    elif tipType == "parabola":
        area = calcParabolicContactArea(penetrationDepth, maxLength, areaCalcType, radius_coeff)

    # Convert to m^2
    return area/1e4 #[m^2]

#TODO: Could use a mask to calc the values for the whole array that is below the maxLength and init array for the part that is above

def calcFFPConeContactArea(penetrationDepth, maxLength, baseRadius, coneTipRadius, coneTipHeight, areaCalcType):
    # Function calculates the contact area for the cone top of the FFP

    # penetrationDepth: Array of penetration depth values
    # maxLength: Max length before the lateral area of the FFP is considered
    # baseRadius: Radius of the Cone at the base (Circular side)
    # coneTipRadius: Radius of the cone tip (is not zero)
    # coneTipHeight: Height of the cone when measured normal to the circular base
    # areaCalcType: Selection of mantle or projected

    # Init array to store area
    area = np.ones(len(penetrationDepth))

    # Calc cone side slope
    coneSideSlope = (baseRadius-coneTipRadius)/coneTipHeight

    for i, depth in enumerate(penetrationDepth):
        
        # if the depth is greater than the maxLength of the cone
        if depth > maxLength:
            # decrease the depth used in the calculation to the maxLength
            depth = maxLength
        
        # Calc the radius
        radius = coneTipRadius + depth * coneSideSlope

        # Check area selection
        if areaCalcType == "mantle":
            # if selected calc mantle area (Same as surface area without the base of cone)
            area[i] = calcConeLateralSurfaceArea(radius, depth)

        elif areaCalcType == "projected":
            # if selected calc projected area
            area[i] = calcCircleArea(radius)

        else: 
            return ValueError("areaCalcType must be mantle or projected. Currently is: {}".format(areaCalcType))

    return area

def calcFFPBluntContactArea(penetrationDepth, maxLength, baseRadius, areaCalcType):

    # init array to store area calcs
    area = np.zeros(len(penetrationDepth))

    for i, depth in enumerate(penetrationDepth):
        # if the depth is greater than the maxLength of the blunt cylinder
        if depth > maxLength:
            # decrease the depth used in the calculation to the maxLength
            depth = maxLength
        
        # Check the maxLength in one line than calc the area in one line instead of looping
        # radius is constant for cylinder
        radius = baseRadius

        # Check area selection
        if areaCalcType == "mantle":
            # if selected calc mantle area (Surface Area of a cylinder)
            area[i] = calcCylinderSurfaceArea(radius, depth)

        elif areaCalcType == "projected":
            # if selected calc projected area
            area[i] = calcCircleArea(radius)

        else: 
            return ValueError("areaCalcType must be mantle or projected. Currently is: {}".format(areaCalcType))
        
    # Might need to convert this to m^2
    return area

def calcParabolicContactArea(penetrationDepth, maxLength, areaCalcType, radius_coeff = 2.4184):
    # Purpose: Calc the contact area for the parabolic type of  the bluedrop
    # radius_coeff: The coefficient required to calc the radius of the parabola from the depth of penetration
    #               radius = \sqrt{depth/a} where a is from the parabola eqn. depth = a * radius^{2}

    # init array to store the area calcs
    area = np.zeros(len(penetrationDepth))

    for i, depth in enumerate(penetrationDepth):
        #if the depth  is greater than the max length of the parabolic  cylinder
        if depth  > maxLength:
            # decrease the depth  used in the calculatiion to the maxLength
            depth = maxLength

        # calc the radius
        radius = np.sqrt(radius_coeff * depth) 

        # Check area selection
        if areaCalcType == "mantle":
            # if selected  calc mantle area (Surface area of a parabola with the form y = f(x) = a*x^{2})
            area[i] = calcParabolaSurfaceArea(depth, radius_coeff)
        
        elif areaCalcType == "projected":
            #if selected calc projected area
            area[i] = calcCircleArea(radius)