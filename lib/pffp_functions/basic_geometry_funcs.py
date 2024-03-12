import numpy as np

def calcCylinderSurfaceArea(radius, height):
    # Calculates the Surface area of a cylinder
    # Inputs:
        # radius: Radius of the cylinder
        # height of the cylinder

        baseArea = np.pi * radius**2 # Surface area of the base of the cylinder
        sideArea = 2 * np.pi * radius * height # Surface area on the side of the cylinder
        totalSurfaceArea = baseArea + sideArea
        return totalSurfaceArea

def calcConeLateralSurfaceArea(radius, height):
    # Calculates the lateral surface of a cone (Doesn't calc the surface area of the base of the cone)
    # Inputs:
        # radius: radius of the base of the cone
        # height: height of the cone (measured normal to the base)
        # sideSlope: Side slope of the cone

    return np.pi * radius * np.sqrt(height**2 + radius**2)

def calcCircleArea(radius):
    # Calc the area of a circle
    # Input:
        # radius: radius of the circle
    return np.pi * radius**2

def calcParabolaSurfaceArea(depth, radius_coeff):
    # Purpose calc the surface of a rotated parabola 
    
    # Derivation of the surface area
    # See first thereom of: https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem#The_first_theorem
    # A_{mantle} = 2 * \pi \int_{0}^{h} r * \sqrt{1 + r'(h)^{2}} dh
    # where
    # r: radius, r = \sqrt{radius_coeff * h}
    # h: height on the parabola (or equivalently the depth of penetration)
    # r': (dr/dh), r' = 0.5 * (\sqrt{radius_coeff/h}) 
    
    #TODO: Check the work on this equation one more time
    # Integral has an analytic solution (goes from depth == 0 to depth == current depth)
    area = 4/3 * PI/radius_coeff *( (radius_coeff * depth + radius_coeff**2/4)**(3/2) - (radius_coeff**2/4)**(3/2) )

    return area