from numpy import sqrt as np_sqrt

from lib.general_functions.global_constants import PI_CONST

def calcCylinderSurfaceArea(radius, height):
    """
    Purpose: Calculates the Surface area of a cylinder
    where:
        radius: Radius of the cylinder
        height of the cylinder

    Parameters

    ----------
    radius : float
        The radius of the cylinder's base, typically in units of length (e.g., meters).
    height : float
        The height of the cylinder, typically in units of length (e.g., meters).

    Returns

    -------
    float
        The total surface area of the cylinder, typically in square units (e.g., square meters).

    Notes

    -----
    The total surface area is calculated as the sum of the area of the two circular bases and the area of the lateral surface. The formula used is:

    .. math::

        A_{\text{total}} = 2 \pi r^2 + 2 \pi r h

    where:
    - :math:`r` is the radius of the cylinder's base.
    - :math:`h` is the height of the cylinder.
    - :math:`\pi` is a mathematical constant approximately equal to 3.14159.

    Example
    
    -------
    Calculate the surface area of a cylinder with a radius of 3 units and a height of 5 units:

    >>> radius = 3
    >>> height = 5
    >>> surface_area = calcCylinderSurfaceArea(radius, height)
    >>> print(surface_area)
    150.79644737231007  # example output

    """
    baseArea = PI_CONST * radius**2 # Surface area of the base of the cylinder
    sideArea = 2 * PI_CONST * radius * height # Surface area on the side of the cylinder
    totalSurfaceArea = baseArea + sideArea
    return totalSurfaceArea

def calcConeLateralSurfaceArea(radius, height):
    """
    Purpose: Calculates the lateral surface of a cone (Doesn't calc the surface area of the base of the cone)
    where:
        radius: radius of the base of the cone
        height: height of the cone (measured normal to the base)
        sideSlope: Side slope of the cone
    """
    return PI_CONST * radius * np_sqrt(height**2 + radius**2)

def calcCircleArea(radius):
    """
    Purpose: Calc the area of a circle
    where:
        radius: radius of the circle
    """

    return PI_CONST * radius**2

def calcParabolaSurfaceArea(depth, radius_coeff):
    """
    Purpose: Calc the surface of a rotated parabola 
    
    Derivation of the surface area
    See first thereom of: https://en.wikipedia.org/wiki/Pappus%27s_centroid_theorem#The_first_theorem
    
    Eqn:
        A_{mantle} = 2 * \pi \int_{0}^{h} r * \sqrt{1 + r'(h)^{2}} dh
        
    where:
        r: radius, r = \sqrt{radius_coeff * h}
        h: height on the parabola (or equivalently the depth of penetration)
        r': (dr/dh), r' = 0.5 * (\sqrt{radius_coeff/h}) 
    """

    #TODO: Check the work on this equation one more time
    # Integral has an analytic solution (goes from depth == 0 to depth == current depth)
    area = 4/3 * PI_CONST/radius_coeff * ( (radius_coeff * depth + radius_coeff**2/4)**(3/2) - (radius_coeff**2/4)**(3/2) )

    return area

if __name__ == "__main__":
    # Add some testing here
    pass