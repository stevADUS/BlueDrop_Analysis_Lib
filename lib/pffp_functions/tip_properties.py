def return_tip_properties(tiptype):
    # Purpose: Returns the mass and length for the specified tip type

    if tiptype == "cone":
        mass = 7.71
        length = 8.48 - 0.93 #originally 7.87, 7.57 for perfect 60 deg, 8.48 measured - .93 measured 90 deg

    elif tiptype == "parabola":
        mass = 9.15
        length = 7.915

    elif tiptype == "blunt":
        mass = 10.30
        length = 8.57
    else:
        raise ValueError("Input tiptype: " + tiptype + "is not an option")
    
    return mass, length