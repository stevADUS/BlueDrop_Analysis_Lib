# import numpy as np


def calc_drag_force(rho_fluid, velocity, drag_coeff, cross_section_area):
    # Purpose: Calc the drag force using:
    # Link to drag info: https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation
    
    # $F_{Drag} = \frac{1}{2} \rho v^{2} C_{D} A$
    # where:
    # rho:   fluid density
    # v:     object velocity
    # C_{D}: Drag coefficent for the object
    # A:     Cross sectional area

    return 0.5 * rho_fluid * velocity**2 * drag_coeff * cross_section_area

def calc_buoyant_force(rho_fluid, displaced_volume, gravity):
    # Purpose: Calc the bouyant for an object in a fluid
    # Link to buoyant force info: https://en.wikipedia.org/wiki/Buoyancy#Forces_and_equilibrium

    # $F_{B} = \rho, V_{disp} g
    #where:
    # \rho:     Fluid density
    # V_{disp}: Volume of fluid displaced
    # g:        gravity

    return rho_fluid * displaced_volume * gravity

if __name__ == "__main__":
    # Calc a test run of the equation

    # Answer should be 24
    drag = calc_drag_force(rho_fluid=1, velocity=2, drag_coeff=3, cross_section_area=4)
    
    # Store the correc answer for the above input vals
    correct_ans = 23

    print(f"Drag force: {drag}")
    if drag == correct_ans:
        print("Answer is correct")
    else:
        raise ValueError(f"Calculated value is {drag} correct answer is {correct_ans}")

