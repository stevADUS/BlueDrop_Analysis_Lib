from scipy.signal import find_peaks
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Init a constant to store the value of gravity
GRAVITY_CONST = 9.80665

class Drop:
    #Purpose: Store info about a pffp drop

    def __init__(self, containing_file, peak_index, file_drop_index, peak_info, pressure_check = False):

        # Store the inputs
        self.containing_file   = containing_file   # Store which file the drop is in 
        self.file_drop_index   = file_drop_index   # Tracks which drop this is in the file e.g. the first drop when there are 5 in the folder
        self.peak_info         = peak_info         # Store some information on the peak
        self.peak_index = peak_index
        self.water_drop = None                    # Store if the drop is a water drop
        self.store_whole_drop = False   # Set True if the drop from the release until the end should be stored, for false only the impulse is stored
        self.drop_indices = None
    def __str__(self):
        # Purpose: Outputs information about the drops
        return f"----- Drop Info ----- \nContaining file: {self.containing_file} \nFile Drop Index: {self.file_drop_index} \nWater Drop: {self.water_drop}\
            \nDrop indices: {self.drop_indices}"
    
    @staticmethod
    def convert_accel_units(val, input_unit, output_unit):
        # Purpose: Convert one acceleration unit to another

        # First convert everything to m/s^2
        input_to_standard ={
            "g": GRAVITY_CONST,
            "m/s^2":1
        }

        standard_to_output = {
            "g": 1/GRAVITY_CONST,
            "m/s^2":1
        }
        
        # Convert the val to m/s^2 (standard value)
        standard = val * input_to_standard[input_unit]

        # Convert the standard value to the desired value
        output_val = standard * standard_to_output[output_unit]

        # Return the value
        return output_val
    
    def find_release(self, accel, accel_offset = 1, time_tol= 0.1, height_tol=0.6, frq = 2000, lower_accel_bound = 0.95, upper_accel_bound = 1.15):
        # Purpose: Find the release point of the drop

        # Flip the accel data so the point closest to free fall becomes a peak
        flip_accel = -1 * (accel - accel_offset)
        
        # Find the peaks of the flipped data
        index, _ = find_peaks(flip_accel, height = height_tol)
                
        # Get the peak free fall indices before the actual peak of the drop
        smaller_indices = np.where(index<self.peak_index)

        # Get the closest "free fall index" before the peak of the drop
        closest_index = self.peak_index - np.min(self.peak_index - index[smaller_indices])
        
        # Find where the original data is close to 1
        release_index = np.where((accel[:closest_index] >lower_accel_bound) & (accel[:closest_index] < upper_accel_bound))[0]

        # Store the release index
        return release_index[-1]

    def get_impulse_start(self, accel, low_tol = 0.95, high_tol=1.08):
        # Purpose: Get the start of the impulse

        index = np.where((accel[:self.peak_index] >low_tol) & (accel[:self.peak_index] < high_tol))[0]
        
        return index[-1]
        
    def get_impulse_end(self, accel, low_tol = 0.95, high_tol=1.05):
        # Purpose: Get the end of the impulse
        index = np.where((accel[self.peak_index:]> low_tol) & (accel[self.peak_index:] < high_tol))[0]

        return self.peak_index + index[0]
        
    def cut_accel_data(self, accel, time, sample_frq = 2000, time_tol = 0.1, input_units = {"accel":"g", "Time":"s"}):
        # Purpose: Store the sensor data only for where the drop is selected to be

        # Store the units of the acceleration and time
        self.units = input_units

        # Get the start of the impulse
        start_drop_index = self.get_impulse_start(accel, low_tol=0.95, high_tol=1.08)
        # print("Start drop index: ", start_drop_index)
        # print("start impulse time: ", time[start_drop_index])

        # Get the end of the impulse
        end_drop_index = self.get_impulse_end(accel, low_tol=0.95, high_tol = 1.05)

        release_index = self.find_release(accel, accel_offset =1, time_tol = 0.1, height_tol = 0.6, frq = sample_frq, lower_accel_bound=0.95, upper_accel_bound=1.15)

        # df's store using the original indices. ie. the times and accelerations have the same indices as they had in the full arrays
        # Store the indices for later use (Stored in sequential order)
        self.drop_indices = {
            "release_index"      : release_index,
            "start_impulse_index": start_drop_index,
            "end_impulse_index"  : end_drop_index
        }

        if self.store_whole_drop:
            # Store from the release point until the end of the drop
            whole_drop_accel= accel[release_index:end_drop_index]
            whole_drop_time =  time[release_index:end_drop_index]

            # Store the drop from the release to the end
            self.whole_drop_df = pd.DataFrame(data = {
                "Time": whole_drop_time,
                "accel": whole_drop_accel
            })

        # Store the impulse either way
        impulse_accel = accel[start_drop_index:end_drop_index]
        impulse_time  = time[start_drop_index:end_drop_index]
        
        # Store the impulse time and acceleration
        self.impulse_df = pd.DataFrame(data = {
            "Time": impulse_time,
            "accel": impulse_accel
        })

    def integrate_accel_data(self):
        # Purpose: Integrate the impulse and store in impulse df

        # Temp storage for the df
        impulse_df = self.impulse_df       

        # Convert the units to m/s^2
        impulse_df["accel"] =  self.convert_accel_units(val = impulse_df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

        # TODO: Make sure this offset makes sense
        impulse_df["accel"] = impulse_df["accel"] - GRAVITY_CONST

        # Cummulative integration takes "y" then "x" -> cummulative_trapezoid(y, x)
        impulse_velocity = np.array(cumulative_trapezoid(impulse_df["accel"], impulse_df["Time"]))

        # Need to cutoff the first time index
        impulse_displacment = np.array(cumulative_trapezoid(impulse_velocity, impulse_df["Time"][1:]))

        # Pad the velocity with one zero
        impulse_velocity = np.concatenate((np.zeros(1), impulse_velocity))
        
        # Pad the displacement with two zeros
        impulse_displacment = np.concatenate((np.zeros(2), impulse_displacment))
        
        # Store the calculated values
        impulse_df["velocity"]     = impulse_velocity
        impulse_df["displacement"] = impulse_displacment

        # Store the df
        self.impulse_df = impulse_df

        # if storing the whole drop, integrate the whole drop as well
        if self.store_whole_drop:
            # Temp storage for the df
            whole_df = self.whole_drop_df
            
            # Convert the acceleration units
            whole_df["accel"] = self.convert_accel_units(val = whole_df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

            # Apply the offset
            whole_df["accel"] = whole_df["accel"] - GRAVITY_CONST

            # Calc the velocity and the displacement
            # Cummulative integration takes "y" then "x" -> cummulative_trapezoid(y, x)
            whole_velocity = cumulative_trapezoid(whole_df["accel"], whole_df["Time"])
            whole_displacement = cumulative_trapezoid(whole_velocity, whole_df["Time"][1:])
            
            # Pad the velocity 
            whole_velocity = np.concatenate((np.zeros(1), whole_velocity))
            whole_displacement = np.concatenate((np.zeros(2), whole_displacement))

            whole_df["velocity"] = whole_velocity
            whole_df["displacement"] = whole_displacement

            # Store the info in the df
            self.whole_drop_df = whole_df   
        
        # Update the acceleration units
        self.units["accel"] = "m/s^2"
        self.units["velocity"] = "m/s"
        self.units["displacement"] = "m"

    def quick_view_impulse(self, interactive = True, figsize= [12, 8], legend = False):
        # Purpose: Provide a quick view of impulse
 
        # Check if the data hasn't been stored and raise an error if so
        if not self.store_whole_drop:
            raise ValueError("Release data has not been stored")
        # Otherwise proceed with plotting

        # Temp df storage
        df = self.impulse_df
        time = df["Time"]
        accel = df["accel"]
        vel = df["velocity"]
        displacement = df["displacement"]

        accel_units = self.units["accel"]
        vel_units = self.units["velocity"]
        disp_units = self.units["displacement"]
        time_units=  self.units["Time"]

        if interactive:
            fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)

            fig.add_trace(
                go.Scatter(x = time, y= accel, mode = "lines", name = "Acceleration"),
                row = 1, col = 1
            )

            fig.add_trace(
                go.Scatter(x= time, y= vel, mode = "lines", name = "Velocity"),
                row = 2, col = 1
            )

            fig.add_trace(
                go.Scatter(x = time, y = displacement, mode = "lines", name = "Displacement"),
                row = 3, col = 1
            )
            
            # Update xaxis properties
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)



            # Update yaxis properties
            fig.update_yaxes(title_text=f"Acceleration ({accel_units})", row=1, col=1)
            fig.update_yaxes(title_text=f"Velocity ({vel_units})", row=2, col=1)
            fig.update_yaxes(title_text=f"Displacement ({disp_units})", row=3, col=1)

            # Update figure title
            fig.update_layout(height = figsize[1] *100, width = figsize[0] *100,
                            title_text=f"File Drop index: {self.file_drop_index}")

            # Turn off interactivity
            fig.show()
        else:
            # Use matplotlib
            fig, axs = plt.subplots(ncols = 1, nrows = 3, figsize = (figsize[0], figsize[1]))

            axs[0].plot(time, accel, label = f"acceleration {accel_units}")
            axs[1].plot(time, vel, label = f"velocity {vel_units}")
            axs[2].plot(time, displacement, label = f"Displacement {disp_units}")

            
            # Turn on the legends
            if legend:
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

            # Label the y-axis
            axs[0].set_ylabel(f"Acceleration [{accel_units}]")
            axs[1].set_ylabel(f"Velocity [{vel_units}]")
            axs[2].set_ylabel(f"Displacement [{disp_units}]")

            # Label the x-axis
            axs[0].set_xlabel(f"Time [{time_units}]")
            axs[1].set_xlabel(f"Time [{time_units}]")
            axs[2].set_xlabel(f"Time [{time_units}]")

            # Give the entire figure a label
            fig.suptitle(f"File drop index: {self.file_drop_index}")

            plt.tight_layout()
            plt.show()

    def quick_view_release(self, interactive = True, figsize = [12, 8], legend = False):
        # Purpose: Provide a quick view of the full release
 
        # Check if the data hasn't been stored and raise an error if so
        if not self.store_whole_drop:
            raise ValueError("Release data has not been stored")
        # Otherwise proceed with plotting

        # Temp df storage
        df = self.whole_drop_df
        time = df["Time"]
        accel = df["accel"]
        vel = df["velocity"]
        displacement = df["displacement"]

        accel_units = self.units["accel"]
        vel_units = self.units["velocity"]
        disp_units = self.units["displacement"]
        time_units=  self.units["Time"]

        if interactive:
            fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)

            fig.add_trace(
                go.Scatter(x = time, y= accel, mode = "lines", name = "Acceleration"),
                row = 1, col = 1
            )

            fig.add_trace(
                go.Scatter(x= time, y= vel, mode = "lines", name = "Velocity"),
                row = 2, col = 1
            )

            fig.add_trace(
                go.Scatter(x = time, y = displacement, mode = "lines", name = "Displacement"),
                row = 3, col = 1
            )
            
            # Update xaxis properties
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)



            # Update yaxis properties
            fig.update_yaxes(title_text=f"Acceleration ({accel_units})", row=1, col=1)
            fig.update_yaxes(title_text=f"Velocity ({vel_units})", row=2, col=1)
            fig.update_yaxes(title_text=f"Displacement ({disp_units})", row=3, col=1)

            # Update figure title
            fig.update_layout(height = figsize[1] *100, width = figsize[0] *100,
                            title_text=f"File Drop index: {self.file_drop_index}")

            # Turn off interactivity
            fig.show()
        else:
            # Use matplotlib
            fig, axs = plt.subplots(ncols = 1, nrows = 3, figsize = (figsize[0], figsize[1]))

            axs[0].plot(time, accel, label = f"acceleration {accel_units}")
            axs[1].plot(time, vel, label = f"velocity {vel_units}")
            axs[2].plot(time, displacement, label = f"Displacement {disp_units}")

            
            # Turn on the legends
            if legend:
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

            # Label the y-axis
            axs[0].set_ylabel(f"Acceleration [{accel_units}]")
            axs[1].set_ylabel(f"Velocity [{vel_units}]")
            axs[2].set_ylabel(f"Displacement [{disp_units}]")

            # Label the x-axis
            axs[0].set_xlabel(f"Time [{time_units}]")
            axs[1].set_xlabel(f"Time [{time_units}]")
            axs[2].set_xlabel(f"Time [{time_units}]")

            # Give the entire figure a label
            fig.suptitle(f"File drop index: {self.file_drop_index}")

            plt.tight_layout()
            plt.show()

    def concate_accelerations(self):
        # Purpose: Tie the accelerometers together where they max out
        pass