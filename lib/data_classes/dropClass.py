from scipy.signal import find_peaks
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from lib.data_classes.exceptions import zeroLenError
from lib.signal_processing.signal_function import moving_average
from lib.general_functions.general_function import convert_accel_units, convert_time_units
from lib.general_functions.global_constants import GRAVITY_CONST # Gravity in m/s^2

class Drop:
    #Purpose: Store info about a pffp drop

    def __init__(self, containing_file, peak_index, file_drop_index, peak_info, pressure_check = False):

        # Store the inputs
        self.containing_file   = containing_file   # Store which file the drop is in 
        self.file_drop_index   = file_drop_index   # Tracks which drop this is in the file e.g. the first drop when there are 5 in the folder
        self.peak_info         = peak_info         # Store some information on the peak
        self.peak_index = peak_index
        self.water_drop = None                    # Store if the drop is a water drop
        self.processed = False                    # Tracks if the drop was processed (ie. all the information to resolve it's start and end was collected)
        self.manual_processed = False             # Tracks if the drop was processed manually
        self.indices_found = False                # Tracks if the indices were all found
        self.impulse_integration = True           # Use impulse integration (old method of integrating the function)
        # init Dict to hold drop indices
        self.drop_indices = {
                "release_index"      : None,
                "start_impulse_index": None,
                "end_impulse_index"  : None
            }
        
    def __str__(self):
        # Purpose: Outputs information about the drops
        return f"----- Drop Info ----- \nContaining file: {self.containing_file} \nFile Drop Index: {self.file_drop_index} \nWater Drop: {self.water_drop}\
            \nDrop indices: {self.drop_indices}"
    @staticmethod
    def check_arr_zero_length(arr, err_message):
        # Purpose: Check if an array has a length of zero and raise an error if it does
        if len(arr) ==0:
            raise zeroLenError(err_message)

    def find_release(self, accel, accel_offset = 1, height_tol=0.6, lower_accel_bound = 0.95, upper_accel_bound = 1.15):
        # Purpose: Find the release point of the drop

        # Flip the accel data so the point closest to free fall becomes a peak
        flip_accel = -1 * (accel - accel_offset)
        
        # Find the peaks of the flipped data
        index, _ = find_peaks(flip_accel, height = height_tol)

        # Get the peak free fall indices before the actual peak of the drop
        smaller_indices = np.where(index<self.peak_index)[0]
        
        self.check_arr_zero_length(smaller_indices, {"message":"Point meeting criteria not found", "criteria":"peak before impulse (release point)", \
                                                     "source":f"drop id: {self.file_drop_index} in {self.containing_file}"})

        # Get the closest "free fall index" before the peak of the drop
        closest_index = self.peak_index - np.min(self.peak_index - index[smaller_indices])
        
        # Find where the original data is close to 1
        release_index = np.where((accel[:closest_index] >lower_accel_bound) & (accel[:closest_index] < upper_accel_bound))[0]

        # Catch if there isn't a release
        self.check_arr_zero_length(release_index, {"message":"Point meeting criteria not found", "criteria":"release point", \
                                                    "source":f"drop id: {self.file_drop_index} in {self.containing_file}"})

        #Otherwise Store the release index
        return release_index[-1]

    def get_impulse_start(self, accel, time, time_tol = 0.005, sample_frq = 120_000, gradient_tol = 400, accel_lim = [5.8, 25],
                          window_size = 100, debug = False):

        # Purpose: Get the start of the impulse

        # Using the gradient of the acceleration to find the location of impact
        # Search time tolerance minutes before the impact for the start of the impulse
        
        # Make sure that the start index doesn't go below zero
        start_index = max(self.peak_index - int(time_tol * sample_frq), 0)
        if debug:
            store_time = time
            store_accel = accel
            # print(self.containing_file)
            # print("accel", accel)
            # print("time", time)
            # plt.plot(time, accel)

        # Only search from the time tolerance to the peak
        # time = np.array(time[start_index:self.peak_index])
        # accel = np.array(accel[start_index:self.peak_index])

        time = time[start_index:self.peak_index]
        accel = accel[start_index:self.peak_index]

        max_val = accel.max()

        # In the case of really high accelerations the smoothig done below bleads over and a higher gradient tolerance is needed
        if max_val > accel_lim[0] and max_val < accel_lim[1]:
            window_size = 7
            scale =  250 #1000
            gradient_tol = accel.max() * scale

        elif max_val >= accel_lim[1]: # Conditions for higher accelerations
            window_size = 7
            scale =  1000
            gradient_tol = accel.max() * scale 

        # Window average the acceleration and the time
        smoothed_time= moving_average(time, window_size)
        smoothed_accel = moving_average(accel, window_size)

        jerk = np.gradient(smoothed_accel, smoothed_time)

        if debug:
            plt.plot(smoothed_time, smoothed_accel, label = "smoothed")
            plt.plot(time, accel, label = "original")
            plt.legend()
            plt.show()

        # Get the points where the gradient meets this criteria
        smoothed_index =  np.where(jerk>gradient_tol)[0]
        self.check_arr_zero_length(smoothed_index, {"message":"Point meeting criteria not found", "criteria":"start impulse point", \
                                                    "source":f"drop id: {self.file_drop_index} in {self.containing_file}"})
        smoothed_index = smoothed_index[0]
        # Convert the index of the smoothed data back to the index of the original data

        # Selecting the last index where time is less than the selected smooth time
        index = np.where(time < smoothed_time[smoothed_index])[0][-1]
        # accel_cutoff_index = np.where()

        # Offset the index for the initial arr cut
        index = index + start_index 

        if debug:
            print("start accel value", accel[index])

            # fig, axs = plt.subplots(ncols = 1, nrows = 2)
            # print(index)
            # axs[0].plot(time, accel)
            # axs[0].scatter(time[index-start_index], accel[index-start_index], s=10, color = "red")
            # axs[1].plot(smoothed_time, jerk)
            # plt.show()

        # Return the first index that meets that criteria
        return index


    def get_impulse_end(self, accel, low_tol = 0.95, high_tol=1.05):
        # Purpose: Get the end of the impulse
        # index = np.where((accel[self.peak_index:]> low_tol) & (accel[self.peak_index:] < high_tol))[0]
        index = np.where(accel[self.peak_index:] < high_tol)[0]

        self.check_arr_zero_length(index, {"message":"Point meeting criteria not found", "criteria":"peak at impulse end", \
                                                     "source":f"drop id: {self.file_drop_index} in {self.containing_file}"})
        # Choose the second point
        point_index = 1

        return self.peak_index + index[point_index]
    
    def cut_accel_data(self, accel, time, input_units = {"accel":"g", "Time":"min"}):
        # Purpose: Store the sensor data only for where the drop is selected to be

        # Store the units of the acceleration and time
        self.units = input_units

        # Store the time in the drop
        self.time = convert_time_units(time, input_unit = self.units["Time"], output_unit = "s")

        # Change the time units
        self.units["Time"] = "s"

        # Get the end of the impulse
        end_drop_index = self.get_impulse_end(accel, low_tol=0.95, high_tol = 1.05)

        # Get the start of the impulse
        start_drop_index = self.get_impulse_start(accel, time)

        # Get the release index of the impulse
        # This often fails becasue drops are too close the beginning of the file, so try it 
        try: 
            release_index = self.find_release(accel, accel_offset =1, height_tol = 0.6, lower_accel_bound=0.95, upper_accel_bound=1.15)
        
        # If it fails due to not being found, catch the error the other indices then raise the error again
        except zeroLenError as err:
            release_index = None

            # df's store using the original indices. ie. the times and accelerations have the same indices as they had in the full arrays
            # Store the indices for later use (Stored in sequential order)
            self.drop_indices["release_index"] = release_index
            self.drop_indices["start_impulse_index"] = start_drop_index
            self.drop_indices["end_impulse_index"] = end_drop_index

            print("Release not found ")
            print(type(err))
            raise err
        
        # Store the time that's been calculted in seconds
        time = self.time

        # In the case finding thee results doesn't fail set the points
        self.drop_indices["release_index"] = release_index
        self.drop_indices["start_impulse_index"] = start_drop_index
        self.drop_indices["end_impulse_index"] = end_drop_index

        # Track that the indices were found
        self.indices_found = True

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
        # TODO: Update this so that the impulse df is 
        # Purpose: Integrate the impulse and store in impulse df

        # Temp storage for the df
        whole_df = self.whole_drop_df
            
        # Convert the acceleration units
        whole_df["accel"] = convert_accel_units(val = whole_df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

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
        # self.whole_drop_df = whole_df   
        
        # Temp storage for the df
        impulse_df = self.impulse_df       

        # Convert the units to m/s^2
        impulse_df["accel"] = convert_accel_units(val = impulse_df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

        # TODO: Make sure this offset makes sense
        impulse_df["accel"] = impulse_df["accel"] - GRAVITY_CONST

        start_index = self.drop_indices["start_impulse_index"]

        # Get the impact velocity
        init_velocity = -1 * whole_df["velocity"][start_index]

        # Cummulative integration takes "y" then "x" -> cummulative_trapezoid(y, x)
        impulse_velocity = cumulative_trapezoid(-1 * impulse_df["accel"], impulse_df["Time"], 
                                                initial = 0) + init_velocity


        # Need to cutoff the first time index
        impulse_displacment = cumulative_trapezoid(impulse_velocity, impulse_df["Time"], initial = 0.0)
        
        # Store the calculated values
        impulse_df["velocity"]     = impulse_velocity
        impulse_df["displacement"] = impulse_displacment

        # Update the acceleration units
        self.units["accel"] = "m/s^2"
        self.units["velocity"] = "m/s"
        self.units["displacement"] = "m"

        # Mark the drop processed
        self.processed = True
        
    def quick_view_impulse(self, interactive = True, figsize= [12, 8], legend = False):
        # Purpose: Provide a quick view of impulse
 
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
            print(time_units)

            # Give the entire figure a label
            fig.suptitle(f"File drop index: {self.file_drop_index}")

            plt.tight_layout()

            plt.show()

    def quick_view_release(self, interactive = True, figsize = [12, 8], legend = False):
        # Purpose: Provide a quick view of the full release

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

    def quick_view_impulse_selection(self, offset = 20, legend = True, draw_line = True, line_val = 0):
        start = self.drop_indices["start_impulse_index"] - offset
        end = self.drop_indices["end_impulse_index"]

        plt.plot(self.whole_drop_df["Time"].loc[start:end], self.whole_drop_df["accel"].loc[start:end], color = "blue", label= "release")
        plt.scatter(self.impulse_df["Time"], self.impulse_df["accel"], color = "red", label = "impulse")

        time_units = self.units["Time"]
        accel_units = self.units["accel"]

        plt.xlabel(f" Time ({time_units}) ")
        plt.ylabel(f"Acceleration ({accel_units})")

        if draw_line:
            plt.axhline(y = line_val)
        if legend:
            plt.legend()
        
        plt.show()

        