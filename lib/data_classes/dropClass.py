from scipy.signal import find_peaks
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from lib.data_classes.exceptions import zeroLenError
from lib.signal_processing.signal_function import moving_average
from lib.general_functions.general_function import convert_accel_units, convert_time_units, convert_mass_units, convert_length_units
from lib.general_functions.global_constants import GRAVITY_CONST, ALLOWED_TIP_TYPES_LIST
from lib.mechanics_functions.bearing_capacity_funcs import calc_dyn_bearing_capacity, calc_qs_bearing_capacity
from lib.pffp_functions.cone_area_funcs import calc_pffp_contact_area

# TODO: In the future add a bearing capacity class to store all of the information about a bearing capacity calculation
# TODO: make a pffp class that store all of the information about the pffp

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
        self.manually_processed = False           # Tracks if the drop was processed manually
        self.indices_found = False                # Tracks if the indices were all found
        self.only_impulse  = False                # Use impulse integration (old method of integrating the function)
        self.qDyn_bearing_col_name = None           # Most recent dynamic bearing column name
        self.bearing_df = None                    # Will be used later to store the dynamic and quasistatic (qsbc) bearing capacities

        # Init dict to hold the unit properties
        self.units = {
                      "mass": "kg",
                      "Time": None,
                      "displacement":None,
                      "velocity": None, 
                      "accel": None
        }

        # init Dict to hold drop indices
        self.drop_indices = {
                "release_index"      : None,
                "start_impulse_index": None,
                "end_impulse_index"  : None
            }
        self.ref_velocities = {}                 # Init dict to hold the reference velocities for qsbc calculations for output to meta data
        
        # Init dict to hold pffp config information
        self.pffp_config = {
                            "volume": None,
                            "tip_type": None,
                            "tip_props": None,
                            "area_type": None,
                            "tip_col_name": None
            }
        
    def __str__(self):
        # Purpose: Outputs information about the drops
        return f"----- Drop Info ----- \nContaining file: {self.containing_file} \nFile Drop Index: {self.file_drop_index} \nWater Drop: {self.water_drop}\
            \nDrop indices: {self.drop_indices} \nProcessed: {self.processed} \nManually Processed: {self.manually_processed}"
    
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
        self.units.update(input_units)

        # Store the time in the drop
        self.time = convert_time_units(time, input_unit = self.units["Time"], output_unit = "s")

        # Change the time units
        self.units["Time"] = "s"

        if not self.manually_processed:
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
                raise err
            
            # In the case finding thee results doesn't fail set the points
            self.drop_indices["release_index"] = release_index
            self.drop_indices["start_impulse_index"] = start_drop_index
            self.drop_indices["end_impulse_index"] = end_drop_index

        # Store the time that's been calculted in seconds
        time = self.time

        # Track that the indices were found
        self.indices_found = True

        # Make and store the relase df
        self.make_release_df(accel, time)

        # Make and store the impulse df
        self.make_impulse_df(accel, time)

    def make_impulse_df(self, accel, time):
        # Store the impulse either way
        impulse_accel = accel[self.drop_indices["start_impulse_index"]:self.drop_indices["end_impulse_index"]]
        impulse_time  = time[self.drop_indices["start_impulse_index"]:self.drop_indices["end_impulse_index"]]
        
        # Store the impulse time and acceleration
        self.impulse_df = pd.DataFrame(data = {
            "Time": impulse_time,
            "accel": impulse_accel
        })

    def make_release_df(self, accel, time):
        # Store from the release point until the end of the drop
        whole_drop_accel= accel[self.drop_indices["release_index"]:self.drop_indices["end_impulse_index"]]
        whole_drop_time =  time[self.drop_indices["release_index"]:self.drop_indices["end_impulse_index"]]

        # Store the drop from the release to the end
        self.release_df = pd.DataFrame(data = {
            "Time": whole_drop_time,
            "accel": whole_drop_accel
        })

    def integrate_accel_data(self):
        # Purpose: Integrates the drop from the release point to the end of the impulse. 
        # Mark the drop processed
        
        if self.only_impulse:
            # Likely was a manaually selected drop only do the impulse integration
            self.impulse_integration()
        else:
            # Integrate the release
            self.release_integration()

            # Then get the impulse data from that
            col_names = ["accel", "velocity", "displacement"]
            
            # TODO: here is the problem something is going wrong with this slicing. Check this tomorrow
            # Just select the part of the release df that is needed
            self.impulse_df[col_names] = self.release_df[col_names].loc[self.drop_indices["start_impulse_index"]:self.drop_indices["end_impulse_index"]]

            # Flip the sign of velocity column
            self.impulse_df[col_names[1]] = -1 * self.impulse_df[col_names[1]]
            
            # Flip the sign of displacement column and make it zero at the start
            self.impulse_df[col_names[2]] = -1 * (self.impulse_df[col_names[2]] - self.impulse_df[col_names[2]].iloc[0])

        # Update the units
        self.units["accel"] = "m/s^2"
        self.units["velocity"] = "m/s"
        self.units["displacement"] = "m" 

        self.processed = True

    def release_integration(self):
        # Purpose: Wrapper for the release and impulse intgration functions. Controls whether the full release integration is done 
        # or just the impulse integration 

        # Temp storage for the df
        df = self.release_df.copy()
            
        # Convert the acceleration units
        df["accel"] = convert_accel_units(val = df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

        # Apply the offset
        df["accel"] = df["accel"] - GRAVITY_CONST

        # Calc the velocity and the displacement
        # Cummulative integration takes "y" then "x" -> cummulative_trapezoid(y, x)
        velocity = cumulative_trapezoid(df["accel"], df["Time"], initial = 0.0)

        displacement = cumulative_trapezoid(velocity, df["Time"], initial = 0.0)
        
        # Update the accel columns and add the new velocity and displacement columns
        self.release_df["accel"] = df["accel"]
        self.release_df["velocity"] = velocity
        self.release_df["displacement"] = displacement

    def impulse_integration(self, init_velocity = 0.0):
        # Purpose: Integrate the impulse. if self.impulse_integration == True the release df is not used to calculate the initial velocity
        
        # Temp storage for the df
        df = self.impulse_df.copy()       

        # Convert the units to m/s^2
        df["accel"] = convert_accel_units(val = df["accel"], input_unit = self.units["accel"], output_unit = "m/s^2")

        df["accel"] = df["accel"] - GRAVITY_CONST
        
        
        # Cummulative integration takes "y" then "x" -> cummulative_trapezoid(y, x)
        velocity = cumulative_trapezoid(df["accel"], df["Time"], initial = 0) + init_velocity

        # Flip the velocity because the probe is deaccelerting and you need the impact velocity at the beginning 
        velocity = velocity.max() - velocity

        # Need to cutoff the first time index
        displacement = cumulative_trapezoid(velocity, df["Time"], initial = 0.0)
        
        # Store the calculated values
        self.impulse_df["accel"]        = df["accel"] 
        self.impulse_df["velocity"]     = velocity
        self.impulse_df["displacement"] = displacement
    
    def calc_drop_qs_bearing(self, strain_rate_correc_type = "log", k_factor = 0.1, ref_velocity = 0.02, bearing_name = None, use_k_name = True, other_name = ""):
        """
        Purpose: Calc the quasi-static bearing capacity (qsbc) for this particular drop and store it in the self.bearing_df
                 with a unique name using the k-factor or a user input
        """

        velocity = self.impulse_df["velocity"]

        if bearing_name is None:
            dynamic_bearing = self.bearing_df[self.qDyn_bearing_col_name]
        else:
            dynamic_bearing = self.bearing_df[bearing_name]

        quasi_static_bearing = calc_qs_bearing_capacity(velocity=velocity, strainrateCorrectionType=strain_rate_correc_type,
                                                        qDyn = dynamic_bearing, k_factor=k_factor, ref_velocity= ref_velocity)
        
        # Construct the name 
        if use_k_name:
            # Construct the column name appending the k factor to make it unique 
            col_name = "qsbc_{}_{}".format(self.pffp_config["area_type"][:4], k_factor)
        else:
            col_name = "qsbc_" + str(other_name)
        
        # Store the reference velocity for later output to metadata and reference velocity should be store for each calculation
        self.ref_velocities[col_name] = ref_velocity

        # Store the column in the df
        self.bearing_df[col_name] = quasi_static_bearing

    def calc_drop_dynamic_bearing(self, gravity = GRAVITY_CONST, rho_w = 1020, other_name = None):
        """
        Purpose: Calc the dynamic bearing capacity (qsbc) for this particular drop and store it in the self.bearing_df 
        """

        # Temp store the necessary parameters
        accel = self.impulse_df["accel"]
        pffp_props = self.pffp_config["tip_props"]
        tip_val_col = self.pffp_config["tip_col_name"]

        mass = pffp_props.loc[pffp_props["Properties"] == "pffp_mass"][tip_val_col].iloc[0]
        volume = pffp_props.loc[pffp_props["Properties"] == "pffp_volume"][tip_val_col].iloc[0]
        
        if other_name is None:
            contact_area_col_name = "{}_{}".format("contact_area", self.pffp_config["area_type"])
            bearing_col_name = "{}_{}".format("qDyn", self.pffp_config["area_type"])

        else:
            contact_area_col_name = "{}_{}".format("contact_area", other_name)
            bearing_col_name = "{}_{}".format("qDyn", other_name)

        # Store the latest bearing column name
        self.qDyn_bearing_col_name = bearing_col_name

        # Check that the water drop value is set
        if self.water_drop is None:
            raise ValueError("To calculate the dynamic bearing capacity the flag for deciding if the drop is in water or not must be set")
        
        contact_area = self.bearing_df[contact_area_col_name]

        # Calc the dynamic bearing capacity
        qDyn = calc_dyn_bearing_capacity(pffp_accel=accel, pffp_mass=mass, contact_area=contact_area, pffp_volume=volume,
                                         water_drop=self.water_drop, gravity=gravity, rho_w = rho_w)
        
        
        # Store the dynamic bearing capcity result
        self.bearing_df[bearing_col_name] = qDyn

    def get_pffp_tip_values(self, pffp_id, tip_type, date_string, file_dir):
        """
        Purpose: Read and store the tip values
        """

        # Check that the tip type is allowed
        if not tip_type in ALLOWED_TIP_TYPES_LIST:
            raise ValueError("Tip type of {} is not allowed".format(tip_type))
        
        sheet_name = "bluedrop_{}".format(pffp_id)
        self.pffp_config["tip_type"] = tip_type

        tip_table = pd.read_excel(file_dir, sheet_name)

        # Construct the column name
        col_name = "{}_{}".format(tip_type, date_string)
        
        # Store the name of the column that the values live in
        self.pffp_config["tip_col_name"] = col_name

        self.pffp_config["tip_props"] = tip_table[["Properties", "units", col_name]]

    def convert_tip_vals(self):
        """"
        Purpose: Convert the units of the tip properties to the units used in the used of the analysis
        """

        # Store the names of the columns need for relabelling
        val_col_name = self.pffp_config["tip_col_name"]
        properties_col_name = "Properties"

        # Temp storage of the df
        df = self.pffp_config["tip_props"]

        # Get the mass row and the row index
        row = df.loc[df[properties_col_name] == "pffp_mass"]
        row_index = df.index[df['Properties'] == "pffp_mass"].tolist()

        # Store the value
        mass_val = row[val_col_name]

        # Store the unit
        mass_unit = row["units"].iloc[0]

        # Convert the unit and store the value back in the df
        df[val_col_name][row_index] = convert_mass_units(mass_val, mass_unit, self.units["mass"])

        df["units"][row_index] = self.units["mass"]
        
        # List of props that may need to have there units converted
        match self.pffp_config["tip_type"]:
            case "parabola":
                lengths_need_conversion = ["tip_height", "base_radius"]  
            case "blunt":
                lengths_need_conversion = ["tip_height", "base_radius"]  
            case "cone":
                lengths_need_conversion = ["tip_height", "base_radius", "tip_radius"]

        # Loop over the lengths that need to be converted
        for label in lengths_need_conversion:
             row = df.loc[df[properties_col_name] == label]
             row_index = df.index[df['Properties'] == label].tolist()
     
             # Store the value
             val = row[val_col_name]
     
             # Store the unit
             length_unit = row["units"].iloc[0]
     
             # Convert the unit and store the value back in the df
             df[val_col_name][row_index] = convert_length_units(val, length_unit, self.units["displacement"])
     
             df["units"][row_index] = self.units["displacement"]
            
    def calc_drop_contact_area(self, area_type):

        """
        Purpose: Calc the contact area for bearing capacity calculations
        """
        # Store the area_type
        self.pffp_config["area_type"] = area_type 
        
        # Temp storage of the df
        tip_props = self.pffp_config["tip_props"]
        
        # Add cases here so that the correct values can be passed 
        # Unpack some values
        displacement = self.impulse_df["displacement"]
        tip_type = self.pffp_config["tip_type"]

        # Calc the contact area
        contact_area = calc_pffp_contact_area(penetrationDepth=displacement, areaCalcType= area_type, tipType= tip_type, tipProps= tip_props, 
                                              tip_val_col= self.pffp_config["tip_col_name"])

        col_name = "{}_{}".format("contact_area", area_type)

        # Store the contact area
        if self.bearing_df is None:
            self.bearing_df = pd.DataFrame(data = {col_name:contact_area}) 
        else:
            self.bearing_df[col_name] = contact_area

    # Plotting functions
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
        df = self.release_df
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

        plt.plot(self.release_df["Time"].loc[start:end], self.release_df["accel"].loc[start:end], color = "blue", label= "release")
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

if __name__ == "__main__":
    # Add some testing here
    pass