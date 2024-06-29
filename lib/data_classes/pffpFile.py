# import datetime
import numpy as np
from lib.data_classes.BinaryFile import BinaryFile
from lib.signal_processing.signal_function import find_drops, moving_average
from lib.data_classes.dropClass import Drop # Class that is used to represent drops
from lib.general_functions.helper_functions import convert_accel_units, convert_time_units, convert_length_units
from lib.data_classes.exceptions import zeroLenError

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from operator import itemgetter
import time

class pffpFile(BinaryFile):
    # Purpose: To store information and modify a PFFP binary file

    # TODO: Add a time zone category and add that data to the datetime
    def __init__(self, file_dir, calibration_params, sensor_units = {"accel":"g", "pressure":"kPa", "Time":"min"}):
        BinaryFile.__init__(self, file_dir)

        # Store the calibration params fro converting from volts to engineering units
        self.calibration_params = calibration_params
        
        # Get the file name
        self.get_file_name()

        # Get the file drop date
        self.get_file_datetime()

        # Init variables for later storage
        self.num_drops = "Not Checked" 
        
        self.df_stored = False
        self.stored_concat_accel = False
        self.sensor_units = sensor_units

    def __str__(self):
        # Purpose: Return some information about the file
        return f"File Directory: {self.file_dir} \nNum Drops in file: {self.num_drops} \
        \nDrop Date: {self.datetime.date()} \nDrop Time: {self.datetime.time()} \
        \ndf stored: {self.df_stored}\
        \nConcat accel stored: {self.stored_concat_accel}"

    @staticmethod
    def convert_acceleration_unit(df, columns, input_unit, output_unit):
        # Purpose: Convert the acceleration unit from one  to another
        # TODO: Generalize this so that it follows the same format as the other converters
        # NOTE: Could potentiallly make a general function that just understands units

        if input_unit  != "g":
            raise IndexError("Conversions starting from units other than g's not implemented")
        
        # Assumes convertion from g's
        acceleration_conversion_dict = {
                                        "m/s^2": 9.80665,
                                        "g": 1
        }

        df[columns] = df[columns] * acceleration_conversion_dict[output_unit]

        return df
        
    @staticmethod
    def get_tightest_sensor(row, resolutions, labels):
        value = row.max()

        # Filter out resolutions smaller than the value
        valid_resolutions = [res for res in resolutions if res >= value]

        # If all resolutions are greater than the value, use the smallest resolution
        if not valid_resolutions:
            closest_resolution = min(resolutions)
        else:
            closest_resolution = min(valid_resolutions, key=lambda x: abs(x - value))
        
        # Get the index of the closest resolution
        index = resolutions.index(closest_resolution)

        # Get the column the closest data is in
        col = labels[index] 
            
        return row[col]
    
    def new_pffp_file_name(self, survey_id, num_char_from_end = 8, use_default = True, other_name = None):
        """
        Purpose: Update the name of a pffp file
        """

        # TODO: Need to make sure this works
        # If the default
        if use_default:
            # Get the base name which is the hex number and the file extension
            base_name = self.file_name[-num_char_from_end:]

            new_pffp_name = "{}_{}".format(survey_id, base_name)
            
            # Rename the file and update the properties of the file class to track that info
            self.new_file_name(new_name=new_pffp_name)
        
        # In case a name other than the default format is wanted
        elif not other_name is None:
            raise ValueError("Using a name other than the default is not implemented at this time")

        # Rename the containing file name for the drops
        for drop in self.drops:
            drop.containing_file = self.file_name

    def binary_2_sensor_df(self, size_byte = 3, byte_order = "big", signed = True, acceleration_unit = "g", pressure_unit = "kPa", time_unit = "min"):
        # Purpose read a binary file and transform it to a processed df
        column_names = ["Count", 'Unknown', "2g_accel", "18g_accel", "50g_accel", "pore_pressure", "200g_accel", 
                                                                "55g_x_tilt","55g_y_tilt", "250g_accel"]
        num_columns = len(column_names)

        num_rows = -1

        df = self.binary_file_2_df(column_names, num_columns, num_rows, size_byte, byte_order, signed)

        # Apply the offset to the df
        
        # NOTE: Assumes the columns of the df are IN EXACT ORDER (names can be differnt but values must be the same):
        # [Count, Unknown, 2g_accel, 18g_accel, 50g_accel, pore_pressure, 200g_accel, 55g_x_tilt, 55g_y_tilt, 250g_accel]

        # Drop the count and unknown columns
        columns_2_drop = [0, 1]
        
        columns_header_2_drop = df.columns[columns_2_drop]
        
        # df now has values:
        # [2g_accel, 18g_accel, 50g_accel, pore_pressure, 200g_accel, 55g_x_tilt, 55g_y_tilt, 250g_accel]
        df.drop(columns_header_2_drop, axis = 1, inplace = True)

        # Local variable for the calibration factors
        params = self.calibration_params

        # Search through the senor names and get the offset and scale for that sensor
        for value in params['Sensor']:
            offset = params[params['Sensor'] == value].iloc[0, 1]
            scale = params[params['Sensor'] == value].iloc[0, 2]

            # Scale the df columns, Formula is (sensor_value + offset)/scale
            df[value] = (df[value] +offset)/scale

        # Get the number of entries in a column
        num_entries = len(df.iloc[:, 0])
        
        # Get the time steps and convert the time to minutes
        time = np.linspace(0, 1, num_entries)
        
        # Store the time in minutes in the df
        df.insert(0, "Time", time)

        # Acceleration scaling
        acceleration_sensor_names = ["2g_accel", "18g_accel", "50g_accel","200g_accel", "55g_x_tilt","55g_y_tilt", "250g_accel"]
        
        df = pffpFile.convert_acceleration_unit(df, acceleration_sensor_names, "g", acceleration_unit)
        
        # Dictionary with conversion from psi to other sets of units
        pressure_conversion_dict = {
                                    "kPa": 6.89476
        }

        if pressure_unit == "psi":
            pass
        elif pressure_unit == "kPa":
            # Convert the pressure to kPa
            df["pore_pressure"] = df["pore_pressure"] * pressure_conversion_dict["kPa"]
        else:
            raise ValueError("Onnly converison from psi to kPa implemented")
        return df
    
    def analyze_file(self, use_pore_pressure = True, store_df = True, overide_water_drop = False, 
                     select_accel = ["2g_accel", "18g_accel", "50g_accel", "250g_accel"]):
        # Purpose: Analyzes a bin file, gets
        # number of drops in the file
        # peak location of the drops
        
        # Set the flag that keeps track if the df is stored

        # TODO: Add inputs for the find drops, this will be the tolerances and the info about the sampling rate

        # Init list to store the drop objects
        self.drops = []
         
        self.funky = False
        
        # list to track if there's a problem with processing
        check_status = [0, 0]

        # Load the file df - Might be able to only load the part of the files I want in the future
        # Skip some sections of the binary file and only load the sensor that I need
        # Need to convert to gravity units because that's what the find_drops is expecting
        df = self.binary_2_sensor_df(acceleration_unit = self.sensor_units["accel"], pressure_unit = self.sensor_units["pressure"], time_unit= self.sensor_units["Time"])

        # Stitch the accelerometers and store the values
        # Don't use the 200g accel it's kind of sketchy sometimes
        self.stitch_accelerometers(df, accel_labels = select_accel)

        # Init flag to track if the drop was checked using pressure sensor
        pressure_check_list = []

        # if use_pore_pressure and np.max(df["pore_pressure"] > 1):
        if use_pore_pressure and np.max(df["pore_pressure"] > 0.5):
            # Init list to store accel indices that need to be deleted
            index_2_delete = []

            # Get the max deceleration so that a guess of the selected peak height can be choosen
            max_accel = np.max(self.concat_accel)
            tol_5g = 6.0

            # Min height that's considered a drop
            min_2g = 2

            if max_accel < tol_5g:
                percentage_tol = 0.6
                min_height = max(percentage_tol * max_accel, min_2g)
            else:
                # For the drops that have a higher max acceleration use a lower proportion of max acceleration
                percentage_tol = 0.4
                min_height = max(percentage_tol * max_accel, min_2g)
                
            #TODO: For the time being just check the 18g sensor in the future multiple sens
            peak_indexs, peak_info, num_accel_drops = find_drops(self.concat_accel, min_peak_height=min_height, impact_time_tol = 0.03)

            # Select the times where the peak acceleration thinks drops occured
            time_peak_acceleration = np.array(df["Time"])[peak_indexs]
                
            # Get where the pressure derivative thinks the drops are
            time_peak_pressure_deriv, num_pressure_drops = self.check_pore_pressure_4_drop(df, window = 2500)

            # Loop over the acceleration times and see if there's a pressure sensor time close by
            # I have more faith in the pressure sensor at detecting drops 
            for i, accel_time in enumerate(time_peak_acceleration):

                if len(time_peak_pressure_deriv) > 0:
                    time_diff = np.abs(time_peak_pressure_deriv - accel_time)
                    # print(time_peak_pressure_deriv)
                    matching_index = np.where(time_diff<0.02)[0]
                else:
                    # Catch the case where no pore pressures are returned
                    matching_index = []

                # print(matching_index)
                if len(matching_index) == 0:
                    # If there are no matching time indexs remove that value from the accleration peak_indexs
                    index_2_delete.append(i)
                    num_accel_drops -= 1

                elif len(matching_index) > 1:
                    print(f"Warning: Multiple acceleration evaluated to the same pore pressure drop predictor {self.file_name}")
                    self.funky = True
            # Delete the indices
            peak_indexs = np.delete(peak_indexs, index_2_delete)

            if num_accel_drops != num_pressure_drops:
                print("\nNum accel drops:", num_accel_drops)
                print("Num pressure drops: ", num_pressure_drops)
                print(f"Warning: Number of predicted drops should match {self.file_name}")
                self.funky = True
            pressure_check_list.append(True)
            
        else:
            #TODO: For the time being just check the 18g sensor in the future multiple sens
            peak_indexs, peak_info, num_accel_drops = find_drops(self.concat_accel)

            # Select the times where the peak acceleration thinks drops occured
            time_peak_acceleration = np.array(df["Time"])[peak_indexs]

            pressure_check_list.append(False)

        self.num_drops = num_accel_drops

        for i, index in enumerate(peak_indexs):
            drop = Drop(containing_file= self.file_name, peak_index = index, file_drop_index= i+1, peak_info = peak_info, pressure_check = pressure_check_list)
                    
            # Add the drop to the list
            self.drops.append(drop)

        # Check if the df should be stored
        if  store_df:
            # Remove the df reference to allow the garbage collector to free up the memory
            self.df = df

            # Set the tracker flag to false
            self.df_stored = True
    
    def process_drops(self):
        # Purpose: Analyze the drops in the files

        # Check if the df is stored
        if self.df_stored:
            # Temporarily store the df
            df = self.df
        else: 
            # Load the df
            df = self.binary_2_sensor_df(acceleration_unit = self.sensor_units["accel"], pressure_unit = self.sensor_units["pressure"], time_unit= self.sensor_units["Time"])
        
        # display(df.head())
        # loop over the drops in the file
        accel = self.concat_accel
        time = df["Time"]

        raise_error = False
        
        for drop in self.drops:  
            skip_integration = False
            # Try to trim the acceleration data and integrate it
            # Do the try here than pass the error back up
            try:
                # Trim the acceleration data
                drop.cut_accel_data(accel, time, input_units = {"accel":"g", "Time":"min"} )

            except zeroLenError as err:
                # Print the error message
                details= err.args[0]

                # Print the error details
                print("\n" + details["message"])
                print("Criteria not met:", details["criteria"])
                print(details["source"])
                print("Moving file to funky folder")

                # set error flag but try to process other drops
                raise_error = True
                skip_integration = True

            if not skip_integration:
                # If no error caught for this drop do the integration
                drop.integrate_accel_data()

                # Set the flag 
                drop.processed = True 

        # Raise error so the file gets moved into the funky folder
        if raise_error:
            raise zeroLenError

    def manually_process_drops(self, interactive_plot = True, figsize = [6,4]):
        """
        Purpose: Wrapper for the manual indices selection and the manual integration 
                 Allows manual drops in this file to be processed
        """
        # Loop over the drops in the file
        for drop in self.drops: 
            # If the drop isn't processed
            if not drop.processed:
                # Do the manual indices selction
                self.manual_indices_selection(drop, interactive = interactive_plot, figsize = figsize)

                # And do the integration
                self.manually_integrate_drop(drop)

    def manually_integrate_drop(self, drop):
        # Purpose: Setup and integrate the drops that were manually processed

        # Check if the df is stored
        if self.df_stored:
            # Temporarily store the df
            df = self.df
        else: 
            # Load the df
            df = self.binary_2_sensor_df(acceleration_unit = self.sensor_units["accel"], pressure_unit = self.sensor_units["pressure"], time_unit= self.sensor_units["Time"])
        
        # display(df.head())
        # loop over the drops in the file
        accel = self.concat_accel
        time = df["Time"]

        if not drop.indices_found:
            raise IndexError("Indices aren't all found yet")
        
        if not drop.manually_processed:
            raise ValueError("Only use this function for drops that had there indices manually selected")
        
        drop.cut_accel_data(accel, time, input_units = {"accel":"g", "Time":"min"} )
        drop.integrate_accel_data()

        # Set the flag 
        drop.processed = True 
        
    def check_drop_in_file(self):
        # Purpose: check if there's a drop in the file
        if self.num_drops > 0:
            return True
        else:
            return False

    def check_pore_pressure_4_drop(self, df, window = 2500):
        # Purpose: use pore pressure to check if there's a drop in the file
        
        # Store the df in a temp variable
        time = np.array(df["Time"])
        pressure = np.array(df["pore_pressure"])

        # Smooth the pressure and time - time smoothing is done so both arrays have the same num values
        smoothed_pressure = moving_average(pressure, window)
        smoothed_time = moving_average(time, window)

        # Calc the pressure derivative
        pressure_deriv = np.gradient(smoothed_pressure, smoothed_time)

        # Get the min peak height to consider for the derivative 10 * the max pressure has worked well
        # TODO: See if there's a good argument for why 10 seems to work well or if it's just chance
        max_pressure_scaling = 12
        min_height = np.max(smoothed_pressure) * max_pressure_scaling


        pressure_indexs, pressure_peak_info, num_drops = find_drops(pressure_deriv, impact_time_tol=0.08, min_peak_height=min_height )

        # Get the time that the pressure gradient peaks occur
        time_peak_pressure_deriv = smoothed_time[pressure_indexs]

        return time_peak_pressure_deriv, num_drops
    
    def stitch_accelerometers(self, accel_df, accel_labels):
        # Purpose: Stich acclerometers together to get the most resolved accelerations
        res_dict = {"2g_accel":1.7, "18g_accel": 18, "50g_accel":50, "200g_accel":200, "250g_accel":250}
        
        resolutions = itemgetter(*accel_labels)(res_dict)
        df = accel_df[accel_labels]

        # NOTE: Assumes that the accel labels and the max sensor values are in the same order
        series = df.apply(self.get_tightest_sensor, axis = 1, args = (resolutions,accel_labels,))
        self.concat_accel = series
        self.stored_concat_accel = True
    
    def manual_indices_selection(self, drop, debug = False, interactive = True, figsize = [12,8], legend = False, lag = 0.1):
        # Purpose: Manually select indices for drops
        
        # Print information about the drop
        print(drop)

        print("\nPlotting file data...")
        # Plot the data
        self.quick_view(interactive=interactive, figsize= figsize, legend = legend)
        
        time.sleep(lag)                            

        allowed_input_type = ["Time", "index", "skip"]
        index_dict = drop.drop_indices.copy()      
        
        for key, value in index_dict.items():
            if not value is None:
                continue
        
            print("--------- Need {} ---------".format(key))
            # Get the type of the input allow users to enter a time or a index
            input_type_string ="Enter {} or {} or {}\n".format(allowed_input_type[0],
                                                                allowed_input_type[1],
                                                                allowed_input_type[2])
            
            input_type = input(input_type_string)
            
            if input_type == "":
                print("Escaping from manual_indices_selection without completion")
                return
            
            while not input_type in allowed_input_type:
                input_type = input(input_type_string)
                print("{} is not an allowed input. Only {}, {} or {} is allowed\n".format(input_type, allowed_input_type[0], allowed_input_type[1], allowed_input_type[2]))
                if input_type == "":
                    print("Escaping from function")
                    return None
            
            if input_type == allowed_input_type[0]:
                
                # Get the limiting time by converting the max time to the file units so it matches the figure
                limit = convert_time_units(drop.time.max(), drop.units["Time"], self.sensor_units["Time"] ) 
                
                if debug:
                    print("time limit", limit)
                
                # Flag to track if the input is good
                good_input = False
                
                # Check that the val is a float
                while not good_input:
                    print("\nInput must be a decimal number and less than {}\
                        \nFunction assumes the time is in {}".format(limit, self.sensor_units["Time"]))
                    
                    # Get the string input and convert it to a float
                    try:
                        input_val = input("Enter a time:\n")
                        
                        if input_val == "":
                            print("Escaping from manual_indices_selection without completion")
                            return None
                        
                        val = float(input_val)
                    except ValueError as err:
                        print(err)
                        good_input = False
                    
                        # Go to the next iteration because input wasn't valid
                        continue
                    
                    # Check that the entered time is within bounds
                    if val >= 0 and val < limit:
                        good_input = True
                    
                # Convert the units back to the drop time units
                val = convert_time_units(val, self.sensor_units["Time"], drop.units["Time"])
                
                # Find the index that corresponds to the time closest to the input time
                index = np.where(val <= drop.time)[0][0]
            elif input_type == allowed_input_type[1]:
                index =0.0
                good_input= False
                while not good_input:
                    print("\nInput must be an integer number (no decimal points) and less than {} and greater than or equal to 0".format(0, len(drop.time)-1))
                    try:
                        # Try to convert the input to an integer
                        index = input("Enter an integer index")
                        if index == "":
                            print("Escaping from manual_indices_selection without completion")
                            return None 
                        index = int(index)
                    except ValueError as err:
                        print(err)
                        good_input = False
                        
                        continue
                    # Check that the integer is positive and inside the bounds of the arr
                    if index <=len(drop.time)-1 and index >=0:
                        good_input = True
            
            elif input_type == allowed_input_type[2]:
                # If the file can be left as move to the next drop
                continue
            
            # Checking if 
            integration_type = "None"
            integration_ans = ["y", "n"]
            while not integration_type in integration_ans:
                integration_type = input("Impulse Integration? (y or n)")
                
                if integration_type == "":
                    print("Escaping from the manual_indices_selection without completion" )
                    return
                
                if integration_type == integration_ans[0]:
                    drop.only_impulse = True
                elif integration_type == integration_ans[1]:
                    drop.only_impulse = False
                
            # set the value into the dict
            drop.drop_indices[key] = index

        drop.indices_found = True
        drop.manually_processed = True

    # Plotting functions
    def quick_view(self, interactive = False, figsize = [12, 8], legend = False):
        # Purpose: Get a quick view of the file

        # if the df isn't stored load it for plotting don't store it though
        if not self.df_stored:
            df = self.binary_2_sensor_df(acceleration_unit="g", pressure_unit= "kPa")
        else:
            # Otherwise use the stored df
            df = self.df

        # generate the figure object
        time = df["Time"]
        
        accel_labels = ["2g_accel", "18g_accel", "50g_accel", "200g_accel", "250g_accel"]
        tilt_labels = ["55g_x_tilt", "55g_y_tilt"]
        pressure_label = "pore_pressure"

        time_unit = self.sensor_units["Time"]
        accel_unit = self.sensor_units["accel"]
        pressure_unit = self.sensor_units["pressure"]
        
        if interactive:
            fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)

            # Accelerometer plots
            for label in accel_labels:
                fig.add_trace(
                    go.Scatter(x = time, y = df[label], mode = "lines", name = label),
                    row = 1, col = 1
                )

            # Pore pressure sensor
            fig.add_trace(
                go.Scatter(x = time, y = df[pressure_label], mode = "lines", name = "pore pressure"),
                row = 2, col = 1
            )
            
            # Tilt sensors
            for label in tilt_labels:
                fig.add_trace(
                    go.Scatter(x = time, y = df[label], mode = "lines", name = label),
                    row = 3, col = 1
                )

            # Update xaxis properties
            fig.update_xaxes(title_text="Time {time_unit}", row=3, col=1)

            # Update yaxis properties
            fig.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
            fig.update_yaxes(title_text="Pressure (kPa)", row=2, col=1)
            fig.update_yaxes(title_text="Acceleration (g)", row=3, col=1)

            # Update figure title
            fig.update_layout(height = figsize[1] *100, width = figsize[0] *100,
                            title_text=f"File Name: {self.file_name}")

            # Show the plot interactivity
            fig.show()
        else:
            # Use matplotlib
            fig, axs = plt.subplots(ncols = 1, nrows = 3, figsize = (figsize[0], figsize[1]))

            # Accelerometers
            for label in accel_labels:
                axs[0].plot(time, df[label], label = label)
            
            # Pressure sensor
            axs[1].plot(time, df[pressure_label], label = pressure_label)

            # Tilt sensors
            for label in tilt_labels:
                axs[2].plot(time, df[label], label = label)
            
            # Turn on the legends
            if legend:
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

            # Label the y-axis
            axs[0].set_ylabel(f"Acceleration [{accel_unit}]")
            axs[1].set_ylabel(f"Pressure [{pressure_unit}]")
            axs[2].set_ylabel(f"Acceleration [{accel_unit}]")

            # Label the x-axis
            axs[0].set_xlabel(f"Time [{time_unit}]")
            axs[1].set_xlabel(f"Time [{time_unit}]")
            axs[2].set_xlabel(f"Time [{time_unit}]")

            # Give the entire figure a label
            fig.suptitle(f"File Name: {self.file_name}")

            plt.tight_layout()
            plt.show()

    def plot_drop_impulses(self, figsize = [4,6], save_figs = False, hold = False, legend = True,
                            colors = ["black", "blue", "green", "orange", "purple", "brown"],
                            units = {"Time":"s", "accel":"g", "velocity":"m/s", "displacement":"cm"},
                            line_style = ["solid", "dashed"]):        
        # Purpose: Plot the standard velocity and acceleration vs. displacement plots for all the drops
        # TODO: Move this to the drop level and call the drop function here
        # Set all of the line colors to black if hold is on
        if not hold:
            colors = ["black"] * self.num_drops
            
        # Loop over the drops and plot them

        first_processed_drop = -1
        for i, drop in enumerate(self.drops):
            
            # If the processing for the drop is not done
            if not drop.processed:
                # Print the drop information
                print(drop, "not finished being processed")

                # Go to the next drop
                continue
            
            # Increment the tracker for plotting
            first_processed_drop+=1

            if not hold:
                # Make a new figure every time
                fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (figsize[0], figsize[1]))
            elif hold and first_processed_drop ==0:
                fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (figsize[0], figsize[1]))

            drop_units = drop.units

            time = drop.impulse_df["Time"]
            accel = drop.impulse_df["accel"]
            vel = drop.impulse_df["velocity"]
            disp = drop.impulse_df["displacement"]

            # Convert the units
            time = convert_time_units(time, drop_units["Time"], units["Time"])
            accel = convert_accel_units(accel, drop_units["accel"], units["accel"])
            disp = convert_length_units(disp, drop_units["displacement"], units["displacement"])

            if drop_units["velocity"] != units["velocity"]:
                raise ValueError("Conversion for velocity not implemented")
            
            axs.plot(accel, disp, color = colors[i], label= "acceleration", linestyle = line_style[0])
            axs.plot(vel, disp, color = colors[i], label = "velocity", linestyle = line_style[1])

            # label the axes
            if hold and first_processed_drop == 0 or not hold:
                accel_unit = units["accel"]
                vel_unit = units["velocity"]
                disp_unit = units["displacement"]

                axs.set_xlabel(f"Acceleration ({accel_unit})/Velocity ({vel_unit})")
                axs.set_ylabel(f"Displacement ({disp_unit})")
                
                # Label the plot only with the file name if hold is on
                if hold:
                    axs.set_title(f"File: {self.file_name} Num Drops: {self.num_drops}")
                else: 
                    axs.set_title(f"File: {self.file_name} Drop id: {drop.file_drop_index}")
                axs.invert_yaxis()

            if legend:
                axs.legend()

if __name__ == "__main__":
    # Add some testing here
    pass