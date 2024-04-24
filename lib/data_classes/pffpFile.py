# import datetime
import numpy as np
from lib.data_classes.BinaryFile import BinaryFile
from lib.signal_processing.signal_function import find_drops, moving_average
from lib.data_classes.dropClass import Drop # Class that is used to represent drops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

class pffpFile(BinaryFile):
    # Purpose: To store information and modify a PFFP binary file

    # TODO: Add a time zone category and add that data to the datetime
    def __init__(self, file_dir, calibration_params, sensor_units = ["g", "kPa", "min"]):
        BinaryFile.__init__(self, file_dir)

        # Store the calibration params fro converting from volts to engineering units
        self.calibration_params = calibration_params
        
        # Get the file name
        self.file_name = self.get_file_name()

        # Get the file drop date
        self.get_file_datetime()

        # Init variables for later storage
        self.num_drops = "Not Checked" 
        
        self.df_stored = False

        self.sensor_units = sensor_units

    def __str__(self):
        # Purpose: Return some information about the file
        return f"File Directory: {self.file_dir} \nNum Drops in file: {self.num_drops} \
        \nDrop Date: {self.datetime.date()} \nDrop Time: {self.datetime.time()} \
        \ndf stored: {self.df_stored}"

    @staticmethod
    def convert_acceleration_unit(df, columns, input_unit, output_unit):
        # Purpose: Convert the acceleration unit from one  to another

        if input_unit  != "g":
            raise IndexError("Conversions starting from units other than g's not implemented")
        
        # Assumes convertion from g's
        acceleration_conversion_dict = {
                                        "m/s^2": 9.80665,
                                        "g": 1
        }

        df[columns] = df[columns] * acceleration_conversion_dict[output_unit]

        return df
    
    def binary_2_sensor_df(self, size_byte = 3, byte_order = "big", signed = True, acceleration_unit = "g", pressure_unit = "kPa", time_unit = "s"):
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

        for value in params['Sensor']:
            offset = params[params['Sensor'] == value].iloc[0, 1]
            scale = params[params['Sensor'] == value].iloc[0, 2]

            # Scale the df columns, Formula is (sensor_value + offset)/scale
            df[value] = (df[value] +offset)/scale

        # Get the number of entries in a column
        num_entries = len(df.iloc[:, 0])
        
        # Get the time steps
        time = np.linspace(0, 1, num_entries)
        
        # Store the time in seconds in the df
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
    
    def analyze_file(self, use_pore_pressure = True, store_df = True, overide_water_drop = False):
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
        df = self.binary_2_sensor_df(acceleration_unit = self.sensor_units[0], pressure_unit = self.sensor_units[1], time_unit= self.sensor_units[2])

        # Init flag to track if the drop was checked using pressure sensor
        pressure_check_list = []

        # if use_pore_pressure and np.max(df["pore_pressure"] > 1):
        if use_pore_pressure and np.max(df["pore_pressure"] > 0.5):
            # Init list to store accel indices that need to be deleted
            index_2_delete = []

            # Get the max deceleration and use 75% of that as the cutoff
            min_height = min(0.75 * np.max(df["18g_accel"]), 2.8)

            #TODO: For the time being just check the 18g sensor in the future multiple sens
            peak_indexs, peak_info, num_accel_drops = find_drops(df["18g_accel"], min_peak_height=min_height, impact_time_tol = 0.03)

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
                print("Nim pressure drops: ", num_pressure_drops)
                print(f"Warning: Number of predicted drops should match {self.file_name}")
                self.funky = True
            pressure_check_list.append(True)
            
        else:
            #TODO: For the time being just check the 18g sensor in the future multiple sens
            peak_indexs, peak_info, num_accel_drops = find_drops(df["18g_accel"])

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
            df = self.binary_2_sensor_df(acceleration_unit = self.sensor_units[0], pressure_unit = self.sensor_units[1], time_unit= self.sensor_units[2])
        
        # display(df.head())
        # loop over the drops in the file
        accel = df["18g_accel"]
        time = df["Time"]

        for drop in self.drops:
            # For the time being store the release and impulse data for all drops
            drop.store_whole_drop = True
            
            # Trim the acceleration data
            drop.cut_accel_data(accel, time, input_units = {"accel":"g", "Time":"s"} )

            # Integrate the drop (impulse and the release part because store_whole_drop=True)
            drop.integrate_accel_data()
         
        
    def check_drop_in_file(self):
        # Purpose: check if there's a drop in the file
        if self.num_drops > 0:
            return True
        else:
            return False
        
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
            fig.update_xaxes(title_text="Time (s)", row=3, col=1)

            # Update yaxis properties
            fig.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
            fig.update_yaxes(title_text="Pressure (kPa)", row=2, col=1)
            fig.update_yaxes(title_text="Acceleration (g)", row=3, col=1)

            # Update figure title
            fig.update_layout(height = figsize[1] *100, width = figsize[0] *100,
                            title_text=f"File Name: {self.file_name}")

            # Turn off interactivity
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
            axs[0].set_ylabel(f"Acceleration [{self.sensor_units[0]}]")
            axs[1].set_ylabel(f"Pressure [{self.sensor_units[1]}]")
            axs[2].set_ylabel(f"Acceleration [{self.sensor_units[0]}]")

            # Label the x-axis
            axs[0].set_xlabel(f"Time [{self.sensor_units[2]}]")
            axs[1].set_xlabel(f"Time [{self.sensor_units[2]}]")
            axs[2].set_xlabel(f"Time [{self.sensor_units[2]}]")

            # Give the entire figure a label
            fig.suptitle(f"File Name: {self.file_name}")

            plt.tight_layout()
            plt.show()


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