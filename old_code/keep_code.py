import numpy as np
import matplotlib.pyplot as plt

def transform_to_frequency(signal, sampling_rate):
    """
    Transform time-domain signal to frequency-domain using FFT.

    Parameters:
    signal (array-like): Input time-domain signal.
    sampling_rate (float): Sampling rate of the signal.

    Returns:
    freqs (array-like): Frequencies corresponding to the FFT result.
    magnitude (array-like): Magnitude of the FFT result.
    """
    # Compute the FFT
    fft_result = np.fft.fft(signal)
    
    # Compute the frequencies corresponding to the FFT result
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    
    # Take the magnitude of the FFT result (use np.abs for complex numbers)
    magnitude = np.abs(fft_result)
    
    return freqs, magnitude

freqs, magnitude = transform_to_frequency(df[selected_accelerometer], len(df["Time"]))

plt.figure()
plt.plot(freqs, magnitude, color = "red")

# Select the extent of the figure
# plt.xlim([-2, 2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain Representation')
plt.grid(True)
plt.show()

# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["2g_accel"], mode = "lines", name = "2g_accel"),
#     row = 1, col = 1
# )

# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["18g_accel"], mode = "lines", name = "18g_accel"),
#     row = 1, col = 1
# )
# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["50g_accel"], mode = "lines", name = "50g_accel"),
#     row = 1, col = 1
# )
# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["200g_accel"], mode = "lines", name = "200g_accel"),
#     row = 1, col = 1
# )
# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["250g_accel"], mode = "lines", name = "250g_accel"),
#     row = 1, col = 1
# )
# fig.add_trace(
#     go.Scatter(x = df["Time"], y = df["pore_pressure"], mode = "lines", name = "pore_pressure"),
#     row = 2, col = 1
# )

# def trim_around_drop(drop_df, window = 0.15):
#     # Purpose: Trim the data around the drop
#     # window: Expected window around the peak of the drop that the entire drop should exist in

#     pass
# def find_start_end_drop(drop_df, peak):
#     # Purpose: Select the start and end of a drop

#     pass


# def find_drops(pffp_df, time_tolerance = 200):
#     # Purpose: Find the peak of a drop, the corresponding index, and number of drops in a bianry file

#     # pffp_df: Dataframe of all pffp sensor data
#     # Index tolerance: The check for drops finds all the values that are greater than a threshold, therefore, it will return multiple values for
#     # Check the 18g sensor for drops
#     # First array value is if there's a drop in the file
#     info = check_peak_in_data(pffp_df["18g_accel"])
    
#     drop_in_file = info[0]
#     if drop_in_file:
#         # Unpack the other values
#         peak_indices = info[1]
        
#         # Get the dict info
#         dict_info = info[2]
        
#         # Get a rough estimate of the extent of the drop (Might just hardcode this)
#         return peak_indices, dict_info
#     else:
#         num_drops_in_file = 0
#         return 0
    
# peak_indices, dict_info = find_drops(df)

# print(peak_indices, "\n", dict_info)

# 1/120_000

# peak_indices[0] - peak_indices[-1]

index = 6

df =test_folder.pffp_drop_files[index].df

time = np.array(df["Time"])
pore_pressure = np.array(df["pore_pressure"])

window = 2500

smoothed_pore_pressure = moving_average(pore_pressure, window)
smoothed_time = moving_average(time, window)
pore_pressure_deriv = np.gradient(smoothed_pore_pressure, smoothed_time)
print(test_folder.pffp_drop_files[index])

########

peak_indexs, peak_info, num_drops = find_drops(df["18g_accel"])

pressure_indexs, pressure_peak_info, num_drops = find_drops(pore_pressure_deriv, impact_time_tol=0.08, min_peak_height=np.max(smoothed_pore_pressure)*12)

print(peak_indexs, peak_info)

print("acceleration peak time: ", time[peak_indexs])
print("pressure peak time", smoothed_time[pressure_indexs])

###
fig, axs = plt.subplots(ncols = 1, nrows = 2)

axs[0].scatter(smoothed_time[pressure_indexs], smoothed_pore_pressure[pressure_indexs], label = "accel peak", color = "orange", s= 100)
# axs[0].plot(time, pore_pressure, label = "pore_pressure")

scaling = np.max(smoothed_pore_pressure)/np.max(df["18g_accel"])
axs[0].plot(time, pore_pressure, label = "smoothed pressure", color = "red" )
axs[0].plot(time, df["18g_accel"], color = "orange")
# To plot them on the same scale
scaling = np.max(pore_pressure_deriv)/np.max(pore_pressure)
axs[1].plot(smoothed_time, pore_pressure_deriv, label = "pore pressure deriv")

axs[1].axhline(y = np.max(smoothed_pore_pressure)*12, color = "red")
plt.legend()

# function to find the first time a signal reaches a point after a certain index

def find_signal_value(signal, value, start_index, end_index = None, num_occurrences = 1, tolerance = 1e-5):
    # Purpose: Find where a signal equals a value after a certain starting index within a tolerance

    # select the part of the signal of interest
    selected_signal = signal[start_index:end_index]

    # Subtract off the value
    arr = selected_signal - value

    # Find the location where the difference is less than the tolerance
    indices = np.where((-tolerance < arr) & (arr < tolerance))[0]
    print(start_index + indices)
    # Select the number of occurences wanted
    return start_index + indices[:num_occurrences]

arr = np.linspace(0,1, 3)
    
# Find where the signal goes back to a certain value
# start_drop_index = find_signal_value(averaged_accel, value = 1.2, start_index = drop_index[0] - index_window, end_index = drop_index[0], num_occurrences=1, tolerance = 1e-1)
# end_drop_index = find_signal_value(averaged_accel, value = 0.99, start_index = drop_index[0], num_occurrences=1, tolerance = 1e-1)

start_drop_index = find_signal_value(accel, value = 1.2, start_index = drop_index[0] - index_window, end_index = drop_index[0], num_occurrences=1, tolerance = 1e-1)
end_drop_index = find_signal_value(accel, value = 0.99, start_index = drop_index[0], num_occurrences=1, tolerance = 1e-1)

print(f"start index: {start_drop_index}")
print(f"end index: {end_drop_index}")

start = drop_index[0] - index_window
end = drop_index[0]

plt.plot(df_time, accel)
plt.scatter(df_time[start_drop_index], accel[start_drop_index], marker = ".", color = "red")
plt.scatter(df_time[end_drop_index], accel[end_drop_index], marker = ".", color = "red", s = 10)
# plt.plot(time[start:end], averaged_accel[start:end])



fig = make_subplots(rows = 2, cols = 1)

# PLot the smoothed acceleration
fig.add_trace(
    go.Scatter(x = df_time, y = accel, mode = "lines", name = "intial"),
    row = 1, col = 1
)

# PLot the smoothed acceleration
fig.add_trace(
    go.Scatter(x = time, y = averaged_accel, mode = "lines", name = "smoothed"),
    row = 1, col = 1
)

# Plot the location where the derivative meets the criteria
fig.add_trace(
    go.Scatter(x = df_time[deriv_changes_index], y = accel[deriv_changes_index], mode = "markers", name = "deriv_changes"),
    row = 1, col = 1
)

# Plot the location of the peak 
fig.add_trace(
    go.Scatter(x = df_time[drop_index], y = accel[drop_index], mode = "markers", name = "drop peak"),
    row = 1, col = 1
)

# Plot the beginning of the drop
fig.add_trace(
    go.Scatter(x = df_time[start_drop_index], y = accel[start_drop_index], mode = "markers", name = "begin drop"),
    row = 1, col = 1
)

# Plot the predicted end of drop
fig.add_trace(
    go.Scatter(x = df_time[end_drop_index], y = accel[end_drop_index], mode = "markers", name = "drop end"),
    row = 1, col = 1
)

# Plot the derivative of the acceleration
fig.add_trace(
    go.Scatter(x = time[drop_index[0] - index_window: drop_index[0] + index_window], y = acceleration_derivative, mode = "lines", name = "derivative"),
    row = 2, col = 1
)

fig.show()

# For each drop index select the part of the 
frequency = 120_000
time_window = 0.01
index_window = int(time_window * frequency)
print(index_window)
# index_arr = np.array([drop_index[0] - index_window: drop_index[0] + index_window])
# print(index_arr)
print(len(time))

# Calc the gradient of the plot

deriv_changes_index = find_deriv_change(df_time[drop_index[0] - index_window: drop_index[0] + index_window], averaged_accel[drop_index[0] - index_window: drop_index[0] + index_window], 300)[0] + drop_index[0]

# Calc the derivative of the index window
acceleration_derivative = np.gradient(averaged_accel[drop_index[0] - index_window: drop_index[0] + index_window], df_time[drop_index[0] - index_window: drop_index[0] + index_window])
plt.plot(time[drop_index[0] - index_window: drop_index[0] + index_window], acceleration_derivative)

print(test_folder.pffp_drop_files[index])
test_folder.pffp_drop_files[index].quick_view(interactive = False)


# Code that does a statistical analysis
# Specify the change point detection method
model = "l1"  # You can choose other models as well

# Apply change point detection
algo = rpt.Dynp(model=model).fit(smoothed_accel)
result = algo.predict(n_bkps=1)  # You can specify the number of change points you want to detect

print(result)
result =np.array(result)-1

def integrate_until_value(x, y, value, tol = 1e-3):
    # Purpose: Integrate a function y= f(x) from a starting index until the integral reaches a certain value
    # Assumes a fairly smooth curve. Increment in a stepwise fashion
    
    diff = 100
    end = 100
    while diff > tol:
        integral = trapezoid(y[:end], x[:])

        # Calc the difference
        diff = integral[-1] - value

def get_change_points(signal, n_change_points= 1, model = "l1"):
    # Purpose: Use second order stationarity as a condtion to find the change points
    # Used by Hunstein et al. in https://ascelibrary.org/doi/full/10.1061/JGGEFK.GTENG-11550
    # For determining the mudline

    # Possible models are: 
    # l1: Penalizes the absolute difference between the data points and their estimated value after the change point. This model assumes that the changes are abrupt.
    # l2: Penalizes the squared difference between the data points and their estimated value after the change point. This model assumes that the changes are smooth.
    # rbf: Uses Gaussian basis functions to model the data. This model is suitable for detecting changes in periodic or oscillatory data.
    # linear: Assumes that the data follows a linear trend with abrupt changes.
    # discrete: Assumes that the data points are independent and identically distributed, with different means before and after the change point.
    # normal: Assumes that the data points are normally distributed, with different means and variances before and after the change point.
    # full: Uses a custom model defined by the user. You can specify your own cost function and penalty.

    # Apply change point detection
    algo = rpt.Dynp(model=model).fit(signal)
    result = algo.predict(n_bkps=n_change_points)  # You can specify the number of change points you want to detect

    # Zero shift the indices and return the results
    # Returns the change points and the last index
    return np.array(result)-1

    

def integrate_accel_data(self):
        # TODO: Update this so that the impulse df is 
        # Purpose: Integrate the impulse and store in impulse df

        # Temp storage for the df
        whole_df = self.release_df
            
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
        # self.release_df = whole_df   
        
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