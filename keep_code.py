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
