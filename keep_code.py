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