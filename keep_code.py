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