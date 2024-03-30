import numpy as np
from scipy.signal import find_peaks

def find_drops(accel_data, impact_time_tol = 0.015, sample_freq = 120_000, min_peak_height = 2.8):
# Purpose: Get the number of drops in the a file
        
    # Calc assumed tolerance for the length of impact
    impact_window = impact_time_tol * sample_freq

    drop_indexs, drop_info = find_peaks(accel_data, height = min_peak_height, distance = impact_window)
        
    num_drops = len(drop_indexs)

    return drop_indexs, drop_info, num_drops

def check_peaks_in_data(accel_data, height_range = 2.8):
    # Purpose: Check if a drop is in the file - for the
    # height_range: Can be a two element list or a single value. If list first val is minimum and second is the maximum for the peak
    #               if single value its interpreted as the minimum for peak classification

    peaks_indices, peak_dict_info = find_peaks(accel_data, height = height_range)
    
    if len(peaks_indices) > 0:
        # Check if there are peaks in the data, if yes return the data about the peaks
        return [True, peaks_indices, peak_dict_info]
    else:
        # If no peaks return false and don't return anymore information
        return [False]
    
# Function to do a moving average
def moving_average(data, window_size):
    # Purpose: Calc the moving average of the data using set window size

    # data: Data that should be smoothed/averaged
    # window_size: Number of values to consider +/- in the averaging

    averaged_data = np.cumsum(data)
    averaged_data[window_size:] = averaged_data[window_size:] - averaged_data[:-window_size]
    return averaged_data[window_size-1:]/window_size

def find_deriv_change(x, y, cutoff, greater = True):
    # Purpose: Find the derivative of the x,y data a determine the location where the derivative reaches the threshold

    # x: Array of independent data
    # y: Array of dependent data
    # greater: Determines if a search from greater than or less than the cutoff

    # Info: np.gradient uses second order central differencing on the interior and first order accurate differencing on the boundaries
    
    # Calc the derivative
    derivative = np.gradient(y, x)
    
    if greater:
        # Find where the value of the derivative is greater than the cutoff
        criteria_indices = np.nonzero(derivative > cutoff)
    else:
        # Find where the value of the derivative is less than the cutoff
        criteria_indices = np.nonzero(derivative < cutoff)

    return criteria_indices