import numpy as np
from scipy.signal import find_peaks

def check_peak_in_data(accel_data, height_range = 1.8):
    # Purpose: Check if a drop is in the file - for the
    # height_range: Can be a two element list or a single value. If list first val is minimum and second is the maximum for the peak
    #               if single value its interpreted as the minimum for peak classification

    peaks_indices, peak_dict_info = find_peaks(accel_data, height = height_range)
    
    if len(peaks_indices) > 0:
        # Check if there are peaks in the data, if yes return the data about the peaks
        return True, peaks_indices, peak_dict_info
    else:
        # If no peaks return false and don't return anymore information
        return False