class Drop:
    #Purpose: Store info about a pffp drop

    def __init__(self, containing_file, peak_index, file_drop_index, peak_info, water_drop = False):

        # Store the inputs
        self.containing_file   = containing_file   # Store which file the drop is in 
        self.file_drop_index   = file_drop_index   # Tracks which drop this is in the file e.g. the first drop when there are 5 in the folder
        self.peak_info         = peak_info         # Store some information on the peak
        self.peak_index = peak_index
        self.water_drop = False                    # Store if the drop is a water drop

    def __str__(self):
        # Purpose: Outputs information about the drops
        return f"----- Drop Info ----- \nContaining file: {self.containing_file} \nFile Drop Index: {self.file_drop_index} \nWater Drop: {self.water_drop}"
    
    def get_drop_start_end(self):
        # Purpose: Get the start and end of a drop
        pass
        # Use the peak index
    def store_sensor_data(self):
        # Purpose: Store the sensor data only for where the drop is selected to be
        pass

    def concate_accelerations(self):
        # Purpose: Tie the accelerometers together where they max out
        pass

    def integrate(self):
        pass

    def plot(self, axs, x = ["acceleration", "velocity"], y = "displacement", x_units = ["g", "kPa"], y_units = ["m"], hold_on = True):
        # Purpose: Plot the drop
        pass