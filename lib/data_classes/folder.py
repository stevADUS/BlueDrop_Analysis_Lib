import os
import datetime
import fnmatch
import glob
from pathlib import Path
import pandas as pd

from lib.data_classes.pffpFile import pffpFile


class Folder:
    # Purpose: Base class for folders

    def __init__(self, folder_dir):
        # Init the instance  varaibles
        self.folder_dir = folder_dir

    
    # Define output string
    def __str__(self):
        return f"Folder dir: {self.folder_dir}"
    
    # Get the 
    def get_num_files(self,  file_extension):
        # Purpose: Get the number of files in a folder
        
        # Get the number of files of type file extension
        count = len(list(Path(self.folder_dir).rglob(f'*{file_extension}')))
        
        # return the values
        return count

    def get_directories_by_extension(self, file_extension):
        file_dirs = []

        # TODO: if the first character is a "." strip it

        for file in glob.glob(self.folder_dir + '**/*.' + file_extension, recursive = True):
            file_path = os.path.join(self.folder_dir, file)
            file_dirs.append(file_path)
        
        return file_dirs

class pffpDataFolder(Folder):
    # Pupose: Hold data about a folder that contains a PFFP data

    # Init the input params and store them in DataFolder Instance
    def __init__(self, folder_dir, pffp_id, calibration_factor_dir):
        # init the parent class
        Folder.__init__(self, folder_dir)

        self.folder_dir = folder_dir # Store the folder directory
        self.pffp_id = pffp_id       # Store the PFFP id
        self.calibration_factor_dir = calibration_factor_dir # Directory containing the calibration factors for the PFFP

        # Init variables that aren't defined
        self.datetime_range = "Not set"
        self.calibration_excel_sheet = None
        self.calibration_params = None

    def __str__(self):
        return f"Folder: {self.folder_dir} \nDate range: {self.datetime_range} \nPFFP id: {self.pffp_id} \
                \nCalibration Param dir: {self.calibration_factor_dir}"
    
    def read_calibration_excel_sheet(self, sheet_name):
        # Purpose: Read the calibartion data for specified pffp id

        self.calibration_excel_sheet = pd.read_excel(self.calibration_factor_dir, sheet_name)

    def get_sensor_calibration_params(self, date_string):
        # Purpose: Retrieve the possible calibration dates for the selected sheet

        # temp storage of the data
        data = self.calibration_excel_sheet

        if type(data) is not pd.DataFrame:
            raise IndexError("Calibration data must be read first")
        
        # Construct the column headers
        offset_string = date_string + "_offset"
        scale_string  = date_string + "_scale"
        
        # Select those columns of the df
        self.calibration_params = data[["Sensor", offset_string, scale_string]]

    def store_pffp_files(self):
        # Purpose: Store the binary files in the 
        binary_file_dirs = self.get_directories_by_extension("bin")

        # init list to hold pffp files
        self.pffp_files = []

        # Check that the calibration params have been read
        if  type(self.calibration_params) is not pd.DataFrame:
            raise IndexError("Calibration data must be read first")
        
        # Loop over binary file directories and create instance of the pffpFile class
        for file_dir in binary_file_dirs: 
            # add the pffpFile to the list
            self.pffp_files.append(pffpFile(file_dir, self.calibration_params))


