import datetime
import os

class File:
    # Purpose: Define a class that holds general file properties
    def __init__(self, file_dir):
        self.file_dir = file_dir # Store the file Dir
        
    def get_file_datetime(self):
        # Purpose: Get the date the file was last modified (Assumed to be the drop date)
        modification_time = os.path.getmtime(self.file_dir)

        self.datetime = datetime.datetime.fromtimestamp(modification_time)
    
    def get_file_name(self):
        # Purpose: Get a file name from a directory
        return os.path.basename(self.file_dir)
    