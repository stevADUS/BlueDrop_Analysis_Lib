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

        for file in glob.glob(self.folder_dir + "/**/*.bin", recursive=True):
            file_path = os.path.join(self.folder_dir, file)
            file_dirs.append(file_path)
        
        return file_dirs




