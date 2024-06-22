import os  # Import the os module for operating system functions
import datetime  # Import the datetime module for handling dates and times
import fnmatch  # Import fnmatch module for filename matching
import glob  # Import glob module for pathname matching
from pathlib import Path  # Import Path class from pathlib module
import pandas as pd  # Import pandas library

from lib.data_classes.pffpFile import pffpFile  # Import pffpFile class from lib.data_classes.pffpFile module


class Folder:
    """
    Base class for folders.
    """

    def __init__(self, folder_dir):
        """
        Initialize the Folder object with folder directory.

        Inputs:
        - folder_dir: The directory path of the folder.
        """
        self.folder_dir = folder_dir  # Store the folder directory

    def __str__(self):
        """
        Define the string representation of the Folder object.
        """
        return f"Folder dir: {self.folder_dir}"

    def get_num_files(self, file_extension, recursive):
        """
        Get the number of files in a folder with a specific file extension.

        Inputs:
        - file_extension: The file extension (e.g., '.txt', '.csv').
        - recursive: Boolean flag to indicate whether to search subdirectories recursively.

        Returns:
        - count: The number of files with the specified extension.
        """
        # Count files matching the file extension in the folder
        count = len(glob.glob(self.folder_dir + "/*" + file_extension, recursive=recursive))
        return count

    def get_directories_by_extension(self, file_extension, recursive, subfolder=""):
        """
        Get a list of file directories with a specific file extension in the folder.

        Inputs:
        - file_extension: The file extension (e.g., '.txt', '.csv').
        - recursive: Boolean flag to indicate whether to search subdirectories recursively.
        - subfolder: Optional subfolder name to search within.

        Returns:
        - file_dirs: List of file directories matching the criteria.
        """
        file_dirs = []

        # Handle subfolder path construction
        if subfolder != "":
            subfolder = "/" + subfolder

        # Handle recursive search based on the flag
        if recursive:
            # Search recursively for files matching the extension
            for file in glob.glob(self.folder_dir + subfolder + "/**/*." + file_extension, recursive=True):
                file_path = os.path.join(self.folder_dir, file)
                file_dirs.append(file_path)
        else:
            # Search in the specified subfolder for files matching the extension
            for file in glob.glob(self.folder_dir + subfolder + "/*." + file_extension):
                file_path = os.path.join(self.folder_dir, file)
                file_dirs.append(file_path)

        return file_dirs


if __name__ == "__main__":
    # Add some testing here
    pass
