import datetime  # Import the datetime module
import os  # Import the os module for operating system functions

class File:
    # Purpose: Define a class that holds general file properties
    def __init__(self, file_dir):
        """
        Initialize a File object with file directory.
        
        Inputs:
        - file_dir: The directory of the file.
        """
        self.file_dir = file_dir  # Store the file directory
        
        # Initialize variable to store the file name
        self.file_name = ""

    def get_file_datetime(self):
        """
        Get the date the file was last modified (Assumed to be the drop date).
        """
        modification_time = os.path.getmtime(self.file_dir)  # Get file modification time
        
        # Convert modification time to datetime object and store it
        self.datetime = datetime.datetime.fromtimestamp(modification_time)
    
    def get_file_name(self):
        """
        Get the file name from the directory.
        """
        self.file_name = os.path.basename(self.file_dir)  # Get file name from directory path

    def new_file_name(self, new_name):
        """
        Update the file name and physically rename the file.
        
        Inputs:
        - new_name: The new name for the file.
        """
        # Store the old file name and directory
        old_name = self.file_name
        old_dir = self.file_dir

        # Update the file directory with the new file name
        self.file_dir = self.file_dir.replace(old_name, new_name)

        # Update the stored file name
        self.file_name = new_name

        # Actually rename the physical file
        os.rename(old_dir, self.file_dir)


if __name__ == "__main__":
    # Add some testing here
    pass
