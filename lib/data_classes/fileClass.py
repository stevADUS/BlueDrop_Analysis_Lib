import datetime
import os

class File:
    # Purpose: Define a class that holds general file properties
    def __init__(self, file_dir):
        self.file_dir = file_dir # Store the file Dir
        
        # Init variable to store the file name
        self.file_name = ""

    def get_file_datetime(self):
        # Purpose: Get the date the file was last modified (Assumed to be the drop date)
        modification_time = os.path.getmtime(self.file_dir)

        self.datetime = datetime.datetime.fromtimestamp(modification_time)
    
    def get_file_name(self):
        # Purpose: Get a file name from a directory
        self.file_name = os.path.basename(self.file_dir)

    def new_file_name(self, new_name):
        """
        Purpose: Update the file name
        """
        # Store the old file name and directory
        old_name = self.file_name
        old_dir = self.file_dir

        # Update the file directory
        self.file_dir = self.file_dir.replace(old_name, new_name)

        # Update the stored file name
        self.file_name = new_name

        # Actually rename the physical file
        os.rename(old_dir, self.file_dir)


if __name__ == "__main__":
    # Add some testing here
    pass