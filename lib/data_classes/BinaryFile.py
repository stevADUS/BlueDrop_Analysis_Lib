import pandas as pd
import numpy as np

from lib.data_classes.fileClass import File
from lib.data_classes.Type_Mixin import TypeMixin


# Binary file class inherits the File class and the TypeMixin
class BinaryFile(File, TypeMixin):
    """
    Pupose: The BinaryFile class is used to represent any type of binary file.
            it contains modules that read and convert binary file to dfs.
    """
    # Purpose define properties of binary files

    def __init__(self, file_dir):
        File.__init__(self, file_dir) # inherit parent methods and store file dir
    
    # method to print info about the binary file
    def __str__(self):
        return f"File Directory: {self.file_dir}"
    
    def read_binary(self, size_byte = 3):
        """
        Purpose: Read a binary file and store it in an arr
        
        where:
            size_byte: number of bytes that should be read at a time.
                In terms of the PFFP binary files this corresponds to the number of bytes a number is represented by
        """
        #TODO: maybe convert the binary right away. You have to unwrap the data again the way it is right now
        
        # Init the arr
        binary_arr = []

        with open(self.file_dir, 'rb') as file:
            while True:
                data_chunk = file.read(size_byte)
                if not data_chunk:
                    break  # break the loop if there is no more data
                binary_arr.append(data_chunk)

        return binary_arr

    def binary_file_2_df(self, column_names, num_columns, num_rows = -1, size_byte = 3, byte_order = "big", signed = True):
        """
        ### Purpose: Read a binary file and store the data in a df

        #### where:
        column_names: columns names for the created df
        num_columns: Number of columns that should be in the new df
        num_rows: Number of the rows in the new df (preset to -1 so that the value is implicit) 
        size_byte: Number of bytes that should be read at once (This is the number of bytes that make up a value)
        byte_order: Defines how the bytes of a binary number are arranged in memory. For PFFP binary files the byte_order is "big"
        """

        # Read the binary file
        binary_arr = self.read_binary(size_byte)
        
        # Convert the binary to an integer arr
        integer_arr = BinaryFile.binary_arr_2_integer_arr(binary_arr, byte_order, signed)

        # Reformat the arr
        integer_matrix = BinaryFile.arr_2_matrix(integer_arr, num_rows, num_columns)

        # return the df
        return pd.DataFrame(integer_matrix, columns = column_names)

if __name__ == "__main__":
    # Add some testing here
    pass

