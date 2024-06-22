import numpy as np
import pandas as pd

class TypeMixin:
    @staticmethod
    def array_2_df(array, column_names):
        """
        Convert a numpy array to a pandas DataFrame with the given column names
        
        where:
        - array: A numpy array to be converted into a DataFrame
        - column_names: A list of column names for the DataFrame
        """
        df = pd.DataFrame(array)  # Convert the numpy array to a DataFrame
        df.columns = column_names  # Set the column names of the DataFrame
        return df  # Return the resulting DataFrame

    @staticmethod
    def binary_arr_2_integer_arr(binary_arr, byte_order, signed_flag):
        """
        Convert an array of binary data to an array of integers
        
        where:
        - binary_arr: A list of byte sequences (binary data)
        - byte_order: A string indicating the byte order ('big' or 'little')
        - signed_flag: A boolean indicating whether the integers are signed or unsigned
        """
        # Convert each byte sequence in binary_arr to an integer using the specified byte order and signed flag
        integer_arr = [int.from_bytes(b, byteorder=byte_order, signed=signed_flag) for b in binary_arr]
    
        # Convert the list of integers to a numpy array and return it
        return np.array(integer_arr)
    
    @staticmethod
    def arr_2_matrix(arr, num_rows, num_cols):
        """
        Reshape a numpy array into a matrix (2D array) with the specified number of rows and columns
        
        where:
        - arr: A numpy array to be reshaped
        - num_rows: The number of rows for the resulting matrix
        - num_cols: The number of columns for the resulting matrix
        """
        # Reshape the numpy array into a matrix with the specified dimensions and return it
        return np.reshape(arr, (num_rows, num_cols))
    
    @staticmethod
    def output_df_2_excel(df, file_dir, sheet_name):
        """
        Output a DataFrame to an Excel sheet
        
        where:
        - df: The DataFrame to be written to an Excel file
        - file_dir: The file path of the Excel file to write to
        - sheet_name: The name of the sheet where the data should be written
        """
        # TODO: Add error handling to ensure the file can be written correctly
        # Write the DataFrame to the specified Excel file and sheet
        df.to_excel(file_dir, sheet_name)

if __name__ == "__main__":
    # Add some testing here
    pass