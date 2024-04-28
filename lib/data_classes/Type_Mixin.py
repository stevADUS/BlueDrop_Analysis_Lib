import numpy as np
import pandas as pd

class TypeMixin:
    @staticmethod
    def array_2_df(array, column_names):
        # Purpose: Convert an array to a dataframe with the given column names
        
        df = pd.DataFrame(array)
        df.columns = column_names
        return df

    @staticmethod
    def binary_arr_2_integer_arr(binary_arr, byte_order, signed_flag):
        # Purpose: Convert a binary file to an integer array

        # COnvert the binary arr to integer array
        integer_arr = [int.from_bytes(b, byteorder= byte_order, signed = signed_flag) for b in binary_arr]
    
        # Convert the data to np_array
        return np.array(integer_arr)
    
    @staticmethod
    def arr_2_matrix(arr, num_rows, num_cols):
        # Purpose: Convert integer array to matrix

        # Reshape the data and store in class variable
        return np.reshape(arr, (num_rows, num_cols))
    
    @staticmethod
    def output_df_2_excel(df, file_dir, sheet_name):
        # Purpose: Output a df to an excel sheet with
        # TODO: Make this so it can create the sheet maybe pandas already does that
        # file_dir: Directory of the excel file
        # sheet_name: Name of the sheet the data should be written to

        # TODO: Add checks and error catches
        # Write the data
       pd.DataFrame.to_excel(df, file_dir, sheet_name)

if __name__ == "__main__":
    # Add some testing here
    pass