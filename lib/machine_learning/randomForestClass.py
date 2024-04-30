from joblib import load as joblib_load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RandForest:
    def __init__(self, name, model_dir):
        # Store the model name
        self.name = name

        # Store the path to the model
        self.model_dir = model_dir
        
        # Load the model object
        self._load_model_object()

    def _load_model_object(self):
        """
        ## Purpose: Wrapper for the package modules that load the model objects
        """
        # NOTE: In the future might need to use different load functions depending on the model type
        
        # Use joblib.load to load the model     
        self.model_obj = joblib_load(self.model_dir)

    def predict_probability(self, accel_data, max_displacement):
        """
        ## Purpose: Run a model that is used to predict the probability of the input
        """
        # Stack the data into a nested column arr (Expected input of the trained model)
        input = np.column_stack((accel_data, max_displacement))

        # Call the predict_proba module of the loaded machine learning object
        output = self.model_obj.predict_proba(input)

        return output

    @staticmethod
    def plot_prediction(data, labels= ['Class 1','Class 2','Class 3','Class 4'], title = "", fig_size = [6, 4]):
        """
        ## Purpose: Plot the prediction data
        """

        fig, axs = plt.subplots(ncols=1, nrows = 1, figsize = fig_size)
        # Plot the data
        axs.bar(labels, data)

        # If the title is something other than an empty string show the title
        if not title == "":
            axs.set_title(title)

        axs.set_xlabel("Classes")
        axs.set_ylabel("Predicted Probability")
        plt.show()

if __name__ == "__main__":
    # Add some tests here
    pass