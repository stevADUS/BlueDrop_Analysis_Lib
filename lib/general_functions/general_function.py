import os
import numpy as np
import pandas as pd
import shutil
import sys

# File Purpose: Store general purpose functions - I will organize this better later
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def apply_mask_to_list(values, mask):
    return [value for value, m in zip(values, mask) if m]

def progress_bar(iterations, total, time_left):
    progress = iterations / total
    bar_length = 50
    completed_length = int(bar_length * progress)
    remaining_length = bar_length - completed_length

    bar = "[" + "=" * completed_length + "-" * remaining_length + "]"
    percentage = "{:.2%}".format(progress)
    time_est = "{:.2}".format(time_left/60)
    sys.stdout.write("\r" + bar + " " + percentage + " ETA (min): " + time_est)
    sys.stdout.flush()
