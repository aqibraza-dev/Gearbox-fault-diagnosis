import os
import scipy.io as sio
import pandas as pd

def convert_mat_to_csv(directory="."):
    """Finds all .mat files in a directory and converts them to .csv format."""
    for file in os.listdir(directory):
        if file.endswith('.mat'):
            print(f"Converting {file} to CSV...")
            mat = sio.loadmat(os.path.join(directory, file))
            # Extract variables skipping internal MATLAB properties
            data = [v for k, v in mat.items() if not k.startswith('__')]
            if data:
                data = data[0]
                csv_filename = file.replace(".mat", ".csv")
                pd.DataFrame(data).to_csv(os.path.join(directory, csv_filename), index=False)