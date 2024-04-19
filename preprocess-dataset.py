import os
import shutil
import global_constants
import pandas as pd
import numpy as np

# clear asthma-dataset-1-augmented-1 of its folders

# for each folders in asthma-dataset-1
#   create a folder of the same name in asthma-dataset-1-augmented-1
#   for each csv file in this folder,
#     load the data from the csv file
#     augment the data:
#       - perhaps add small gaussian noise,
#           mean 0
#           variance should be small
#               maybe, for a given column, get the average distance between consecutive rows
#               OR some fraction of the difference between the min and max val in the col
#       - perhaps offset the event by removing a uniform random number of
#           seconds of rows from the beginning of the dataset
#       - perhaps remove rows altogether!
#       - perhaps make multiple copies of same csv with different augmentations. let's say 5

def augment_data(data_folder, augmented_folder="asthma-dataset-1-augmented-1") -> None:
    """

    """
    np.random.seed(1)
    if augmented_folder in os.listdir("./"):
        shutil.rmtree(augmented_folder)
    os.mkdir(augmented_folder)

    dataset = []
    categories = sorted(os.listdir(data_folder), key=lambda x: global_constants.ASTHMA_CASES[x])
    categories = {category: i for i, category in enumerate(categories)}

    for category in categories:
        os.mkdir(f"{augmented_folder}/{category}")
        for csv_file in os.listdir(f"{data_folder}/{category}"):
            df = pd.read_csv(f"{data_folder}/{category}/{csv_file}")

            # Make 3 augmentations
            for i in range(3):
                # Add Gaussian noise (not to time column)
                # Use 0 mean and standard deviation equal to column's range / 8 
                column_ranges = df.max(0)-df.min(0)
                for col in df.columns[1:]:
                    noise = np.random.normal(0, column_ranges[col] / 8, len(df))
                    df[col] += noise

                # Save augmented data
                df = df.round(4)
                df.to_csv(f"{augmented_folder}/{category}/{csv_file[:-4]}-{i}{csv_file[-4:]}", index=False)

    return dataset

augment_data("asthma-dataset-1", "asthma-dataset-1-augmented-1")
