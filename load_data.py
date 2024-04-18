import pandas as pd
import os
import torch
import numpy as np
import torch.utils
import torch.utils.data
import global_constants

def load_data(data_folder) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads csv files as a list of tuples of data and labels.
    Data are 1401x13 (1401 time points and 13 health metrics).
    Labels are 1x5 (one-hot encoding for 5 severity levels).
    """
    dataset = []
    categories = sorted(os.listdir(data_folder), key=lambda x: global_constants.ASTHMA_CASES[x])
    categories = {category: i for i, category in enumerate(categories)}

    for category in categories:
        category_index = categories[category]
        one_hot_label = np.zeros(len(categories))
        one_hot_label[category_index] = 1
        for csv_file in os.listdir(f"{data_folder}/{category}"):
            dataset.append((
                pd.read_csv(f"{data_folder}/{category}/{csv_file}").values,
                one_hot_label
            ))
    return dataset

def split_dataset(dataset):
    """
    Splits dataset randomly into training (70%), validation (15%), and test (15%)
    """
    generator = torch.Generator().manual_seed(1)
    train, val, test = torch.utils.data.random_split(
        dataset,
        [0.7, 0.15, 0.15],
        generator=generator
    )

    return train, val, test
