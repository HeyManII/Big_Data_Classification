import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt


def cleaning_dirty_data(original_data):
    # remove rows with empty cells
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    # replace "?" record with 0
    cleaned_data = cleaned_data.replace("?", int(0))
    # convert the data type to float
    cleaned_data.iloc[:, 1:19] = cleaned_data.iloc[:, 1:19].astype(float)
    return cleaned_data


def normalize_data(original_data):
    # perform a max--min normalization to colume 1 to 19
    original_data.iloc[:, 1:19] = (
        original_data.iloc[:, 1:19] - original_data.iloc[:, 1:19].min()
    ) / (original_data.iloc[:, 1:19].max() - original_data.iloc[:, 1:19].min())
    return original_data


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")

    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)

    print(cleaned_training_data.head(5))

    # normalizing 18 columns excluding the id and the class label
    normalized_training_data = normalize_data(cleaned_training_data)

    print(normalized_training_data.head(5))
