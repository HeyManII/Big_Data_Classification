import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt


def cleaning_dirty_data(original_data):
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    return cleaned_data


if __name__ == "__main__":
    training_data = pd.read_csv("data-release/data1/training.csv")
    print(training_data.info())
    print("Shape of dataframe:", training_data.shape)
    # print(training_data.head(5))
    print(training_data.isnull().sum())

    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    print(cleaned_training_data.info())
    print("Shape of dataframe:", cleaned_training_data.shape)
    print(cleaned_training_data.isnull().sum())

    print(training_data.head(5))
