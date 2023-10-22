import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt


def cleaning_dirty_data(original_data):
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    return cleaned_data


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data1/training.csv")
    # print(training_data.info())
    # print("Shape of dataframe:", training_data.shape)
    # print(training_data.isnull().sum())

    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    # print(cleaned_training_data.info())
    # print("Shape of dataframe:", cleaned_training_data.shape)
    # print(cleaned_training_data.isnull().sum())

    # print max and min values of columns "S1" and "S2"
    # print("Max value of S1:", cleaned_training_data["S1"].max())
    # print("Min value of S1:", cleaned_training_data["S1"].min())
    # print("Max value of S2:", cleaned_training_data["S2"].max())
    # print("Min value of S2:", cleaned_training_data["S2"].min())

    print(cleaned_training_data.size)
    # show unique variables in columns "S" and "T0"
    # unique_S = cleaned_training_data[3, :].unique()
    # unique_T0 = cleaned_training_data[:, 4].unique()
    # print("Unique values in column S:", unique_S)
    # print("Unique values in column T0:", unique_T0)

    # print(cleaned_training_data.head(5))
