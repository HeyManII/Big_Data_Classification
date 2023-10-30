import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy


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
    # normalizing 18 columns excluding the id and the class label
    normalized_training_data = normalize_data(cleaned_training_data)

    # reading the validation data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # cleaning empty cell inside the training_data
    cleaned_validation_data = cleaning_dirty_data(training_data)
    # normalizing 18 columns excluding the id and the class label
    normalized_validation_data = normalize_data(cleaned_validation_data)

    # calculate the macro F1 score and micro F1 score
    Y_validate = normalized_validation_data.iloc[:, 19]
    f1_macro = f1_score(Y_validate, y_pred, average="macro")
    f1_micro = f1_score(Y_validate, y_pred, average="micro")
    print(f"Macro F1 score: {f1_macro:.4f}")
    print(f"Micro F1 score: {f1_micro:.4f}")
