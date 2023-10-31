import numpy as np
import sklearn
import pandas as pd
from sklearn.metrics import f1_score
import statistics


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


def calculate_distance(x, y):
    # calculate the distance between two records
    return np.sqrt(np.sum((x - y) ** 2))


# set the number of nearest neighbors
k = 10

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

    # initial the matrix to record the distance between the validation data i and each training data j
    distance = np.zeros(
        [normalized_validation_data.shape[0], normalized_training_data.shape[0]]
    )
    # initial the array to record the prediction of the validation data
    predict = np.zeros([normalized_validation_data.shape[0]], dtype=int)
    # Loop through each validation to calculate the distance between the validation data i and each training data j
    for i in range(normalized_validation_data.shape[0] - 29999):
        for j in range(normalized_training_data.shape[0]):
            distance[i, j] = calculate_distance(
                normalized_validation_data.iloc[i, 1:19],
                normalized_training_data.iloc[j, 1:19],
            )
            print(i, j)
        # find the k nearest neighbors and its corresponding index
        sorted_indices = np.argsort(distance[i, :])
        # initial the array to store the class label of the k nearest neighbors
        voting = []
        for x in sorted_indices[:k]:
            voting.append(normalized_training_data.iloc[x, 19])
        # find the most frequent class label in k nearest neighbors
        predict[i] = statistics.mode(voting)

    # calculate the macro F1 score and micro F1 score
    Y_validate = normalized_validation_data.iloc[:, 19]
    f1_macro = f1_score(Y_validate, predict, average="macro")
    f1_micro = f1_score(Y_validate, predict, average="micro")
    print(f"Macro F1 score: {f1_macro:.4f}")
    print(f"Micro F1 score: {f1_micro:.4f}")
