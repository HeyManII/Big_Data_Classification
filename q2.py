import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats
import os
import matplotlib.pyplot as plt
import tqdm


def data_cleaning(original_data):
    # remove rows with empty cells
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    # replace "?" record with nan value
    cleaned_data = cleaned_data.replace("?", np.nan)
    # convert the data type to float
    cleaned_data.iloc[:, 1:19] = cleaned_data.iloc[:, 1:19].astype(float)
    # perform a max--min normalization to colume 1 to 19
    cleaned_data.iloc[:, 1:19] = (
        cleaned_data.iloc[:, 1:19] - cleaned_data.iloc[:, 1:19].min()
    ) / (cleaned_data.iloc[:, 1:19].max() - cleaned_data.iloc[:, 1:19].min())
    # replace the nan to the mean of the column
    cleaned_data.fillna(value=cleaned_data.mean(), inplace=True)
    return cleaned_data


# KNN algorithm
class KNN:
    # initial the number of nearest neighbors
    def __init__(self, k):
        self.k = k

    # fit the training data
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # predict the class label of the testing data
    def predict(self, X_test):
        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []

        # check if the distance matrix is already calculated
        if os.path.isfile("data-release/data2/distance.csv"):
            temp = pd.read_csv(
                "data-release/data2/distance.csv", header=None, index_col=None
            )
            # reading the data from csv and converting for further use
            temp = np.array(temp)
            distances = []
            for i in range(temp.shape[0]):
                row = []
                for j in range(temp.shape[1]):
                    ceil = temp[i, j]
                    distance, index = ceil.split(",")
                    distance = distance.replace("(", "")
                    index = index.replace(")", "")
                    row.append((float(distance), int(index)))
                distances.append(row)
        else:
            progress_per_loop = m
            distances = []
            # Calculating Euclidean distances
            for i in range(n):
                distance = []
                with tqdm.trange(progress_per_loop, unit="%", mininterval=0) as bar:
                    bar.set_description(f"Testing Record {i}: ")
                    for j in bar:
                        d = np.sqrt(
                            np.sum(np.square(X_test[i, :] - self.X_train[j, :]))
                        )
                        distance.append((d, y_train[j]))
                # sorting distances in ascending order
                distance = sorted(distance)
                distances.append(distance)
            distances = pd.DataFrame(distances)
            distances.to_csv(
                "data-release/data2/distance.csv", index=False, header=False
            )

        # Getting k nearest neighbors
        print(f"Getting {self.k} nearest neighbors")
        for i in range(n):
            # Getting k nearest neighbors
            neighbors = []
            distance = distances[i]
            for item in range(self.k):
                neighbors.append(distance[item][1])  # appending K nearest neighbors
            y_pred.append(stats.mode(neighbors)[0])  # For Classification
        return y_pred


# set the number of nearest neighbors testing for validation data
N = [2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]

if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")
    # cleaning the training_data
    cleaned_training_data = data_cleaning(training_data)
    # split to training data and training label
    X_train = np.array(cleaned_training_data.iloc[:, 1:19])
    y_train = np.array(cleaned_training_data.iloc[:, 19])

    # reading the validation data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # cleaning the training_data
    cleaned_validation_data = data_cleaning(training_data)
    # split to training data and training label
    X_valid = np.array(cleaned_validation_data.iloc[:, 1:19])
    y_valid = np.array(cleaned_validation_data.iloc[:, 19])

    f1_macro = []
    f1_micro = []
    for n in N:
        model = KNN(n)
        model.fit(X_train, y_train)
        predict = model.predict(X_valid)
        f1_macro.append(f1_score(y_valid, predict, average="macro"))
        f1_micro.append(f1_score(y_valid, predict, average="micro"))

    plt.plot(f1_macro, label="F1 Macro")
    plt.plot(f1_micro, label="F1 Micro")
    plt.xticks(np.arange(len(N)), N)
    plt.xlabel("Number of nearest neighbors")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()

    print(f1_macro.index(max(f1_macro)), max(f1_macro))
    print(f1_micro.index(max(f1_micro)), max(f1_micro))

    if max(f1_macro) >= max(f1_micro):
        optimal_k = f1_macro[f1_macro.index(max(f1_macro))]
    else:
        optimal_k = f1_micro[f1_micro.index(max(f1_micro))]

    # reading the testing data
    testing_data = pd.read_csv("data-release/data2/testing.csv")
    # cleaning the testing data
    cleaned_testing_data = data_cleaning(testing_data)
    # assign X_test
    X_test = np.array(cleaned_training_data)
    model = KNN(k=optimal_k)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    f1_macro.append(f1_score(y_valid, predict, average="macro"))
    f1_micro.append(f1_score(y_valid, predict, average="micro"))
