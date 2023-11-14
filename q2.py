import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import os
import tqdm
import tracemalloc


# data precessing
def data_cleaning(data):
    # replace "?" record with nan value
    cleaned_data = data.replace("?", np.nan)
    # convert the data type to float
    cleaned_data.iloc[:, 1:19] = cleaned_data.iloc[:, 1:19].astype(float)
    # perform a max-min normalization to colume 1 to 19
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
    def predict(self, X_test, calculated, distances):
        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []
        index = []

        # if the distances are already calculated, then no need to calculate again
        if calculated:
            self.distances = distances
        else:
            self.distances = []

        # parameter to show the progress of the loop
        progress_per_loop = n
        with tqdm.trange(progress_per_loop, unit="row(s)", mininterval=0) as bar:
            bar.set_description(f"Training Progress: ")
            # Calculating Euclidean distances
            for i in bar:
                if calculated:
                    # assign the distance from the distances which is calculated in the first loop
                    distance = self.distances[i]
                else:
                    # initial the distance
                    distance = []
                    # loop through the training data to calculate the Euclidean distance
                    for j in range(m):
                        d = np.sqrt(
                            np.sum(np.square(X_test[i, 1:19] - self.X_train[j, 1:19]))
                        )
                        distance.append((d, y_train[j]))
                    # sorting distances in ascending order
                    distance = sorted(distance)
                    # save the distance to the list
                    distances.append(distance)
                # Getting k nearest neighbors
                neighbors = []
                # loop through the k nearest neighbors
                for item in range(int(self.k)):
                    neighbors.append(distance[item][1])
                # save the index
                index.append(int(X_test[i][0]))
                # save the prediction which is the mode of the k nearest neighbors
                y_pred.append(stats.mode(neighbors)[0])
        return index, y_pred, distances


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")
    # remove rows with empty cells
    training_data = training_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_training_data = data_cleaning(training_data)
    # split to training data and training label
    X_train = np.array(cleaned_training_data.iloc[:, 0:19])
    y_train = np.array(cleaned_training_data.iloc[:, 19])

    # reading the validation data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # remove rows with empty cells
    validation_data = validation_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_validation_data = data_cleaning(training_data)
    # split to training data and training label
    X_valid = np.array(cleaned_validation_data.iloc[:, 0:19])
    y_valid = np.array(cleaned_validation_data.iloc[:, 19])

    # initialize the f1 score list
    f1_macro = []
    f1_micro = []
    calculated = False
    distances = []
    # Set the number of nearest neighbors from 2 to 1% of the training data which is 300
    N = np.linspace(
        2, int(X_train.shape[0] * 0.001 + 1), num=(int(X_train.shape[0] * 0.01))
    )
    N = N.astype(int)
    # loop through the number of nearest neighbors to get the f1 score of the training data and validation data
    for n in N:
        # if n != 2, the distances are calculated in the first loop, then the distances are not calculated
        if n != 2:
            calculated = True
        # set the model
        model = KNN(n)
        # fit the data to the model
        model.fit(X_train, y_train)
        print(f"Getting {n} nearest neighbors...")
        # start the memory tracing
        tracemalloc.start(1)
        # get the prediction of the validation data
        index, predict, distances = model.predict(X_valid, calculated, distances)
        # get the memory usage
        snap = tracemalloc.take_snapshot()
        top_stats = snap.statistics("traceback")
        print(top_stats[0])
        tracemalloc.stop()
        # calculate the f1 score and save it to the list
        f1_macro.append(f1_score(y_valid, predict, average="macro"))
        f1_micro.append(f1_score(y_valid, predict, average="micro"))

    # plot the f1 score of different number of nearest neighbors
    plt.plot(f1_macro, label="F1 Macro")
    plt.plot(f1_micro, label="F1 Micro")
    plt.xticks(np.arange(len(N)), N)
    plt.xlabel("Number of nearest neighbors")
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()

    # Find the highest f1 score and the corresponding number of nearest neighbors
    print(
        f"Maximum Macro F1 score is {max(f1_macro)} when k = {f1_macro.index(max(f1_macro))+2}"
    )
    print(
        f"Maximum Micro F1 score is {max(f1_micro)} when k = {f1_micro.index(max(f1_micro))+2}"
    )

    # find the optimal number of nearest neighbors
    if max(f1_macro) >= max(f1_micro):
        optimal_k = f1_macro[f1_macro.index(max(f1_macro))]
    else:
        optimal_k = f1_micro[f1_micro.index(max(f1_micro))]

    # reading the testing data
    testing_data = pd.read_csv("data-release/data2/test.csv")
    # replace "?" record with nan value
    cleaned_testing_data = data_cleaning(testing_data)
    # assign X_test
    X_test = np.array(cleaned_testing_data.iloc[:, 0:19])

    # predict the class label of the testing data with the optimal k
    model = KNN(optimal_k)
    # fit the testing data to the model
    model.fit(X_train, y_train)
    print(f"Getting {optimal_k} nearest neighbors...")
    # start the memory tracing
    tracemalloc.start(1)
    # get the prediction of testing data
    index, predict, distances = model.predict(X_test, calculated=False, distances=[])
    snap = tracemalloc.take_snapshot()
    top_stats = snap.statistics("traceback")
    print(top_stats[0])
    tracemalloc.stop()
    # set the class label colume to object type so that the prediction will be savesd as integer
    testing_data = testing_data.astype({"class label": "object"})
    # assign the prediction to the testing data file and check again whether the prediction matches the index
    for i in range(X_test.shape[0]):
        if index[i] == testing_data.iloc[i, 0]:
            testing_data.iloc[i, 19] = predict[i]
    # save the prediction
    testing_data.to_csv("data-release/data2/test_predict.csv", index=False)
