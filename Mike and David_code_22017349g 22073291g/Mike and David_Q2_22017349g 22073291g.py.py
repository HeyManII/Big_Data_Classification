import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import tqdm
from sklearn.neighbors import BallTree
import copy
import seaborn as sns
import tracemalloc
import time


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


def plot_distribution(data):
    # plot X_train column 1 to 19 as a scatter plot with different color of different columns
    data.iloc[:, 1:19] = data.iloc[:, 1:19].apply(pd.to_numeric, errors="coerce")
    print(data.info)
    for i in range(1, 19):
        plt.figure(figsize=(8, 5))
        plt.scatter(data.iloc[:, i], data.iloc[:, 0], s=5)
        plt.xlabel(f"Feature Value {i} Value")
        plt.ylabel("Record ID")
        plt.title(f"Scatter Plot of Attribute {i} Distribution")
        plt.show()


# --------- First trial on building KNN brute force algorithm from scratch ---------
# KNN algorithm
# class KNN:
#     # initial the number of nearest neighbors
#     def __init__(self, k):
#         self.k = k

#     # fit the training data
#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train

#     # predict the class label of the testing data
#     def predict(self, X_test, calculated, distances):
#         m = self.X_train.shape[0]
#         n = X_test.shape[0]
#         y_pred = []
#         index = []

#         # if the distances are already calculated, then no need to calculate again
#         if calculated:
#             self.distances = distances
#         else:
#             self.distances = []

#         # parameter to show the progress of the loop
#         progress_per_loop = n
#         with tqdm.trange(progress_per_loop, unit="row(s)", mininterval=0) as bar:
#             bar.set_description(f"Training Progress: ")
#             # Calculating Euclidean distances
#             for i in bar:
#                 if calculated:
#                     # assign the distance from the distances which is calculated in the first loop
#                     distance = self.distances[i]
#                 else:
#                     # initial the distance
#                     distance = []
#                     # loop through the training data to calculate the Euclidean distance
#                     for j in range(m):
#                         d = np.sqrt(
#                             np.sum(np.square(X_test[i, 1:19] - self.X_train[j, 1:19]))
#                         )
#                         distance.append((d, y_train[j]))
#                     # sorting distances in ascending order
#                     distance = sorted(distance)
#                     # save the distance to the list
#                     distances.append(distance)
#                 # Getting k nearest neighbors
#                 neighbors = []
#                 # loop through the k nearest neighbors
#                 for item in range(int(self.k)):
#                     neighbors.append(distance[item][1])
#                 # save the index
#                 index.append(int(X_test[i][0]))
#                 # save the prediction which is the mode of the k nearest neighbors
#                 y_pred.append(stats.mode(neighbors)[0])
#         return index, y_pred, distances


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")
    # show the data distribution of the asstributes in training data
    plot_distribution(training_data.iloc[:, 0:19])
    # remove rows with empty cells
    training_data = training_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_training_data = data_cleaning(training_data)
    # plot the distribution of the attributes in training data after data processing
    plot_distribution(cleaned_training_data)
    # split to training data and training label
    X_train = np.array(cleaned_training_data.iloc[:, 1:19])
    y_train = np.array(cleaned_training_data.iloc[:, 19])

    # calculate the correlation matrix of the training data
    corr_matrix = np.corrcoef(np.array(X_train), rowvar=False)
    # plot the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=0.8)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True, fmt=".2f")
    plt.title("Correlation Matrix of X_train")
    plt.savefig("correlation.png")
    plt.show()

    # plot the distribution of the class label after data processing
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(train_labels, train_counts)
    plt.title("Training Label Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Labels")
    plt.savefig("label.png")
    plt.show()

    # reading the validation data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # remove rows with empty cells
    validation_data = validation_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_validation_data = data_cleaning(training_data)
    # split to training data and training label
    X_valid = np.array(cleaned_validation_data.iloc[:, 1:19])
    y_valid = np.array(cleaned_validation_data.iloc[:, 19])

    # set the number of nearest neighbors from 2 to 0.01% of the training data which is 30
    K = np.linspace(
        2, int(X_train.shape[0] * 0.001), num=(int(X_train.shape[0] * 0.001 - 1))
    )
    K = K.astype(int)
    # initial the f1 score list
    f1_macro = []
    f1_micro = []

    # monitor the memory used for building ball tree
    tracemalloc.start(20)
    # monitor the time used for building ball tree
    time1 = time.perf_counter()
    # build the ball tree for testing data
    tree = BallTree(X_train, metric="euclidean", leaf_size=35000)
    time2 = time.perf_counter()
    snapshot = tracemalloc.take_snapshot()
    top_memory_stats = snapshot.statistics("traceback")
    memory_stat = top_memory_stats[0]
    print(
        f"Memory used on building Ball Tree: {round(memory_stat.size / 1024**2, 4)} MiB"
    )
    print(f"Time on building Ball Tree: {round(time2 - time1, 5)} seconds")
    tracemalloc.stop()
    print()

    # predict the class labels of the validation data
    for k in K:
        print(f"Predicting Validation dataset with {k} nearest neighbors")
        tracemalloc.start(20)
        time1 = time.perf_counter()
        # find the k nearest neighbors with k = 2 to 30
        distances, indices = tree.query(X_valid, k=k)
        y_pred = []
        # parameter to show the progress of the loop
        progress_per_loop = len(indices)
        with tqdm.trange(progress_per_loop, unit="row(s)", mininterval=0) as bar:
            bar.set_description(f"Training Progress: ")
            for i in bar:
                # initialize the neighbors list
                neighbors = []
                for j in range(len(indices[i])):
                    # append the class label of the nearest neighbors to the list
                    neighbors.append(y_train[indices[i][j]])
                # Voting process of KNN: find the mode of the neighbors listand save the prediction
                y_pred.append(stats.mode(neighbors)[0])
        time2 = time.perf_counter()
        snapshot = tracemalloc.take_snapshot()
        top_memory_stats = snapshot.statistics("traceback")
        memory_stat = top_memory_stats[0]
        tracemalloc.stop()
        print(f"Memory used: {round(memory_stat.size / 1024**2, 4)} MiB")
        print(
            f"Time on looking for {k} nearest neighbors: {round(time2 - time1, 2)} seconds"
        )
        f1_macro.append(f1_score(y_valid, y_pred, average="macro"))
        f1_micro.append(f1_score(y_valid, y_pred, average="micro"))

    # plot the f1 score of different number of nearest neighbors
    plt.plot(f1_macro, label="F1 Macro")
    plt.plot(f1_micro, label="F1 Micro")
    plt.xlabel("Number of nearest neighbors")
    plt.xticks(np.arange(len(K)), K)
    plt.ylabel("F1 score")
    plt.legend()
    plt.show()

    print(
        f"Maximum Macro F1 score is {max(f1_macro)} when k = {K[f1_macro.index(max(f1_macro))]}"
    )
    print(
        f"Maximum Micro F1 score is {max(f1_micro)} when k = {K[f1_micro.index(max(f1_micro))]}"
    )

    # reading the testing data
    testing_data = pd.read_csv("data-release/data2/test.csv")
    # replace "?" record with nan value
    cleaned_testing_data = data_cleaning(testing_data)
    # assign X_test
    X_test = np.array(cleaned_testing_data.iloc[:, 1:19])

    # find the optimal number of nearest neighbors
    if max(f1_macro) >= max(f1_micro):
        optimal_k = K[f1_macro.index(max(f1_macro))]
    else:
        optimal_k = K[f1_micro.index(max(f1_micro))]
    print(f"Therefore, {optimal_k} is the optimal value of k")

    print()
    print(f"Predicting Testing dataset with {optimal_k} nearest neighbors")
    tracemalloc.start(20)
    time1 = time.perf_counter()
    distances, indices = tree.query(X_test, k=optimal_k)
    y_pred = []
    progress_per_loop = len(indices)
    with tqdm.trange(progress_per_loop, unit="row(s)", mininterval=0) as bar:
        bar.set_description(f"Predicting Progress: ")
        for i in bar:
            neighbors = []
            for j in range(len(indices[i])):
                neighbors.append(int(y_train[indices[i][j]]))
            y_pred.append(int(stats.mode(neighbors)[0]))
    time2 = time.perf_counter()
    snapshot = tracemalloc.take_snapshot()
    top_memory_stats = snapshot.statistics("traceback")
    memory_stat = top_memory_stats[0]
    tracemalloc.stop()
    print(f"Memory used: {round(memory_stat.size / 1024**2, 4)} MiB")
    print(f"Time on predicting testing data: {round(time2 - time1, 2)} seconds")

    # set the class label colume to object type so that the prediction will be savesd as integer
    testing_data = testing_data.astype({"class label": "object"})
    testing_data.iloc[:, 19:] = y_pred
    # save the prediction
    testing_data.to_csv("data-release/data2/test_predict.csv", index=False)

    # --------- First trial on building KNN brute force algorithm from scratch ---------
    # # initialize the f1 score list
    # f1_macro = []
    # f1_micro = []
    # calculated = False
    # distances = []
    # # Set the number of nearest neighbors from 2 to 0.1% of the training data which is 300
    # N = np.linspace(
    #     2, int(X_train.shape[0] * 0.001), num=(int(X_train.shape[0] * 0.001 - 1))
    # )
    # N = N.astype(int)
    # # loop through the number of nearest neighbors to get the f1 score of the training data and validation data
    # for n in N:
    #     # if n != 2, the distances are calculated in the first loop, then the distances are not calculated
    #     if n != 2:
    #         calculated = True
    #     # set the model
    #     model = KNN(n)
    #     # fit the data to the model
    #     model.fit(X_train, y_train)
    #     print(f"Getting {n} nearest neighbors...")
    #     # get the prediction of the validation data
    #     index, predict, distances = model.predict(X_valid, calculated, distances)
    #     # calculate the f1 score and save it to the list
    #     f1_macro.append(f1_score(y_valid, predict, average="macro"))
    #     f1_micro.append(f1_score(y_valid, predict, average="micro"))
    # # plot the f1 score of different number of nearest neighbors
    # plt.plot(f1_macro, label="F1 Macro")
    # plt.plot(f1_micro, label="F1 Micro")
    # plt.xticks(np.arange(len(N)), N)
    # plt.xlabel("Number of nearest neighbors")
    # plt.ylabel("F1 score")
    # plt.legend()
    # plt.show()
    # # Find the highest f1 score and the corresponding number of nearest neighbors
    # print(
    #     f"Maximum Macro F1 score is {max(f1_macro)} when k = {f1_macro.index(max(f1_macro))+2}"
    # )
    # print(
    #     f"Maximum Micro F1 score is {max(f1_micro)} when k = {f1_micro.index(max(f1_micro))+2}"
    # )
    # # find the optimal number of nearest neighbors
    # if max(f1_macro) >= max(f1_micro):
    #     optimal_k = f1_macro.index(max(f1_macro)) + 2
    # else:
    #     optimal_k = f1_micro.index(max(f1_micro)) + 2
