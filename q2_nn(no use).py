import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import seaborn as sns


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
        plt.title(f"Scatter Plot of Features {i} Distribution")
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


# nerual network [18 input] [36 hidden] [5 output]
class Neural_Network_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
            nn.Linear(36, 54),
            nn.ReLU(),
            nn.Linear(54, 54),
            nn.ReLU(),
            nn.Linear(54, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


if __name__ == "__main__":
    # check if GPU is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")
    # show the data distribution of the asstributes in training data
    # plot_distribution(training_data.iloc[:, 0:19])
    # remove rows with empty cells
    training_data = training_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_training_data = data_cleaning(training_data)
    # shuffle the rows of training data
    cleaned_training_data = cleaned_training_data.sample(frac=1).reset_index(drop=True)
    # plot the distribution of the attributes in training data after data processing
    # plot_distribution(cleaned_training_data)
    # split to training data and training label
    X_train = cleaned_training_data.iloc[:, 1:19]
    y_train = cleaned_training_data.iloc[:, 19:]

    # calculate the correlation matrix of the training data
    corr_matrix = np.corrcoef(np.array(X_train), rowvar=False)
    # plot the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=0.8)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True, fmt=".2f")
    plt.title("Correlation Matrix of X_train")
    plt.show()

    # plot the distribution of the class label after data processing
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(train_labels, train_counts)
    plt.title("Training Label Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Labels")
    plt.show()

    # transform the class label to one hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_train)
    y_train = ohe.transform(y_train)

    # reading the validation data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # remove rows with empty cells
    validation_data = validation_data.dropna(axis=0, inplace=False)
    # cleaning the training_data
    cleaned_validation_data = data_cleaning(training_data)
    # split to training data and training label
    X_valid = cleaned_validation_data.iloc[:, 1:19]
    y_valid = cleaned_validation_data.iloc[:, 19:]
    # transform the class label to one hot encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_valid)
    y_valid = ohe.transform(y_valid)

    # knn = KNeighborsClassifier(n_neighbors=2, metric="euclidean", algorithm="brute")
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_valid)
    # y = np.zeros((2, len(y_valid)))
    # y[0, :] = y_valid
    # y[1, :] = y_pred
    # print(y)
    # f1_macro = f1_score(y_valid, y_pred, average="macro")
    # f1_micro = f1_score(y_valid, y_pred, average="micro")
    # print(f"Macro F1 score: {f1_macro:.4f}")
    # print(f"Micro F1 score: {f1_micro:.4f}")

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_valid = torch.tensor(X_valid.values, dtype=torch.float)
    y_valid = torch.tensor(y_valid, dtype=torch.float)

    # training parameters
    n_epochs = 300
    batch_size = 128
    batches_per_epoch = len(X_train) // batch_size
    lr = 0.001

    # nerual network
    model = Neural_Network_Classification()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = -np.inf
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (
                    (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                )
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc))

        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_valid)
        ce = loss_fn(y_pred, y_valid)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_valid, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%"
        )

    # Restore best model
    model.load_state_dict(best_weights)

    # set model in evaluation mode and run through the validation set
    model.eval()
    y_pred = model(X_valid)
    y_pred = torch.argmax(y_pred, 1)
    y_pred = y_pred + 1
    # convert y_validate to numpy array
    Y_validate = ohe.inverse_transform(y_valid)
    # calculate F1 score
    f1_macro = f1_score(Y_validate, y_pred, average="macro")
    f1_micro = f1_score(Y_validate, y_pred, average="micro")
    print(f"Macro F1 score: {f1_macro:.4f}")
    print(f"Micro F1 score: {f1_micro:.4f}")

    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # --------- Second trial on building KNN ball tree algorithm with sklearn library ---------
    # K = np.linspace(
    #     2, int(X_train.shape[0] * 0.001), num=(int(X_train.shape[0] * 0.001 - 1))
    # )
    # K = K.astype(int)
    # f1_macro = []
    # f1_micro = []
    # tree = BallTree(X_train, metric="euclidean", leaf_size=60000)

    # # predict the class labels of the validation data
    # for k in K:
    #     print(k)
    #     distances, indices = tree.query(X_valid, k=k)
    #     print(indices)
    #     y_pred = []
    #     for i in range(len(indices)):
    #         neighbors = []
    #         for j in range(len(indices[i])):
    #             neighbors.append(y_train[indices[i][j]])
    #         y_pred.append(stats.mode(neighbors)[0])
    #     f1_macro.append(f1_score(y_valid, y_pred, average="macro"))
    #     f1_micro.append(f1_score(y_valid, y_pred, average="micro"))

    # # plot the f1 score of different number of nearest neighbors
    # plt.plot(f1_macro, label="F1 Macro")
    # plt.plot(f1_micro, label="F1 Micro")
    # plt.xlabel("Number of nearest neighbors")
    # plt.xticks(np.arange(len(K)), K)
    # plt.ylabel("F1 score")
    # plt.legend()
    # plt.show()

    # print(
    #     f"Maximum Macro F1 score is {max(f1_macro)} when k = {f1_macro.index(max(f1_macro))+2}"
    # )
    # print(
    #     f"Maximum Micro F1 score is {max(f1_micro)} when k = {f1_micro.index(max(f1_micro))+2}"
    # )

    # # reading the testing data
    # testing_data = pd.read_csv("data-release/data2/test.csv")
    # # replace "?" record with nan value
    # cleaned_testing_data = data_cleaning(testing_data)
    # # assign X_test
    # X_test = np.array(cleaned_testing_data.iloc[:, 0:19])

    # # find the optimal number of nearest neighbors
    # if max(f1_macro) >= max(f1_micro):
    #     optimal_k = f1_macro.index(max(f1_macro)) + 2
    # else:
    #     optimal_k = f1_micro.index(max(f1_micro)) + 2

    # distances, indices = tree.query(X_test, k=optimal_k)
    # y_pred = []
    # for i in range(len(indices)):
    #     neighbors = []
    #     for j in range(len(indices[i])):
    #         neighbors.append(y_train[indices[i][j]])
    #     y_pred.append(stats.mode(neighbors)[0])
    # testing_data[:, 19] = y_pred
    # # set the class label colume to object type so that the prediction will be savesd as integer
    # testing_data = testing_data.astype({"class label": "object"})
    # # save the prediction
    # testing_data.to_csv("data-release/data2/test_predict.csv", index=False)

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
