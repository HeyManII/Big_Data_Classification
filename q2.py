import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import tqdm


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
    def predict(self, X_test):
        m = self.X_train.shape[0]
        n = X_test.shape[0]
        y_pred = []
        index = []

        # parameter to show the progress of the loop
        progress_per_loop = m
        distances = []
        # Calculating Euclidean distances
        for i in range(n):
            distance = []
            with tqdm.trange(progress_per_loop, unit="%", mininterval=0) as bar:
                bar.set_description(f"Testing Record {i}: ")
                for j in bar:
                    d = np.sqrt(
                        np.sum(np.square(X_test[i, 1:19] - self.X_train[j, 1:19]))
                    )
                    distance.append((d, y_train[j]))
            # sorting distances in ascending order
            distance = sorted(distance)
            distances.append(distance)

        # Getting k nearest neighbors
        print(f"Getting {self.k} nearest neighbors")
        for i in range(n):
            # Getting k nearest neighbors
            neighbors = []
            distance = distances[i]
            for item in range(self.k):
                neighbors.append(distance[item][1])
            index.append(int(X_test[i][0]))
            y_pred.append(stats.mode(neighbors)[0])
        return index, y_pred


# set the number of nearest neighbors testing for validation data
N = [3]
testing_no = 5

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
    X_valid = np.array(cleaned_validation_data.head(testing_no).iloc[:, 0:19])
    y_valid = np.array(cleaned_validation_data.head(testing_no).iloc[:, 19])

    f1_macro = []
    f1_micro = []
    for n in N:
        model = KNN(n)
        model.fit(X_train, y_train)
        index, predict = model.predict(X_valid)
        print(index, predict)
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
    testing_data = pd.read_csv("data-release/data2/test.csv")
    # replace "?" record with nan value
    cleaned_testing_data = data_cleaning(testing_data)
    # assign X_test
    X_test = np.array(cleaned_testing_data.head(testing_no).iloc[:, 0:19])

    # predict the class label of the testing data
    model = KNN(3)
    model.fit(X_train, y_train)
    index, predict = model.predict(X_test)
    # set the class label colume to object type so that the prediction will be savesd as integer
    testing_data = testing_data.astype({"class label": "object"})
    for i in range(X_test.shape[0]):
        print(index[i], predict[i])
        if index[i] == testing_data.iloc[i, 0]:
            testing_data.iloc[i, 19] = predict[i]
    testing_data.to_csv("data-release/data2/test_predict.csv", index=False)
