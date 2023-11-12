import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy


def cleaning_dirty_data(original_data):
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    return cleaned_data


def normalize_data(original_data):
    # perform a max--min normalization to colume 1 to 19
    original_data.iloc[:, 5:6] = (
        original_data.iloc[:, 5:6] - original_data.iloc[:, 5:6].min()
    ) / (original_data.iloc[:, 5:6].max() - original_data.iloc[:, 5:6].min())
    return original_data


def tf_idf_calculation(data):
    tfidf1 = TfidfVectorizer()
    t1_tfidf = tfidf1.fit_transform(data.iloc[:, 1])
    t1_tfidf_df = pd.DataFrame(
        t1_tfidf.toarray(), columns=tfidf1.get_feature_names_out()
    )
    tfidf2 = TfidfVectorizer()
    t2_tfidf = tfidf2.fit_transform(data.iloc[:, 2])
    t2_tfidf_df = pd.DataFrame(
        t2_tfidf.toarray(), columns=tfidf2.get_feature_names_out()
    )
    return t1_tfidf_df, t2_tfidf_df


# nerual network [18 input] [36 hidden] [5 output]
class Neural_Network_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(49169, 49169),
            nn.ReLU(),
            nn.Linear(49169, 49169),
            nn.ReLU(),
            nn.Linear(49169, 5),
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
    training_data = pd.read_csv("data-release/data1/training.csv")

    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    cleaned_training_data = normalize_data(cleaned_training_data)
    Y = cleaned_training_data.iloc[:, 7]
    Y = np.array(Y)

    t1_tfidf, t2_tfidf = tf_idf_calculation(cleaned_training_data)
    cleaned_training_data = cleaned_training_data.iloc[:4].replace("obama", int(0))
    cleaned_training_data = cleaned_training_data[:4].replace("microsoft", int(1))
    cleaned_training_data = cleaned_training_data[:4].replace("economy", int(2))
    cleaned_training_data = cleaned_training_data[:4].replace("palestine", int(3))

    # split training data into X and y
    # Train SVM model for obama
    X = pd.concat([t1_tfidf, t2_tfidf, cleaned_training_data.iloc[:, 4:6]], axis=1)
    # split data into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X_train = X_train.astype(float)
    X_valid = X_valid.astype(float)
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    X_valid = torch.tensor(X_valid.values, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float)

    # nerual network
    model = Neural_Network_Classification()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training parameters
    n_epochs = 1000
    batch_size = 49169
    batches_per_epoch = len(X_train) // batch_size

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
                y_batch = Y_train[start : start + batch_size]
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
        ce = loss_fn(y_pred, Y_valid)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(Y_valid, 1)).float().mean()
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
    # Y_validate = ohe.inverse_transform(Y_valid)
    # calculate F1 score
    f1_macro = f1_score(Y_valid, y_pred, average="macro")
    f1_micro = f1_score(Y_valid, y_pred, average="micro")
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
