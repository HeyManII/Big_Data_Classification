import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(18, 36)
        self.act = nn.ReLU()
        self.output = nn.Linear(36, 5)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data2/training.csv")
    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    # normalizing 18 columns excluding the id and the class label
    normalized_training_data = normalize_data(cleaned_training_data)

    # split data into X and y
    X = cleaned_training_data.iloc[:, 1:19]
    Y = cleaned_training_data.iloc[:, 19:]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Y)
    Y = ohe.transform(Y)

    # nerual network [18 input] [36 hidden] [5 output]
    model = Multiclass()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X = X.astype(float)
    X = torch.tensor(X.values, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=0.7, shuffle=True
    )

    # training parameters
    n_epochs = 50
    batch_size = 5
    batches_per_epoch = len(X) // batch_size

    best_acc = -np.inf  # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

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
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
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
