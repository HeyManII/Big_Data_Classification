import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
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
            nn.Linear(54, 18),
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
    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    # normalizing 18 columns excluding the id and the class label
    normalized_training_data = normalize_data(cleaned_training_data)

    # reading the training data
    validation_data = pd.read_csv("data-release/data2/validation.csv")
    # cleaning empty cell inside the training_data
    cleaned_validation_data = cleaning_dirty_data(training_data)
    # normalizing 18 columns excluding the id and the class label
    normalized_validation_data = normalize_data(cleaned_validation_data)

    # split training data into X and y
    X_train = cleaned_training_data.iloc[:, 1:19]
    Y_train = cleaned_training_data.iloc[:, 19:]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Y_train)
    Y_train = ohe.transform(Y_train)

    # split validation data into X and y
    X_validate = cleaned_validation_data.iloc[:, 1:19]
    Y_validate = cleaned_validation_data.iloc[:, 19:]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Y_validate)
    Y_validate = ohe.transform(Y_validate)

    # nerual network
    model = Neural_Network_Classification()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X_train = X_train.astype(float)
    X_train = torch.tensor(X_train.values, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float)
    X_validate = X_validate.astype(float)
    X_validate = torch.tensor(X_validate.values, dtype=torch.float)
    Y_validate = torch.tensor(Y_validate, dtype=torch.float)

    # training parameters
    n_epochs = 200
    batch_size = 20
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
        y_pred = model(X_validate)
        ce = loss_fn(y_pred, Y_validate)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(Y_validate, 1)).float().mean()
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
