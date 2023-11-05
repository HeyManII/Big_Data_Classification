import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def cleaning_dirty_data(original_data):
    cleaned_data = original_data.dropna(axis=0, inplace=False)
    return cleaned_data


def normalize_data(original_data):
    # perform a max--min normalization to colume 1 to 19
    original_data.iloc[:, 1:19] = (
        original_data.iloc[:, 1:19] - original_data.iloc[:, 1:19].min()
    ) / (original_data.iloc[:, 1:19].max() - original_data.iloc[:, 1:19].min())
    return original_data


def tf_idf_calculation(data):
    tfidf = TfidfVectorizer()
    t1_tfidf = tfidf.fit_transform(data["T1"])
    t2_tfidf = tfidf.fit_transform(data["T2"])
    return t1_tfidf, t2_tfidf


if __name__ == "__main__":
    # reading the training data
    training_data = pd.read_csv("data-release/data1/training.csv")
    print(training_data.info())

    # cleaning empty cell inside the training_data
    cleaned_training_data = cleaning_dirty_data(training_data)
    print(cleaned_training_data.info())

    obama = cleaned_training_data[cleaned_training_data["TO"] == "obama"]
    microsoft = cleaned_training_data[cleaned_training_data["TO"] == "microsoft"]
    economy = cleaned_training_data[cleaned_training_data["TO"] == "economy"]
    palestine = cleaned_training_data[cleaned_training_data["TO"] == "palestine"]

    obama_t1_tfidf, obama_t2_tfidf = tf_idf_calculation(obama)
    microsoft_t1_tfidf, microsoft_t2_tfidf = tf_idf_calculation(microsoft)
    economy_t1_tfidf, economy_t2_tfidf = tf_idf_calculation(economy)
    palestine_t1_tfidf, palestine_t2_tfidf = tf_idf_calculation(palestine)

    # # Train SVM model for obama
    obama_train = pd.concat([obama["S1"], obama["S2"]], axis=1)
    # print(obama_t1_tfidf.shape)
    # print(obama_t2_tfidf.shape)
    # print(obama["S1"].shape)
    # print(obama["S2"].shape)
    # print(obama_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        obama_train, obama["class label"], test_size=0.2, random_state=42
    )
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
