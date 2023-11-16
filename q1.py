import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack, coo_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.tokenize import word_tokenize


def preprocess_text_csv(df, text_column):
    nltk.download("punkt")
    stop_words = set(stopwords.words("english"))
    pd.options.mode.chained_assignment = None

    if text_column == "S":
        df[text_column].fillna(
            "isnull", inplace=True
        )  # replace the empty value with string "isnull"
    else:
        # Remove punctuation
        df[text_column] = df[text_column].apply(
            lambda text: text.translate(str.maketrans("", "", string.punctuation))
        )

        # Convert text to lowercase
        df[text_column] = df[text_column].str.lower()

        # Remove stopwords
        df[text_column] = df[text_column].apply(
            lambda text: " ".join(
                word for word in word_tokenize(text) if word not in stop_words
            )
        )

    print("df", df[text_column])
    return df[text_column]


# Testing
if __name__ == "__main__":
    data = pd.read_csv("./data-release/data1/training.csv")
    pd.options.mode.chained_assignment = None

    train_data = data.dropna()  # Drop rows with missing values
    train_data.drop_duplicates(
        inplace=True
    )  # remove the duplicates rows of the dataset

    text_column = "T1"
    train_data["T1"] = preprocess_text_csv(train_data, text_column)

    text_column = "T2"
    train_data["T2"] = preprocess_text_csv(train_data, text_column)

    text_column = "S"
    train_data["S"] = preprocess_text_csv(train_data, text_column)

    X = train_data[["T1", "T2", "S", "TO", "S1", "S2"]]
    y = train_data["class label"]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # transform the T1 and T2 data into Tfidf format with remove the stopword, punctuation
    vectorizer = TfidfVectorizer()
    X_train_tfidf_t1 = vectorizer.fit_transform(
        X_train["T1"]
    )  # Transform the T1 text column
    X_test_tfidf_t1 = vectorizer.transform(X_validation["T1"])

    X_train_tfidf_t2 = vectorizer.fit_transform(
        X_train["T2"]
    )  # Transform the T2 text column
    X_test_tfidf_t2 = vectorizer.transform(X_validation["T2"])

    # transform the category columns into numerical data using one hot encoder
    cat_features = ["S", "TO"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train_cat = encoder.fit_transform(X_train[cat_features])
    X_test_cat = encoder.transform(X_validation[cat_features])

    # concatenate sparse matrices with the same number of rows (horizontal concatenation)
    num_features = ["S1", "S2"]
    X_train_combined = hstack(
        [X_train_tfidf_t1, X_train_tfidf_t2, X_train_cat, X_train[num_features]]
    )

    X_test_combined = hstack(
        [X_test_tfidf_t1, X_test_tfidf_t2, X_test_cat, X_validation[num_features]]
    )

    # need to be implemented if the normliazer doesn't support sparse matrix
    # regular_array_train = coo_matrix((X_train_combined.data,(X_train_combined.row,X_train_combined.col)),shape=(max(X_train_combined.row)+1,max(X_train_combined.col)+1)).toarray()
    # regular_array_test = coo_matrix((X_test_combined.data,(X_test_combined.row,X_test_combined.col)),shape=(max(X_test_combined.row)+1,max(X_test_combined.col)+1)).toarray()

    # Normalize the dataset before machine learning
    X_train = Normalizer().fit_transform(X_train_combined)
    X_test = Normalizer().transform(X_test_combined)

    # Mechine learning
    SVC_classifier = SVC().fit(X_train, y_train)
    SVC_classifier_prediction = SVC_classifier.predict(X_test)

    # print the score data
    print(accuracy_score(y_validation, SVC_classifier_prediction))
    print(f1_score(y_validation, SVC_classifier_prediction, average="micro"))
    print(f1_score(y_validation, SVC_classifier_prediction, average="macro"))
