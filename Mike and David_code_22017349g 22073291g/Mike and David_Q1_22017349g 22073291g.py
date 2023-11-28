import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import tracemalloc
import time

nltk.download('words')
nltk.download('punkt')

def preprocess_text_csv(df, text_column):
    stop_words = set(stopwords.words('english'))
    pd.options.mode.chained_assignment = None  

    if text_column == "S":
        df[text_column].fillna("isnull",inplace=True) #replace the empty value with string "isnull"
    else:
        df[text_column].fillna("isnull",inplace=True)
        # Convert text to lowercase
        df[text_column] = df[text_column].str.lower()
        # Remove common most fequent words in all classes
        df[text_column] = df[text_column].apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in ['obama', 'economy','microsoft','barack','Â','quot','s','ÂÂ']))
        # Remove punctuation
        df[text_column] = df[text_column].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
        # Remove stopwords
        df[text_column] = df[text_column].apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))

    return df[text_column]

# apply the L2 normalization
def normalize_sparse_matrix(sparse_matrix):
    normalized_sparse_matrix = csr_matrix(sparse_matrix.shape)

    for i in range(sparse_matrix.shape[0]):
        row_start = sparse_matrix.indptr[i] 
        row_end = sparse_matrix.indptr[i + 1]
        row_data = sparse_matrix.data[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]].copy()
        row_indices = sparse_matrix.indices[row_start:row_end]
        squared_sum = sum(row_data ** 2)
        norm = squared_sum ** 0.5 if squared_sum != 0 else 1
        
        row_data = row_data / norm
        if((row_end - row_start) == sparse_matrix.shape[1]):
          row_data = csr_matrix(row_data)
        else:
          value_index = 0
          tmp_row_data = np.zeros(sparse_matrix.shape[1])
          for j in (row_indices):
            tmp_row_data[j] = row_data[value_index]
            value_index = value_index + 1
        
          row_data = csr_matrix(tmp_row_data)

        normalized_sparse_matrix[i] = row_data
    
    normalized_sparse_matrix.indices = sparse_matrix.indices
    normalized_sparse_matrix.indptr = sparse_matrix.indptr
    return normalized_sparse_matrix


if __name__ == "__main__":
    data = pd.read_csv('./data-release/data1/training.csv')
    validation_data = pd.read_csv('./data-release/data1/validation.csv')
    testing_data = pd.read_csv('./data-release/data1/test.csv')


    pd.options.mode.chained_assignment = None  

    train_data = data.dropna() # Drop rows with missing values
    validation_data = validation_data.dropna()

    train_data.drop_duplicates(inplace=True) #remove the duplicates rows of the dataset
    validation_data.drop_duplicates(inplace=True)

    text_column = "T1"
    train_data["T1"] = preprocess_text_csv(train_data, text_column)
    validation_data["T1"] = preprocess_text_csv(validation_data, text_column)
    testing_data["T1"] = preprocess_text_csv(testing_data, text_column)
    print("finished T1")

    text_column = "T2"
    train_data["T2"] = preprocess_text_csv(train_data, text_column)
    validation_data["T2"] = preprocess_text_csv(validation_data, text_column)
    testing_data["T2"] = preprocess_text_csv(testing_data, text_column)
    print("finished T2")

    text_column = "S"
    train_data["S"] = preprocess_text_csv(train_data, text_column)
    validation_data["S"] = preprocess_text_csv(validation_data, text_column)
    testing_data["S"] = preprocess_text_csv(testing_data, text_column)
    print("finished S")


    X_train = train_data[['id','T1', 'T2', 'S', 'TO','S1','S2']]  
    y_train = train_data['class label'] 

    X_validation = validation_data[['id','T1', 'T2', 'S', 'TO','S1','S2']]  
    y_validation  = validation_data['class label']

    X_test = testing_data[['id','T1', 'T2', 'S', 'TO','S1','S2']]  
    y_test  = testing_data['class label']

    # transform the T1 and T2 data into Tfidf format 
    vectorizer = TfidfVectorizer()
    X_train_tfidf_t1 = vectorizer.fit_transform(X_train['T1'])  # Transform the T1 text column
    X_validation_tfidf_t1 = vectorizer.transform(X_validation['T1'])
    X_test_tfidf_t1 = vectorizer.transform(X_test['T1'])

    X_train_tfidf_t2 = vectorizer.fit_transform(X_train['T2'])  # Transform the T2 text column
    X_validation_tfidf_t2 = vectorizer.transform(X_validation['T2'])
    X_test_tfidf_t2 = vectorizer.transform(X_test['T2'])

    # transform the category columns into numerical data using one hot encoder
    cat_features = ['S','TO']
    encoder = OneHotEncoder(handle_unknown = 'ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_features])
    X_validation_cat = encoder.transform(X_validation[cat_features])
    X_test_cat = encoder.transform(X_test[cat_features])

    # concatenate sparse matrices with the same number of rows (horizontal concatenation)
    num_features = ['S1', 'S2']
    X_train_combined = hstack([
        X_train_tfidf_t1,
        X_train_tfidf_t2,
        X_train_cat,
        X_train[num_features]
    ])

    X_validation_combined = hstack([
        X_validation_tfidf_t1,
        X_validation_tfidf_t2,
        X_validation_cat,
        X_validation[num_features]
    ])


    X_test_combined = hstack([
        X_test_tfidf_t1,
        X_test_tfidf_t2,
        X_test_cat,
        X_test[num_features]
    ])

    #Normalize the dataset before machine learning
    X_train = normalize_sparse_matrix(csr_matrix(X_train_combined))
    X_validation = normalize_sparse_matrix(csr_matrix(X_validation_combined))
    X_test = normalize_sparse_matrix(csr_matrix(X_test_combined))

    
    # training model with four kernel functions
    kernel_func = ['rbf', 'linear', 'sigmoid', 'poly']
    for i in kernel_func:
        # monitor the memory used 
        tracemalloc.start(20)
        # monitor the time used for building ball tree
        time1 = time.perf_counter()
        # Model training
        SVC_classifier = SVC(kernel=f'{i}').fit(X_train,y_train)
        time2 = time.perf_counter()
        snapshot_training = tracemalloc.take_snapshot()
        top_memory_stats = snapshot_training.statistics("traceback")
        memory_stat = top_memory_stats[0]
        print(
            f"Memory used on training SVM with kernel function {i}: {round(memory_stat.size / 1024**2, 4)} MiB"
        )
        print(f"Time on training SVM with kernel function {i} : {round(time2 - time1, 5)} seconds")
        tracemalloc.stop()

        SVC_prediction = SVC_classifier.predict(X_validation)
        # print the score data
        print(f"SVM with kernel function {i} F1-score (micro)",f1_score(y_validation, SVC_prediction, average='micro'))
        print(f"SVM with kernel function {i} F1-score (marco)",f1_score(y_validation, SVC_prediction, average='macro'))
    

    # since kernel Polynomial obtained the highest f1 score, hence it would choose its prediction to export the result
    SVC_classifier = SVC(kernel='poly').fit(X_train,y_train)
    tracemalloc.start(20)
    time1 = time.perf_counter()
    SVC_prediction = SVC_classifier.predict(X_test)
    time2 = time.perf_counter()
    print(f"Time on testing data by SVM with kernel function poly : {round(time2 - time1, 5)} seconds")
    testing_data_export = pd.read_csv("./data-release/data1/test.csv")
    testing_data_export.iloc[:, 7:] = SVC_prediction
    testing_data_export = testing_data_export.astype({"class label": int})
    testing_data_export.to_csv("./data-release/data1/test_predict.csv", index=False)
