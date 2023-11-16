import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack, coo_matrix, csr_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords, words
import string
import re
from nltk.tokenize import word_tokenize
#import enchant 

nltk.download('words')
nltk.download('punkt')

#english_dict = enchant.Dict("en_US")

def preprocess_text_csv(df, text_column):
    stop_words = set(stopwords.words('english'))
    pd.options.mode.chained_assignment = None  

    if text_column == "S":
        df[text_column].fillna("isnull",inplace=True) #replace the empty value with string "isnull"
    else:
        # Keep a-z & 0-9 data
        #df[text_column] = df[text_column].apply(lambda text: re.sub(r"[^A-Za-z0-9\s$:?\"!]", "", text))
        #print("df[text_column] after word_list", df[text_column])

        # Convert text to lowercase
        df[text_column] = df[text_column].str.lower()
        # check if the word is a real word
        #df[text_column] = df[text_column].apply(lambda text: ' '.join(word for word in word_tokenize(text) if english_dict.check(word) or re.search(r"[$?!:\"]", text)))
        #print("df[text_column] after word_list", df[text_column])

        # Remove common most fequent words in all classes
        df[text_column] = df[text_column].apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in ['obama', 'economy','microsoft','barack','Â','quot']))
        # Remove punctuation
        #df[text_column] = df[text_column].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

        # Remove stopwords
        df[text_column] = df[text_column].apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in stop_words))

    print("df",df[text_column])
    return df[text_column]

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


# Testing
if __name__ == "__main__":
    data = pd.read_csv('./data-release/data1/training.csv')
    pd.options.mode.chained_assignment = None  

    train_data = data.dropna() # Drop rows with missing values 
    train_data.drop_duplicates(inplace=True) #remove the duplicates rows of the dataset

    text_column = "T1"
    train_data["T1"] = preprocess_text_csv(train_data, text_column)
    print("finished T1")

    text_column = "T2"
    train_data["T2"] = preprocess_text_csv(train_data, text_column)
    print("finished T2")

    text_column = "S"
    train_data["S"] = preprocess_text_csv(train_data, text_column)
    print("finished S")


    X = train_data[['id','T1', 'T2', 'S', 'TO','S1','S2']]  
    y = train_data['class label'] 

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    # transform the T1 and T2 data into Tfidf format with remove the stopword, punctuation
    vectorizer = TfidfVectorizer()
    X_train_tfidf_t1 = vectorizer.fit_transform(X_train['T1'])  # Transform the T1 text column
    X_test_tfidf_t1 = vectorizer.transform(X_validation['T1'])


    X_train_tfidf_t2 = vectorizer.fit_transform(X_train['T2'])  # Transform the T2 text column
    X_test_tfidf_t2 = vectorizer.transform(X_validation['T2'])

    # transform the category columns into numerical data using one hot encoder
    cat_features = ['S','TO']
    encoder = OneHotEncoder(handle_unknown = 'ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_features])
    X_test_cat = encoder.transform(X_validation[cat_features])

    # concatenate sparse matrices with the same number of rows (horizontal concatenation)
    num_features = ['S1', 'S2']
    X_train_combined = hstack([
        X_train_tfidf_t1,
        X_train_tfidf_t2,
        X_train_cat,
        X_train[num_features]
    ])

    X_test_combined = hstack([
        X_test_tfidf_t1,
        X_test_tfidf_t2,
        X_test_cat,
        X_validation[num_features]
    ])

    # need to be implemented if the normliazer doesn't support sparse matrix
    #regular_array_train = coo_matrix((X_train_combined.data,(X_train_combined.row,X_train_combined.col)),shape=(max(X_train_combined.row)+1,max(X_train_combined.col)+1)).toarray()
    #regular_array_test = coo_matrix((X_test_combined.data,(X_test_combined.row,X_test_combined.col)),shape=(max(X_test_combined.row)+1,max(X_test_combined.col)+1)).toarray()


    #Normalize the dataset before machine learning
    X_train = normalize_sparse_matrix(csr_matrix(X_train_combined))
    X_test = normalize_sparse_matrix(csr_matrix(X_test_combined))

    
    # Mechine learning 
    SVC_classifier = SVC().fit(X_train,y_train)
    SVC_classifier_prediction = SVC_classifier.predict(X_test)

    # print the score data
    print(accuracy_score(y_validation, SVC_classifier_prediction))
    print(f1_score(y_validation, SVC_classifier_prediction, average='micro'))
    print(f1_score(y_validation, SVC_classifier_prediction, average='macro'))

    #train the result
    # testing_result = hstack([
    # coo_matrix(np.array(X_validation['id']).reshape(X_validation['id'].shape[0],1)),
    # coo_matrix(SVC_classifier_prediction.reshape(SVC_classifier_prediction.shape[0],1))])

    # testing_result = pd.DataFrame(testing_result.toarray())

    #testing_result.astype(int).to_csv('./data-release/q1_prediction.csv', index=False) 
        
