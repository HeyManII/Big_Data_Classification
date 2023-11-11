import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack, coo_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import re

'''
def process_text(text):
    nltk.download('stopwords')
    #1 clear the punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    #2 clear the stopwords
    clean_words = [word for word in nopunc if word.lower() not in stopwords.words('english')]
    #3 return value
    return clean_words
'''

# Testing
if __name__ == "__main__":
    data = pd.read_csv('./data-release/data1/training.csv')
    pd.options.mode.chained_assignment = None  

    train_data = data.dropna() # Drop rows with missing values 
    train_data.drop_duplicates(inplace=True) #remove the duplicates rows of the dataset

    '''
    print("before",train_data['T1'].head())
    pd.options.mode.chained_assignment = None  
    train_data['T1'] = train_data['T1'].apply(process_text)
    train_data['T2'] = train_data['T2'].apply(process_text)
    print("after",train_data['T1'].head())
    '''

    X = train_data[['T1', 'T2', 'S', 'TO','S1','S2']]  
    y = train_data['class label'] 

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    # transform the T1 and T2 data into Tfidf format with remove the stopword, punctuation
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    X_train_tfidf_t1 = vectorizer.fit_transform(X_train['T1'].apply(lambda x: re.sub(r'[^\w\s]', '', x)))  # Transform the T1 text column
    X_train_tfidf_t2 = vectorizer.transform(X_train['T2'].apply(lambda x: re.sub(r'[^\w\s]', '', x)))  # Transform the T2 text column

    X_test_tfidf_t1 = vectorizer.transform(X_validation['T1'].apply(lambda x: re.sub(r'[^\w\s]', '', x)))
    X_test_tfidf_t2 = vectorizer.transform(X_validation['T2'].apply(lambda x: re.sub(r'[^\w\s]', '', x)))

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
    X_train = Normalizer().fit_transform(X_train_combined)
    X_test = Normalizer().transform(X_test_combined)

    
    # Mechine learning 
    SVC_classifier = SVC().fit(X_train,y_train)
    SVC_classifier_prediction = SVC_classifier.predict(X_test)

    # print the score data
    print(accuracy_score(y_validation, SVC_classifier_prediction))
    print(f1_score(y_validation, SVC_classifier_prediction, average='micro'))
    print(f1_score(y_validation, SVC_classifier_prediction, average='macro'))
    
