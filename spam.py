#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:18:59 2022

@author: vamsi
"""

import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('spam.csv',encoding='latin-1')
print(data.head())

del data["Unnamed: 2"]
del data["Unnamed: 3"]
del data["Unnamed: 4"]

print(data.head())

data.rename(columns = {'v1':'label', 'v2':'messages'}, inplace = True)
print(data.head())

data.describe()

data["length"] = data["messages"].apply(len)
data.sort_values(by='length', ascending=False).head(10)

data.hist(column = 'length', by ='label',figsize=(12,4), bins = 5)

message = "Hi everyone!!! it is a pleasure to meet you."
message_not_punc = []
for punctuation in message:
    if punctuation not in string.punctuation:
           message_not_punc.append(punctuation)
# Join the characters again to form the string.
message_not_punc = ''.join(message_not_punc)
print(message_not_punc)

from nltk.corpus import stopwords

# Remove any stopwords for remove_punc, but first we should to transform this into the list.

message_clean = list(message_not_punc.split(" "))

# Remove any stopwords
i = 0

while i <= len(message_clean):
    for mess in message_clean:
        if mess.lower() in stopwords.words('english'):
            message_clean.remove(mess)
    i =i +1
    print(message_clean)


def transform_message(message):
    message_not_punc = [] # Message without punctuation
    i = 0
    for punctuation in message:
        if punctuation not in string.punctuation:
            message_not_punc.append(punctuation)
    # Join words again to form the string.
    message_not_punc = ''.join(message_not_punc) 

    # Remove any stopwords for message_not_punc, but first we should     
    # to transform this into the list.
    message_clean = list(message_not_punc.split(" "))
    while i <= len(message_clean):
        for mess in message_clean:
            if mess.lower()  in stopwords.words('english'):
                message_clean.remove(mess)
        i =i +1
    return  message_clean

data['messages'].head(5).apply(transform_message)


from sklearn.feature_extraction.text import CountVectorizer

vectorization = CountVectorizer(analyzer = transform_message )

X = vectorization.fit(data['messages'])

X_transform = X.transform([data['messages']])

print(X_transform)


tfidf_transformer = TfidfTransformer().fit(X_transform)

X_tfidf = tfidf_transformer.transform(X_transform)
print(X_tfidf.shape)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['messages'], test_size=0.30, random_state = 50)    

clf = SVC(kernel='linear').fit(X_train, y_train)


predictions = clf.predict(X_test)
print('predicted', predictions)

from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
