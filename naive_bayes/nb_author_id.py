#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
#########################################################

# Initialize classifier
classifier = GaussianNB()

# Train classifier with training data
t0 = time()
classifier.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

# Determine accuracy of classifier with test data
t1 = time()
accuracy = classifier.score(features_test, labels_test)
print("Scoring time:", round(time()-t1, 3), "s")
print("Accuracy of Naive Bayes author identifier:", accuracy)

# Make classification prediction on test data
t2 = time()
classifier.predict(features_test)
print("Prediction time:", round(time()-t2, 3), "s")
