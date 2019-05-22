#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################


# Initialize classifier
classifier = DecisionTreeClassifier(min_samples_split=40)

# Use only 1% of data
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

print("Number of features:", len(features_train[0]))

# Train classifier with training data
t0 = time()
classifier.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

# Determine accuracy of classifier with test data
t1 = time()
accuracy = classifier.score(features_test, labels_test)
print("Scoring time:", round(time()-t1, 3), "s")
print("Accuracy of Decision Tree author identifier:", accuracy)

# Make classification prediction on test data
t2 = time()
predictions = classifier.predict(features_test)
print("Prediction time:", round(time()-t2, 3), "s")
