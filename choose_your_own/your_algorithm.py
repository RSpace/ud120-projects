#!/usr/bin/python

from time import time
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


#
# K Nearest Neighbour
#

# Initialize classifier
classifier = KNeighborsClassifier()

# Train classifier with training data
t0 = time()
classifier.fit(features_train, labels_train)
print("K Nearest Neighbour training time:", round(time()-t0, 3), "s")

# Determine accuracy of classifier with test data
t1 = time()
accuracy = classifier.score(features_test, labels_test)
print("Scoring time:", round(time()-t1, 3), "s")
print("Accuracy of K Nearest Neighbour:", accuracy)

# Make classification prediction on test data
t2 = time()
predictions = classifier.predict(features_test)
print("K Nearest Neighbour prediction time:", round(time()-t2, 3), "s")


#
# Adaboost
#

# Initialize classifier
classifier = AdaBoostClassifier(n_estimators=100, random_state=0)

# Train classifier with training data
t0 = time()
classifier.fit(features_train, labels_train)
print("AdaBoost training time:", round(time()-t0, 3), "s")

# Determine accuracy of classifier with test data
t1 = time()
accuracy = classifier.score(features_test, labels_test)
print("Scoring time:", round(time()-t1, 3), "s")
print("Accuracy of AdaBoost:", accuracy)

# Make classification prediction on test data
t2 = time()
predictions = classifier.predict(features_test)
print("AdaBoost prediction time:", round(time()-t2, 3), "s")


#
# Random Forest
#

# Initialize classifier
classifier = RandomForestClassifier()

# Train classifier with training data
t0 = time()
classifier.fit(features_train, labels_train)
print("RandomForest training time:", round(time()-t0, 3), "s")

# Determine accuracy of classifier with test data
t1 = time()
accuracy = classifier.score(features_test, labels_test)
print("Scoring time:", round(time()-t1, 3), "s")
print("Accuracy of RandomForest:", accuracy)

# Make classification prediction on test data
t2 = time()
predictions = classifier.predict(features_test)
print("RandomForest prediction time:", round(time()-t2, 3), "s")


### visualization code (prettyPicture) to show you the decision boundary
# prettyPicture(classifier, features_test, labels_test)
