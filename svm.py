# The objective of the Support Vector Machine is to find the best splitting boundary between data.
# In two dimensional space, you can think of this like the best fit line that divides your dataset.

# What the Support Vector Machine aims to do is, one time, generate the "best fit" line (but actually a plane,
# and even more specifically a hyperplane!) that best divides the data. Once this hyperplane is discovered,
# we refer to it as a decision boundary. We do this, because, this is the boundary between being one class or another.

# First, we find the support vectors
# Once you find the support vectors, you want to create lines that are maximally separated between each other.
# From here, we can easily find the decision boundary by taking the total width, and dividing by 2.

# It is worth noting, of course, that this method of learning is only going to work natively on linearly-separable data.
# Also SVM is a binary classifier, in the sense that it draws a line to divide two groups

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)

# Same accuracy as KNN, only significantly faster
# The Support Vector Machine, in general, handles pointless data better than the K Nearest Neighbors algorithm,
# and definitely will handle outliers better, but, in this example, the meaningless data is still very misleading for us.