# K Nearest Neighbors is a simple and effective machine learning classification algorithm overall.
# K is a number you can choose, and then neighbors are the data points from known data.
# We're looking for any number of the "nearest" neighbors.
# Let's say K = 3, so then we're looking for the two closest neighboring points.

# Note that, due to the nature of the vote, you will likely want to use an odd number for K,
# otherwise you may find yourself in a 50/50 split situation.

# The main issue with this objective is that, per datapoint, you have to compare it to every single datapoint to get the distances,
# thus the algorithm just doesn't scale well, despite being fairly reliable accuracy-wise.

#https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

import numpy as np
from sklearn import  preprocessing, model_selection, neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

eg_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
eg_measures = eg_measures.reshape(len(eg_measures),-1)
prediction = clf.predict(eg_measures)
print(prediction)