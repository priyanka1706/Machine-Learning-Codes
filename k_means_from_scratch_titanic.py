import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn import preprocessing


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        # The tol value is our tolerance, which will allow us to say we're optimized if the centroid
        # is not moving more than the tolerance value.
        self.max_iter = max_iter
        # The max_iter value is to limit the number of cycles we're willing to run.

    def fit(self, data):
        self.centroids = {}

        for i in range (self.k):
            self.centroids[i] = data[i]

        for i in range (self.max_iter):
            self.classifications = {}
            for i in range (self.k):
                self.classifications[i] = []
                # While here, we start with empty classifications, and then create two dict keys (by iterating through range of self.k)
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # this is subtractubg  from freatureset akk the k centroids
                # 9th element is distance from 9th centroid, 1dt from one so on
                classification = distances.index(min(distances))
                # finding index value of minimum of dustances
                self.classifications[classification].append(featureset)
                # iterate through our features, calculate distances of the features to the current centroids, and classify them

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                # finding mean of all the features in a particular class

            # we're going to need to create the new centroids, as well as measuring the movement of the centroids.
            # If that movement is less than our tolerance (self.tol), then we're all set.
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
            # This is basically new_column = list(map( function_to_map, parameter1, parameter2, ... ))
    return df

df = handle_non_numerical_data(df)
# print(df.head())
df.drop(['ticket','home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)
correct = 0
for i in range (len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct+=1
print(correct/len(X))