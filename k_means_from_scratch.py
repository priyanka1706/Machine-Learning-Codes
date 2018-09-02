import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


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


X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])
colors = 10*["g","r","c","b","k"]

# plt.scatter(X[:,0], X[:, 1], s=100, c='b')
# plt.show()

clf = K_Means()
clf.fit(X)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker = 'o', color = 'k', linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=100, linewidths=5)

unknowns = ([[1,3],
             [8,9],
             [0,3],
             [5,4],
             [6,4]])
for unknown in unknowns:
    classification = clf.predict(unknown)
    print('classification for ',unknown, 'is',classification)
    plt.scatter(unknown[0],unknown[1],marker='*',color=colors[classification], s=50, linewidths=5)

plt.show()