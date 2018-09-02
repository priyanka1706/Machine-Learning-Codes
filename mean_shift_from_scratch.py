import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')


class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        # So the plan here is to create a massive radius, but make that radius go in steps, like bandwidths,
        # or a bunch of radiuses with different lengths, which we'll call steps. If a featureset is in the closest radius,
        # it will have a much higher "weight" than one much further away.

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis = 0 )
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm/self.radius_norm_step
            # If the user hasn't hard-coded the radius, then we're going to find the "center" of ALL of the data.
            # Then, we will take the norm of that data, then we say each radius with self.radius is basically the full data-length,
            # divided by how many steps we wanted to have.

            # were basically making a huge radius, and penaliszing for the further away you are in it

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]
            # Recall the method for Mean Shift is:
            # 1. Make all datapoints centroids
            # 2.Take mean of all featuresets within centroid's radius, setting this mean as new centroid.
            # 3.Repeat step 2 until convergence.

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
                    #if np.linalg.norm(featureset - centroid) < self.radius:
                    #    in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weight_index = int(distance/self.radius)
                    # lesser the steps away it is radius wise, higher its weight should be
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add
                # The weights list is just a simple list that we'll take how many "radius steps" a featureset is from
                # the centroid, take those # of steps, treating them as index values for the weight list. Iterating through
                # the features, we calculate distances, add weights, then add the "weighted" number of centroids to the in_bandwidth.

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            # uniques variable, which tracks the sorted list of all known centroids.
            # set prevents duplicates

            # With this method, however, it is highly likely that we have centroids that are extremely close,
            # but not identical. We want to merge these too, foe convergence.
            to_pop = []
            for i in uniques:
                for ii in [i for i in uniques]:
                    if i==ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        #here if the centoids are within one radius step of each other, we are converging them
                        # print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass


            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i]=[]

        for featureset in data:
            distances = [np.linalg.norm(featureset-centroids[centroid]) for centroid in centroids]
            classification = (distances.index(min(distances)))
            self.classifications[classification].append(featureset)

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification




X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

#X, y = make_blobs(n_samples=15, centers=3, n_features=2)

# plt.scatter(X[:,0], X[:,1], s=90, c='b')
# plt.show()

colors = 10*["g","r","b","c","k"]

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], s=80, color=color, linewidths = 5, zorder = 10)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], s=90, c='k', marker='x')

y = ([[4,4],[8,4],[8,10]])

for unknown in y:
    classification=clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', s=80, c=colors[classification])

plt.show()