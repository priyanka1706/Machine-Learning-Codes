import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')
# We use warnings to avoid using a lower K value than we have groups,
# Math for the square root functionality, at least initially (since I will show a more efficient method),
# and then Counter from collections to get the most popular votes.

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]
# The dataset is just a Python dictionary with the keys being the color of the points (think of these as the class),
# and then the datapoints that are attributed with this class.
# Next, we just specify a simple data point, 5,7 to be data we want to test.

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        #if less than or equal to the total no of groups, we could have an even split
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    # euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
    # euclid_dist = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
    votes = [i[1] for i in sorted(distances)[:k]] # we only care about first k
    vote_result = Counter(votes).most_common(1)[0][0] # 1 means the most common
    # Without doing the [0][0] part, you get [('r', 3)]. Thus, [0][0] gives us the first element in the tuple.
    # The three you see is how many votes 'r' got.
    return vote_result

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)
plt.scatter(new_features[0],new_features[1],s=100)
#plt.show()

result = k_nearest_neighbors(dataset, new_features)
print(result)
plt.scatter(new_features[0], new_features[1], s=100, color = result)
plt.show()

#K nearest neighbors must be individually compared, no exact formula
#hence it gets slower as the number of data points increases