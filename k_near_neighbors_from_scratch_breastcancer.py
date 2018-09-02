import numpy as np
import warnings
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import  style

style.use('fivethirtyeight')

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
# Converts the data to a list of lists. Note that we're explicitly converting the entire dataframe to float.
# For some reason, at least for me, some of the data points were numbers still, but of the string datatype, so that was no good.

random.shuffle(full_data)
test_size=0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] #start to full-0.2 ie to 0.8
test_data = full_data[-int(test_size*len(full_data)):] #full - 0.2 to end ie 0.8 to end
# the dictionaries have two keys: 2 and 4. The 2 is for the benign tumors (the same value the actual dataset used),
# and the 4 is for malignant tumors, same as the data.

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, 5)
        if group==vote:
            correct+=1
        total+=1

print('Accuracy: ', correct/total)