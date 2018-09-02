import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

#handling non numerical data
#what were doing - we take the list of a column, then the set of it (for unique vals), then just keep assigning the value to an id
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            # we're going to convert the column to a list of its values, then we take the set of
            # that column to get just the unique values.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
            # This is basically new_column = list(map( function_to_map, parameter1, parameter2, ... ))
    return df

df = handle_non_numerical_data(df)
# print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
# Preprocessing aims to put your data in a range from -1 to +1, which can make things better.
y = np.array(df['survived'])

clf=KMeans(n_clusters=2)
clf.fit(X)
# it will randomly assign a cluster name to the data, therefore actually 0 could be 1, or 1 could be 0 in data
# Thus, if you consistently get 30% and 70% accuracy, then your model is 70% accurate.
correct = 0
for i in range (len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct+=1

print(correct/len(X))