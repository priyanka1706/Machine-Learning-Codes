import pandas as pd
import quandl, math, datetime, pickle #With pickle, you can save any Python object, like our classifier.
import numpy as np  #numpy module to convert data to numpy arrays, which is what Scikit-learn wants
from sklearn import preprocessing, svm 
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

# preprocessing is the module used to do some cleaning/scaling of data prior to machine learning

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100  #Spread or volatility
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100  #percentage change
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
# Replace missing data with -99,999; With many machine learning classifiers - recognized and treated as an outlier feature.
forecast_out = int(math.ceil(0.01*len(df)))
# We're saying we want to forecast out 1% of the entire length of the dataset.
# a row will predict data into like if 1% was say 5 days, then one row will show data for the label 5 days into the future

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
# Our X_lately variable contains the most recent features, which we're going to predict against.
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# splitting the data into train and test

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('LinearRegression: ',confidence)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

#pickle_in = open('linearregression.pickle', 'rb')
#clf = pickle.load(pickle_in)

forecast_set = clf.predict(X_lately)
# The forecast_set is an array of forecasts, showing that not only could you just seek out a single prediction,
# but you can seek out many at once.

#print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan
 #deleted last days so that we can forecast it

#We need to first grab the last day in the dataframe, and begin assigning each new forecast to a new day.
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
# Now we have the next day we wish to use, and one_day is 86,400 seconds.
# Now we add the forecast to the existing dataframe

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)] + [i]
# So here all we're doing is iterating through the forecast set, taking each forecast and day,
# and then setting those values in the dataframe (making the future "features" NaNs). The last line's code
# simply takes all of the first columns, setting them to NaNs, and then the final column is whatever i is (the forecast in this case).

# what [next_date] does is that it makes the date the index. So f the index doesnt exist as date, itll make it
# the np.nan says columns-1 columns are basically NAN
# so like close, PCT change all nan because were predicting so we dont know it
# and just the last column is i, that is the forecast

#try df.tail() if you want help

#plotting it out
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# EXPLANATION FOR FORECAST OUT
# forecast out basically is how may days into the future are we predicting data
# for eg if forecast_out=10, then i will use data from 10 days ago to predict today, from 9 days ago for tomorrow
# When we do this, some 10 days at the end are left un predicted
# so we are using those 10 days, for our prediction algo, apart from training and testing
# END