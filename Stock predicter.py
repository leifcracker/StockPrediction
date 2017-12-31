
#import pandas as pd
import quandl

import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style


#quandl.ApiConfig.api_key = "###use your quandl API key here, inside the quotes###"
#Insert a stock symbol you prefer below
df = quandl.get('WIKI/AMD')


df = df[['Adj. Open', 'Adj. High','Adj. Low','Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna('-99999', inplace=True)

#changing the factor in front of len(df) will change how far the prediction goes out
#shorter predictions are more accurate
#adjust so it predicts only about 10 or so days out, will be a little different depending on how much data exists for each share price
forecast_out = int(math.ceil(0.001*len(df)))
#print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)



x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

#training on sklearn's LinearRegression model
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)

#making a plot
style.use('ggplot')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




