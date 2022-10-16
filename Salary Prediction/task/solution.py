import os

import numpy as np
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# write your code here
X = data['rating']
y = data['salary']
X = np.array(data['rating']).reshape(-1, 1)
y = np.array(data['salary']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

lnr = LinearRegression()
lnr.fit(X_train, y_train)
pred = lnr.predict(X_test)
mape = mape(y_test, pred)

print(round(lnr.intercept_[0], 5), round(lnr.coef_[0,0], 5), round(mape, 5))
