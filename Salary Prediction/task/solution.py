import os
import sys

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

def calc_lnr(X, y, y_val):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    lnr = LinearRegression()
    lnr.fit(X_train, y_train)
    pred = lnr.predict(X_test)
    if y_val != 0:
        y_val = y_train.median()
    pred = np.where(pred < 0, y_val, pred)
    mp = mape(y_test, pred)
    # print(round(mp, 5))
    return mp

# read data
data = pd.read_csv('../Data/data.csv')

X = pd.DataFrame(data).drop(columns='salary')
y = data['salary']
# corr = X.corr()
var_list = ['rating', 'age', 'experience']
drop_list = ['age', 'experience']
m_0 = calc_lnr(X.drop(columns=drop_list), y, 0)
m_1 = calc_lnr(X.drop(columns=drop_list), y, 1)

print(round(min(m_0, m_1), 5))
