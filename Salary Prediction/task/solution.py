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

def calc_lnr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    lnr = LinearRegression()
    lnr.fit(X_train, y_train)
    pred = lnr.predict(X_test)
    mp = mape(y_test, pred)
    return mp

# read data
data = pd.read_csv('../Data/data.csv')

X = pd.DataFrame(data['rating'])
y = data['salary']
m_min = sys.maxsize
for k in range(2, 4):
    m = calc_lnr(X ** k, y)
    if m < m_min:
        m_min = m

print(round(m_min, 5))
