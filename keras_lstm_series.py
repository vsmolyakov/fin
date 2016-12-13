import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pandas_datareader import data
from datetime import datetime
import pytz

import matplotlib.pyplot as plt

np.random.seed(0)

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),0])
        dataY.append(dataset[i+look_back,0])

    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":

    #load data
    start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
    spy = data.DataReader("SPY", "google", start, end)
    dataset = np.array(spy['Close'].values).reshape(-1,1)
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape for look_back
    look_back = 10
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # LSTM
    model = Sequential()
    model.add(LSTM(32, input_dim=1)) #look_back))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=100, batch_size=5, verbose=2)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test) 
   
    # scale back 
    train_pred = scaler.inverse_transform(train_pred)
    y_train = scaler.inverse_transform(y_train)
    test_pred = scaler.inverse_transform(test_pred)
    y_test = scaler.inverse_transform(y_test)
   
    # shift predictions for plotting
    train_pred_plot = np.empty_like(dataset)
    train_pred_plot[:,:] = np.nan
    train_pred_plot[look_back:len(train_pred)+look_back,:] = train_pred

    test_pred_plot = np.empty_like(dataset)
    test_pred_plot[:,:] = np.nan
    test_pred_plot[len(train_pred)+(look_back*2)+1:len(dataset)-1,:] = test_pred

    f = plt.figure()
    plt.plot(scaler.inverse_transform(dataset), color='b', lw=2.0, label='S&P 500')
    plt.plot(train_pred_plot, color='g', lw=2.0, label='LSTM train')
    plt.plot(test_pred_plot, color='r', lw=2.0, label='LSTM test')
    plt.legend(loc=3)
    plt.grid(True)
    f.savefig('./lstm.png')
  




       
