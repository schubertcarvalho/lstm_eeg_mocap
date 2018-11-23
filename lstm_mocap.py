# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:15:48 2018

@author: Schubert Ribeiro de Carvalho

This file makes part of the LSTM reaching model: sequence learning 
lstm_mocap learns to predict the next movement from a previous motion sequence
"""


# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
from matplotlib import pyplot
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import plotlosses as ptloss

lag = 10 #time lag
fr = 16 #Frame rate
mint = 8 #time in minutes
sec = 60 #seconds
decidamte = 8 #16HZ
n_train_minutes = fr*mint*sec # training data size 


def take_mocap(file):
    # Importing the training set
    dataset_train = pd.read_csv(file)
    # Take the mocap: x y z
    dataset = dataset_train.loc[:, ['RWRI.X','RWRI.Y','RWRI.Z']]    
    mocap = dataset.values.astype('float64') 
    return mocap

def decimate_mocap(decim, mocap):
    return mocap[::decim]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

mocap_data = take_mocap('train.csv')
mocap_decim16hz = decimate_mocap(decidamte, mocap_data) #Mocap decimation 16hz
mocap_X = np.reshape(mocap_decim16hz[:,0], (mocap_decim16hz[:,0].shape[0], 1))

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
mocap_scaled = sc.fit_transform(mocap_X)

# Creating a data structure with lag = 10, 160 timesteps and 1 output
reframed = series_to_supervised(mocap_scaled, fr*lag, 1)
mocap_X = np.array(reframed)

#take input (X) and output (Y) data
X, Y = mocap_X[:, :-1], mocap_X[:, -1]

# Split data into train and test
X_train, X_test = X[:n_train_minutes, :], X[n_train_minutes:, :]
Y_train, Y_test = Y[:n_train_minutes], Y[n_train_minutes:]

# Reshaping for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1], 1))



#Building the lstm_mocap
lstm_mocapx = Sequential()
lstm_mocapx.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
lstm_mocapx.add(BatchNormalization(axis=1, epsilon=1e-3))
lstm_mocapx.add(LSTM(units = 50, return_sequences = True))
lstm_mocapx.add(BatchNormalization(axis=1, epsilon=1e-3))
lstm_mocapx.add(LSTM(units = 50))
lstm_mocapx.add(BatchNormalization(axis=1, epsilon=1e-3))
lstm_mocapx.add(Dense(units = 1))
#regressor.load_weights("lstm_mocap.bestmodel.hdf5")

lstm_mocapx.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Model checkpoit: save only the best model
checkpointer = ModelCheckpoint(filepath='lstm_mocap.bestmodel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop    = EarlyStopping(monitor='loss', min_delta=1e-4, patience=30, verbose=0, mode='auto')
visloss = ptloss.PlotLosses()
callback_list = [checkpointer, earlystop, visloss]

lstm_mocapx.load_weights(filepath = 'lstm_mocap.bestmodel.hdf5')
history =lstm_mocapx.fit(X_train, Y_train, epochs = 100, batch_size = 128, validation_data=(X_test, Y_test),verbose=1,callbacks=callback_list)


pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#regressor.save_weights(filepath = 'rnn_reach_Hidden4-100_inp3s_ep100_bn_bs256_loss0.0266.h5')

        

#Making the predictions and visualising the results
mocap_predict = lstm_mocapx.predict(X_test)
mocap_predict = mocap_predict.astype('float64')

mocap_real = np.reshape(Y_test, (len(Y_test),1)) #mocap_scaled[n_train_minutes+160:,1:3]
#mocap_cont = np.concatenate((yhat, mocap_temp), axis=1)

mocap_predict_inv = sc.inverse_transform(mocap_predict)
mocap_real_inv    = sc.inverse_transform(mocap_real)

# Visualising the results
pyplot.figure()
pyplot.plot(mocap_real_inv[:,0], color = 'red', label = 'Real X Trajectoty')
pyplot.plot(mocap_predict_inv[:,0], color = 'blue', label = 'Predicted X Trajectory')
pyplot.title('X Trajectory Prediction')
pyplot.xlabel('Time')
pyplot.ylabel('X value')
pyplot.legend()
pyplot.show()


## Getting the real stock price of 2017
#dataset_test = pd.read_csv('test.csv')
#real_stock_price = dataset_test.iloc[:, 1:2].values
#￼
## Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#inputs        = dataset_total[len(dataset_total) - len(dataset_test) - ts_i:].values
#inputs        = inputs.reshape(-1,1)
#inputs        = sc.transform(inputs)
#X_test        = []
#for i in range(ts_i, np.shape(inputs)[0]): #(60, 80)
#    X_test.append(inputs[i-ct.TS:i, 0])
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#predicted_stock_price = regressor.predict(X_test)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#
## Visualising the results
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
#plt.title('Google Stock Price Prediction')
#plt.xlabel('Time')
#plt.ylabel('Google Stock Price')
#plt.legend()
#plt.show()
