#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:19:42 2018

@author: 3i521392
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import plotlosses as ptloss

lag = 10 #time lag
fr = 16 #Frame rate
mint = 8 #time in minutes
sec = 60 #seconds
decimate = 8 #16HZ
n_train_minutes = fr*mint*sec # training data size 


def take_mocap(file):
    # Importing the training set
    dataset_train = pd.read_csv(file)
    # Take the mocap: x y z
    dataset = dataset_train.loc[:, ['RWRI.X','RWRI.Y','RWRI.Z']]    
    mocap = dataset.values.astype('float64') 
    return mocap

def take_eeg(file):
    dataset_train = pd.read_csv(file)
    dataset = dataset_train.loc[:, ['F3','FC5','FC6','F4']]    
    eeg = dataset.values.astype('float64') 
    return eeg

dataset_train = pd.read_csv('train.csv')

def decimate_data(decim, data):
    return data[::decim]

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

eegmocap_data = pd.read_csv('train.csv')
mocap_data  = eegmocap_data.loc[:, ['RWRI.X']] 
eeg_data    = eegmocap_data.loc[:, ['F3','FC5','FC6','F4']] 
#Dacimate to 16hz
mocap_decim16hz = decimate_data(decimate, mocap_data)
eeg_decim16hz   = decimate_data(decimate, eeg_data)

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
mocap_scaled = sc.fit_transform(mocap_decim16hz)
eeg_scaled   = sc.fit_transform(eeg_decim16hz)

# Creating a data structure with lag = 10, 160 timesteps and 1 output
reframed_eeg = series_to_supervised(eeg_scaled, fr*lag, 0)
reframed_mcp = series_to_supervised(mocap_scaled, fr*lag, 1)

reframed_eeg = np.array(reframed_eeg)
mocap_X = np.array(reframed_mcp)

#take input (X) and output (Y) data
X, Y = reframed_eeg, np.reshape(mocap_X[:, -1], (len(mocap_X[:, -1]),1))

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
checkpointer = ModelCheckpoint(filepath='lstm_eeg_mocap.bestmodel.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop    = EarlyStopping(monitor='loss', min_delta=1e-4, patience=30, verbose=0, mode='auto')
visloss = ptloss.PlotLosses()
#callback_list = [checkpointer, earlystop, visloss]
callback_list = [checkpointer, earlystop]
#lstm_mocapx.load_weights(filepath = 'lstm_mocap.bestmodel.hdf5')
history =lstm_mocapx.fit(X_train, Y_train, epochs = 100, batch_size = 64, validation_data=(X_test, Y_test),verbose=1,callbacks=callback_list)


pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#regressor.save_weights(filepath = 'rnn_reach_Hidden4-100_inp3s_ep100_bn_bs256_loss0.0266.h5')

        

#Making the predictions and visualising the results

yhat = lstm_mocapx.predict(X_test)
yhat = yhat.astype('float64')
mocap_temp = mocap_scaled[n_train_minutes+160:,1:3]
mocapk = np.concatenate((yhat, mocap_temp), axis=1)

yhat_inv = sc.inverse_transform(mocapk)
yori     = sc.inverse_transform(mocap_scaled[n_train_minutes+160:,:])

# Visualising the results
pyplot.figure()
plt.plot(yori[:,0], color = 'red', label = 'Real X Trajectoty')
plt.plot(yhat_inv[:,0], color = 'blue', label = 'Predicted X Trajectory')
plt.title('X Trajectory Prediction')
plt.xlabel('Time')
plt.ylabel('X value')
plt.legend()
plt.show()


