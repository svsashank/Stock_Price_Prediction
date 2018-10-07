# Recurrent Neural Network

# Part 1 - Data Pre-Processing

# Importing Lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Training Set
dataset_training = pd.read_csv('Google_Stock_Price_Train.csv')
open_values_training = dataset_training.iloc[:, 1:2].values
trading_volume_training = dataset_training.iloc[:, -1].values
Open_volume_training_set = pd.concat((dataset_training['Open'], dataset_training['Volume']), axis = 1)
training_set = Open_volume_training_set[].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a specific data stricture - timestamps and output
X_train = []
y_train = []
for i in range(80, 1258):
    X_train.append(training_set_scaled[i-80:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding first LSTM layer and some dropout regularization
regressor.add(LSTM(units = 75, return_sequences = True, input_shape =  (X_train.shape[1], 1)))
regressor.add(Dropout(0.25))

# Adding second LSTM layer and some dropout regularization
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.25))

# Adding third LSTM layer and some dropout regularization
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.25))

# Adding fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.25))

# Adding fifth LSTM layer and some dropout regularization
regressor.add(LSTM(units = 75, return_sequences = False))
regressor.add(Dropout(0.25))

# Adding output layer
regressor.add(Dense(1))

#Compiling the RNN
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

# Fit the training set
regressor.fit(x = X_train, y = y_train, epochs = 50, batch_size = 32)


# Part - 3 Predict the Result

# Get the test dataset
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Predict the stock price
dataset_total = pd.concat((dataset_training['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()