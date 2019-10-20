import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import theano
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sklearn.metrics as sklm

# quandl.ApiConfig.api_key = #####

tickers = ['002015', '600797', '600650']
columns = ['ticker', 'date', 'adj_pre_close', 'adj_open', 'adj_high', 'adj_low']
dfs = []  # this is a list of data frames

df = quandl.get_table('DY/SPA', ticker='002015,600797,600650', paginate=True)
# remove NAs and zeros
for column in columns:
    df = df[pd.notnull(df[column])]
    df.drop(df[(df[column] == 0)].index, inplace=True)
for ticker in tickers:
    dfs.append(df.loc[df['ticker'] == ticker, columns])

# plot time series for a visual check
x_var = 'date'
y_var = 'adj_pre_close'
dfs_sel = []
for n in range(len(dfs)):
    dfn = dfs[n].set_index(x_var)[y_var].astype(float)
    # dfn.plot(figsize=(13, 8))
    # plt.show()
    # convert series to data frame, append
    dfs_sel.append(dfn.to_frame())


# select dates where there are prices for all n stocks
df_new0 = pd.merge(dfs_sel[0], dfs_sel[1], left_index=True, right_index=True)
df_new = pd.merge(df_new0, dfs_sel[2], left_index=True, right_index=True)
df_new.columns = tickers
# print(df_new)


# Multivariate LSTM Forecast Model
# Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/


def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    dft = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dft.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dft.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


# ensure all data is float
values = df_new.values
values = values.astype('float64')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# remove NAs and zeros
for i in range(reframed.shape[1]):
    reframed = reframed[pd.notnull(reframed.iloc[:, i])]
    reframed.drop(reframed[(reframed.iloc[:, i] == 0)].index, inplace=True)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[4, 5]], axis=1, inplace=True)
# print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 250 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=50, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')  # there is no output val_loss
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = np.sqrt(sklm.mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# WORK IN PROGRESS
