import numpy as np
import pandas as pd
from numpy import array
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import time
from matplotlib import pyplot
from keras.layers import TimeDistributed



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():

    # load the dataset
    data_csv = pd.read_csv("prices.csv")

    data_csv['Oslo'].fillna((data_csv['Oslo'].mean()), inplace=True)
    dataset = data_csv['Oslo'].values
    n = dataset.shape[0]
    print(n)
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    series = array(scaled)


    nrow = round(0.8 * series.shape[0])- (round(0.8 * series.shape[0]) % 24)

    train_end = round(0.8 * series.shape[0])- (round(0.8 * series.shape[0]) % 24)
    test_start = train_end + 1
    test_end = n
    index = (test_end - test_start) - ((test_end - test_start) % 24)

    # print("Indexx", index)
    # print(nrow)
    # print(train_end)


    series = series.ravel()
    train = np.zeros(train_end)
    test = np.zeros(n - test_start)
    # print(test.shape)
    # print(train.shape)
    train = series[0:train_end]
    test = series[test_start:test_end]

    # print(len(train))
    # print(len(test))

    length = 24

    samples = np.empty((0,length),dtype=float)

    for i in range(0, nrow, length):
        sample = train[i:i + length]
        samples = np.append(samples,sample.reshape(1,24),axis = 0)
    # print(len(samples))

    train_data = np.array(samples)
    # print(train_data.shape)
    #print(train_data)

    train_data = train_data.reshape((len(samples), length, 1))
    # print(train_data.shape)


    samples = np.empty((0, length), dtype=float)

    for i in range(0, index, length):
        sample = test[i:i + length]
        samples = np.append(samples, sample.reshape(1, 24), axis=0)
    # print(len(samples))

    test_data = np.array(samples)
    # print(train_data.shape)
    # print(train_data)

    test_data = test_data.reshape((len(samples), length, 1))
    # print(train_data.shape)


    # series.to_csv('povodne.csv', sep=" ")
    #
    # window_size = 24
    # j = 0
    #
    # series_s = series.copy()
    # while j < n:
    #     # print(series_s[i])
    #     print(series_s.shift(24))
    #     # series = pd.concat([series, series_s.shift(24)], axis=1)
    #     j = j + window_size
    #
    # series.dropna(axis=0, inplace=True)
    #
    # # print(dataset)
    # series.to_csv('series.csv',sep=" ")
    # print(series.shape)
    # print(series_s)
    # print(series_s.shape)
    # series_s.to_csv('series_s.csv', sep=" ")

    # entireData = data((n, 52, 1))
    train_X = train_data[:, :-1, :]
    train_y = train_data[:, 1:, :]
    test_X = test_data[:, :-1, :]
    test_y = test_data[:, 1:, :]


    # print("X kove", X)
    # print("Y kove", y)

    print(train_X.shape)
    print(train_y.shape)

    #
    # nrow = round(0.8 * series.shape[0])
    # train = series.iloc[:nrow, :]
    # test = series.iloc[nrow:, :]

    # train = shuffle(train)

    # train_X = train.iloc[:, :-1]
    # train_y = train.iloc[:, -1]
    # test_X = test.iloc[:, :-1]
    # test_y = test.iloc[:, -1]
    #
    #
    # train_X = train_X.values
    # train_y = train_y.values
    # test_X = test_X.values
    # test_y = test_y.values
    #
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    #
    # train_X = train_X.reshape(train_X.shape[0], 24, 1)
    # test_X = test_X.reshape(test_X.shape[0], 24, 1)
    # train_y = train_y.reshape(train_y.shape[0],24, 1)
    # test_y = test_y.reshape(test_y.shape[0], 24, 1)
    #
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    #
    model = Sequential()
    model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    start = time.time()
    model.fit(train_X, train_y, batch_size=256, epochs=500, validation_split=0.1)
    print("> Compilation Time : ", time.time() - start)
#
#     # preds_moving = moving_test_window_preds(24,test_X, model,scaler)
#     # test_y = test_y.reshape(-1, 1)
#     # actuals = scaler.inverse_transform(test_y)
#     # print(preds_moving.shape)
#     # print(actuals.shape)
#     # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds_moving)))
#
#
#     preds = model.predict(test_X)
#     print("TRENOVACIE")
#     # print("Tvar predikcie ", preds.shape)
#     # print("Tvar testovacich ", test_y.shape)
#
#     nsamples, nx, ny = preds.shape
#     normalizovane_preds = preds.reshape((nsamples, nx * ny))
#
#     nsamples, nx, ny = test_y.shape
#     normalizovane_test_y = test_y.reshape((nsamples, nx * ny))
#
#     # print("Predikcie: ", preds)
#     # print("Testovacie: ", test_y)
#     # print(preds.shape)
#     # print(test_y.shape)
#     RMSE = math.sqrt(mean_squared_error(normalizovane_test_y, normalizovane_preds))
#     print("Scaled test RMSE je ", RMSE)
#
#
#     nsamples, nx, ny = preds.shape
#     preds = preds.reshape((nsamples, nx * ny))
#     preds = scaler.inverse_transform(preds)
#
#     nsamples, nx, ny = test_y.shape
#     test_y = test_y.reshape((nsamples, nx * ny))
#     test_y = scaler.inverse_transform(test_y)
#
#     print("Originalny scale test RMSE je ", math.sqrt(mean_squared_error(test_y,preds)))
#
#
# #### trenovacie
    preds = model.predict(train_X)
    print("TRENOVACIE")
    # print("MAPE trenovacie ", mean_absolute_percentage_error(train_y, preds))
    nsamples, nx, ny = train_y.shape
    train_y = train_y.reshape((nsamples, nx * ny))
    train_y = scaler.inverse_transform(train_y)

    nsamples, nx, ny = preds.shape
    preds = preds.reshape((nsamples, nx * ny))
    preds = scaler.inverse_transform(preds)

    print("Originalny scale trenovacie RMSE je ", math.sqrt(mean_squared_error(train_y, preds)))
    print("MAPE trenovacich ", mean_absolute_percentage_error(train_y, preds))


    preds = model.predict(test_X)
    print("TESTOVACIE")
    # print("MAPE testovacie nenormalizovane", mean_absolute_percentage_error(train_y, preds))
    nsamples, nx, ny = test_y.shape
    test_y = test_y.reshape((nsamples, nx * ny))
    test_y = scaler.inverse_transform(test_y)

    nsamples, nx, ny = preds.shape
    preds = preds.reshape((nsamples, nx * ny))
    preds = scaler.inverse_transform(preds)


    print("Originalny scale testovacie RMSE je ", math.sqrt(mean_squared_error(test_y, preds)))

    print("MAPE testovacie ", mean_absolute_percentage_error(test_y,preds))
#
#     nsamples, nx, ny = preds.shape
#     normalizovane_preds = preds.reshape((nsamples, nx * ny))
#
#     nsamples, nx, ny = train_y.shape
#     normalizovane_train_y = train_y.reshape((nsamples, nx * ny))
#
#     # print("Predikcie: ", preds)
#     # print("Testovacie: ", test_y)
#     # print(preds.shape)
#     # print(test_y.shape)
#     RMSE = math.sqrt(mean_squared_error(normalizovane_train_y, normalizovane_preds))
#     print("Scaled trenovacie RMSE je ", RMSE)
#
#     nsamples, nx, ny = preds.shape
#     preds = preds.reshape((nsamples, nx * ny))
#     preds = scaler.inverse_transform(preds)
#
#     nsamples, nx, ny = train_y.shape
#     train_y = train_y.reshape((nsamples, nx * ny))
#     train_y = scaler.inverse_transform(train_y)
#
#     print("Originalny scale trenovacie RMSE je ", math.sqrt(mean_squared_error(train_y, preds)))
#
#
#    # test_y = test_y.reshape(-1, 1)
#     # actuals = scaler.inverse_transform(test_y)
#
#     # volaco = np.round(actuals[:,0], 2)
#     # preds = np.round(preds[:,0], 2)
#     # print(actuals.shape)
#     # print(preds.shape)
#     # print("Predikcia", volaco)
#     # print("Aktualne", actuals)
#
#     # pyplot.plot(preds)
#     # pyplot.plot(test_y)
#     # pyplot.show()


if __name__ == '__main__':
    main()



