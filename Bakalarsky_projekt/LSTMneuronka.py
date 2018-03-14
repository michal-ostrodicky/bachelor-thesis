import numpy as np
import pandas as pd
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


def main():

    # load the dataset
    data_csv = pd.read_csv("prices.csv")
    dataset = data_csv['Bergen']
    # print(dataset.shape)

    # dataset = dataset.astype('float32')
    # dataset = dataset[:,None]

    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(dataset)
    series = pd.DataFrame(scaled)

    window_size = 24

    series_s = series.copy()
    for i in range(window_size):
        series = pd.concat([series, series_s.shift((i + 1))], axis=1)

    series.dropna(axis=0, inplace=True)
    # print(dataset)
    series.to_csv('series.csv',sep=" ")
    # print(series.shape)
    # print(series_s)
    # print(series_s.shape)
    series_s.to_csv('series_s.csv', sep=" ")

    nrow = round(0.8 * series.shape[0])
    train = series.iloc[:nrow, :]
    test = series.iloc[nrow:, :]

    train = shuffle(train)

    train_X = train.iloc[:, :-1]
    train_y = train.iloc[:, :-1]
    test_X = test.iloc[:, :-1]
    test_y = test.iloc[:, :-1]

    train_X = train_X.values
    train_y = train_y.values
    test_X = test_X.values
    test_y = test_y.values

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    train_y = train_y.reshape(train_y.shape[0], train_y.shape[1], 1)
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1], 1)

    model = Sequential()
    model.add(LSTM(24, input_shape=(24, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    start = time.time()
    model.fit(train_X, train_y, batch_size=256, epochs=500, validation_split=0.1)
    print("> Compilation Time : ", time.time() - start)

    # preds_moving = moving_test_window_preds(24,test_X, model,scaler)
    # test_y = test_y.reshape(-1, 1)
    # actuals = scaler.inverse_transform(test_y)
    # print(preds_moving.shape)
    # print(actuals.shape)
    # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds_moving)))


    preds = model.predict(test_X)

    print("Tvar predikcie ", preds.shape)
    print("Tvar testovacich ", test_y.shape)

    nsamples, nx, ny = preds.shape
    normalizovane_preds = preds.reshape((nsamples, nx * ny))

    nsamples, nx, ny = test_y.shape
    normalizovane_test_y = test_y.reshape((nsamples, nx * ny))

    # print("Predikcie: ", preds)
    # print("Testovacie: ", test_y)
    # print(preds.shape)
    # print(test_y.shape)
    RMSE = math.sqrt(mean_squared_error(normalizovane_test_y, normalizovane_preds))
    print("Scaled RMSE je ", RMSE)


    nsamples, nx, ny = preds.shape
    preds = preds.reshape((nsamples, nx * ny))
    preds = scaler.inverse_transform(preds)

    nsamples, nx, ny = test_y.shape
    test_y = test_y.reshape((nsamples, nx * ny))
    test_y = scaler.inverse_transform(test_y)

    print("Originalny scale RMSE je ", math.sqrt(mean_squared_error(test_y,preds)))

   # test_y = test_y.reshape(-1, 1)
    # actuals = scaler.inverse_transform(test_y)

    # volaco = np.round(actuals[:,0], 2)
    # preds = np.round(preds[:,0], 2)
    # print(actuals.shape)
    # print(preds.shape)
    # print("Predikcia", volaco)
    # print("Aktualne", actuals)

    # pyplot.plot(preds)
    # pyplot.plot(test_y)
    # pyplot.show()


if __name__ == '__main__':
    main()



