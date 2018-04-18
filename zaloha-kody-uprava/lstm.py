import numpy as np
import pandas as pd
from numpy import array
import math
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.layers import TimeDistributed



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    # load the dataset
    data_csv = pd.read_csv("prices.csv")
    market = 'FI'
    data_csv[market].fillna((data_csv[market].mean()), inplace=True)
    dataset = data_csv[market].values
    n = dataset.shape[0]
    print(n)
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    series = array(scaled)

    nrow = round(0.8 * series.shape[0])- (round(0.8 * series.shape[0]) % 25)

    train_end = round(0.8 * series.shape[0])- (round(0.8 * series.shape[0]) % 25)
    test_start = train_end + 1
    test_end = n
    index = (test_end - test_start) - ((test_end - test_start) % 24)


    series = series.ravel()
    train = np.zeros(train_end)
    test = np.zeros(n - test_start)
    # print(test.shape)
    # print(train.shape)
    train = series[0:train_end]
    test = series[test_start:test_end]

    # print(len(train))
    # print(len(test))

    length = 25

    samples = np.empty((0,length),dtype=float)

    for i in range(0, nrow, length):
        sample = train[i:i + length]
        samples = np.append(samples,sample.reshape(1,length),axis = 0)
    # print(len(samples))

    train_data = np.array(samples)
    # print(train_data.shape)
    #print(train_data)

    train_data = train_data.reshape((len(samples), length, 1))
    # print(train_data.shape)


    samples = np.empty((0, length), dtype=float)

    for i in range(0, index, length):
        sample = test[i:i + length]
        samples = np.append(samples, sample.reshape(1, length), axis=0)
    # print(len(samples))

    test_data = np.array(samples)
    # print(train_data.shape)
    # print(train_data)

    test_data = test_data.reshape((len(samples), length, 1))



    # series.to_csv('povodne.csv', sep=" ")

    train_X = train_data[:, :-1, :]
    train_y = train_data[:, 1:, :]
    test_X = test_data[:, :-1, :]
    test_y = test_data[:, 1:, :]
    #
    buduce = test_X[test_X.shape[0] - 1,:,:]
    buduce = buduce.reshape((1, buduce.shape[0],1))
    # print(buduce.shape)
    # print(buduce)

    # print("Trenovac", train_X)
    # print("Testovac", train_y)

    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(24,1), return_sequences=True))
    model.add(LSTM(40, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_X, train_y, batch_size=48, epochs=100, validation_split=0.1,verbose=0)

    preds = model.predict(train_X)


    print("TRENOVACIE")

    # print("MAPE trenovacie ", mean_absolute_percentage_error(train_y, preds))
    nsamples, nx, ny = train_y.shape
    train_y = train_y.reshape((nsamples, nx * ny))
    train_y = scaler.inverse_transform(train_y)
    nsamples, nx, ny = preds.shape
    preds = preds.reshape((nsamples, nx * ny))
    preds = scaler.inverse_transform(preds)
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


    preds = preds.ravel()
    test_y = test_y.ravel()
    # pyplot.plot(test_y)
    # pyplot.plot(preds)
    # pyplot.show()
    print("MAPE testovacie ", mean_absolute_percentage_error(test_y,preds))
       

    preds = model.predict(buduce)
    nsamples, nx, ny = preds.shape
    preds = preds.reshape((nsamples, nx * ny))
    preds = scaler.inverse_transform(preds)
    preds = preds.ravel()
    print("NEXT VALUES: ", preds)

    # pyplot.plot(preds)
    # pyplot.plot(dataset)
    # pyplot.show()



if __name__ == '__main__':
    main()



