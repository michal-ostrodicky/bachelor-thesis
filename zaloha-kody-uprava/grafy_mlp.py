import numpy as np
from numpy import array
import pandas as pd
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():

    # data_xls = pd.read_excel("elspot-prices_2017_hourly_eur.xls", 'elspot-prices_2017_hourly_eur', index_col=None, )
    # data_xls.to_csv('prices.csv', encoding='utf-8')

    # load the dataset
    data_csv = pd.read_csv("prices.csv")
    market = 'Bergen'

    data_csv[market].fillna((data_csv[market].mean()), inplace=True)
    dataset = data_csv[market].values

    end_date = data_csv.iat[data_csv.shape[0] - 1, 0]

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
    index = (test_end - test_start) - ((test_end - test_start) % 25)


    series = series.ravel()
    train = np.zeros(train_end)
    test = np.zeros(n - test_start)
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

    train_data = train_data.reshape((len(samples), length))
    # print(train_data.shape)


    samples = np.empty((0, length), dtype=float)

    for i in range(0, index, length):
        sample = test[i:i + length]
        samples = np.append(samples, sample.reshape(1, length), axis=0)
    # print(len(samples))

    test_data = np.array(samples)
    # print(train_data.shape)
    # print(train_data)

    test_data = test_data.reshape((len(samples), length))

    train_X = train_data[:, :-1]
    train_y = train_data[:, 1:]
    test_X = test_data[:, :-1]
    test_y = test_data[:, 1:]


    buduce = test_X[test_X.shape[0]-1]
    buduce = buduce.reshape((1,buduce.shape[0]))

    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)


    model = Sequential()
    model.add(Dense(50, input_dim=24, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(24))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(train_X, train_y, epochs=40, batch_size=48 , validation_split=0.1, verbose=0)


    preds = model.predict(train_X)
    print(preds.shape)
    print("TRENOVACIE")
    # print("MAPE trenovacie ", mean_absolute_percentage_error(train_y, preds))
    # nsamples, nx = train_y.shape
    # train_y = train_y.reshape((nsamples, nx))
    train_y = scaler.inverse_transform(train_y)
    # nsamples, nx = preds.shape
    # preds = preds.reshape((nsamples, nx))
    preds = scaler.inverse_transform(preds)
    print("MAPE trenovacich ", mean_absolute_percentage_error(train_y, preds))


    preds = model.predict(test_X)
    # print(preds)
    # preds.to_csv('preds.csv',sep=" ")
    print("TESTOVACIE")

    # print("MAPE testovacie nenormalizovane", mean_absolute_percentage_error(train_y, preds))
    # nsamples, nx = test_y.shape
    # test_y = test_y.reshape((nsamples, nx))
    test_y = scaler.inverse_transform(test_y)

    # nsamples, nx = preds.shape
    # preds = preds.reshape((nsamples, nx ))
    preds = scaler.inverse_transform(preds)

    preds = preds.ravel()
    test_y = test_y.ravel()

    pyplot.figure(figsize=(18, 9))
    pyplot.plot(test_y, 'b')
    pyplot.plot(preds, 'r')

    pyplot.legend(['Skutočné', 'Predikované'])
    pyplot.ylabel('Cena')
    pyplot.title('Cena elektriny - testovacia sada');
    pyplot.show()

    preds = model.predict(buduce)
    preds = scaler.inverse_transform(preds)

    preds = preds.ravel()


    datumy = [None] * 24
    td = np.timedelta64(1, 'h')
    last_date = pd.to_datetime(end_date)

    datumy[0] = last_date + 24*td

    for i in range(23):
        datumy[i + 1] = datumy[i] + td


    output = np.round(preds, 2)
    result_network = [datumy, output]

    # print("Predikcia: ", result_network)
    for i in range(24):
        print(result_network[0][i], " ", result_network[1][i])

    pyplot.figure(figsize=(18, 9))
    pyplot.plot(result_network[0],result_network[1], 'y')
    pyplot.legend(['Budúca cena'])
    pyplot.ylabel('Cena')
    pyplot.title('Cena elektriny');
    pyplot.show()


if __name__ == '__main__':
    main()



