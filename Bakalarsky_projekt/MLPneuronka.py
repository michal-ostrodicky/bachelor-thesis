import numpy as np
from numpy import array
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
import sys


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Metoda na vypocet presnosti predikcie.
    :param y_true: skutocne hodnoty ceny elektriny
    :param y_pred: predpovedane ceny
    :return:
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        market = sys.argv[2]

    # data_xls = pd.read_excel("elspot-prices_2017_hourly_eur.xls", 'elspot-prices_2017_hourly_eur', index_col=None, )
    # data_xls.to_csv('prices.csv', encoding='utf-8')

    # load the dataset
    data_csv = pd.read_csv(filename)

    ''''
    Chybajuce hodnoty v datasete nahradzame priemerom. 
    '''
    data_csv[market].fillna((data_csv[market].mean()), inplace=True)
    dataset = data_csv[market].values

    end_date = data_csv.iat[data_csv.shape[0] - 1, 0]

    n = dataset.shape[0]
    print("Počet dát: ",n)
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
    train = series[0:train_end]
    test = series[test_start:test_end]

    length = 25

    samples = np.empty((0,length),dtype=float)

    for i in range(0, nrow, length):
        sample = train[i:i + length]
        samples = np.append(samples,sample.reshape(1,length),axis = 0)

    train_data = np.array(samples)
    train_data = train_data.reshape((len(samples), length))

    samples = np.empty((0, length), dtype=float)

    for i in range(0, index, length):
        sample = test[i:i + length]
        samples = np.append(samples, sample.reshape(1, length), axis=0)


    test_data = np.array(samples)
    test_data = test_data.reshape((len(samples), length))

    train_X = train_data[:, :-1]
    train_y = train_data[:, 1:]
    test_X = test_data[:, :-1]
    test_y = test_data[:, 1:]
    buduce = test_X[test_X.shape[0]-1]
    buduce = buduce.reshape((1,buduce.shape[0]))

    model = Sequential()
    model.add(Dense(50, input_dim=24, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(24))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(train_X, train_y, epochs=40, batch_size=48 , validation_split=0.1, verbose=0)

    preds = model.predict(test_X)
    test_y = scaler.inverse_transform(test_y)
    preds = scaler.inverse_transform(preds)

    preds = preds.ravel()
    test_y = test_y.ravel()

    print("MAPE na testovacích dátach:", mean_absolute_percentage_error(test_y, preds))

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

    print("Predikcia cien elektriny na nasledujúcich 24 hodín:")
    for i in range(24):
        print(result_network[0][i], " ", result_network[1][i])


if __name__ == '__main__':
    main()



