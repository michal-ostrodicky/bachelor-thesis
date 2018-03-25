import numpy as np
from numpy import array
import pandas as pd
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import time
from matplotlib import pyplot

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def prediction_neural_network_flask(X):
    train_size = int(len(X) * 0.67)
    train, test = X[0:train_size, :], X[train_size:len(X), :]
    print(train)

    # reshape dataset
    look_back = 2
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(48, input_dim=look_back, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=600, batch_size=256, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


    return model,math.sqrt(testScore)

def main():
    # fix random seed for reproducibility
    # numpy.random.seed(7)

    data_xls = pd.read_excel("elspot-prices_2017_hourly_eur.xls", 'elspot-prices_2017_hourly_eur', index_col=None, )
    data_xls.to_csv('prices.csv', encoding='utf-8')

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
        samples = np.append(samples,sample.reshape(1,25),axis = 0)

    # print(len(samples))

    train_data = np.array(samples)
    # print(train_data.shape)
    #print(train_data)

    train_data = train_data.reshape((len(samples), length))
    # print(train_data.shape)


    samples = np.empty((0, length), dtype=float)

    for i in range(0, index, length):
        sample = test[i:i + length]
        samples = np.append(samples, sample.reshape(1, 25), axis=0)
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
    print(buduce.shape)
    print(buduce)
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)


    model = Sequential()
    model.add(Dense(128, input_dim=24, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(24))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(train_X, train_y, epochs=100, batch_size=48 , validation_split=0.1, verbose=2)
    start = time.time()

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

    print("> Compilation Time : ", time.time() - start)
    preds = model.predict(test_X)
    print(preds)
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

    pyplot.plot(test_y)
    pyplot.plot(preds)
    pyplot.show()
    print("MAPE testovacie ", mean_absolute_percentage_error(test_y, preds))

    # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds)))

    preds = model.predict(buduce)
    preds = scaler.inverse_transform(preds)

    preds = preds.ravel()
    print("NEXT VALUES: ", preds)

   # pyplot.plot(dataset)
   #  pyplot.plot(preds)
   #
   #  pyplot.show()




if __name__ == '__main__':
    main()



