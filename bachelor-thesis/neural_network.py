import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
import time
from matplotlib import pyplot

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def prediction_neural_network_flask(X,market):
    dataset = X[market].values

    n = X.shape[0]
    print(n)
    dataset = dataset.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    series = array(scaled)

    nrow = round(0.8 * series.shape[0]) - (round(0.8 * series.shape[0]) % 25)

    train_end = round(0.8 * series.shape[0]) - (round(0.8 * series.shape[0]) % 25)
    test_start = train_end + 1
    test_end = n
    index = (test_end - test_start) - ((test_end - test_start) % 25)

    series = series.ravel()
    train = np.zeros(train_end)
    test = np.zeros(n - test_start)
    train = series[0:train_end]
    test = series[test_start:test_end]

    length = 25

    samples = np.empty((0, length), dtype=float)

    for i in range(0, nrow, length):
        sample = train[i:i + length]
        samples = np.append(samples, sample.reshape(1, 25), axis=0)

    # print(len(samples))

    train_data = np.array(samples)
    # print(train_data.shape)
    # print(train_data)

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

    buduce = test_X[test_X.shape[0] - 1]
    buduce = buduce.reshape((1, buduce.shape[0]))

    model = Sequential()
    model.add(Dense(128, input_dim=24, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(24))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(train_X, train_y, epochs=100, batch_size=48, validation_split=0.1, verbose=2)
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

    MAPE = mean_absolute_percentage_error(test_y, preds)
    print("MAPE testovacie ", MAPE)

    # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds)))

    preds = model.predict(buduce)
    preds = scaler.inverse_transform(preds)

    preds = preds.ravel()
    print("NEXT VALUES: ", preds)
    return model,MAPE, preds

def main():
   pass




if __name__ == '__main__':
    main()



