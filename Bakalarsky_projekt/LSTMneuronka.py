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


def moving_test_window_preds(n_future_preds,test_X,model,scaler):
    ''' n_future_preds - Represents the number of future predictions we want to make
                         This coincides with the number of windows that we will move forward
                         on the test data
    '''
    preds_moving = np.array(np.zeros(24))  # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0, :].tolist()]  # Creating the first test window
    moving_test_window = np.array(moving_test_window)  # Making it an numpy array

    for i in range(n_future_preds):
        preds_one_step = model.predict(
            moving_test_window)  # Note that this is already a scaled prediction so no need to rescale this
        preds_moving = np.append(preds_moving,preds_one_step[0, 0])  # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1, 1,
                                                1)  # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:, 1:, :], preds_one_step),
                                            axis=1)  # This is the new moving test window, where the first element from the window has been removed and the prediction  has been appended to the end
        print(preds_one_step)


    preds_moving = preds_moving.reshape(-1, 1)
    preds_moving = scaler.inverse_transform(preds_moving)

    return preds_moving


def main():
    # fix random seed for reproducibility
    # numpy.random.seed(7)


    # load the dataset
    data_csv = pd.read_csv("prices.csv")
    dataset = data_csv['Bergen']
    # print(dataset.shape)

    # dataset = dataset.astype('float32')
    # dataset = dataset[:,None]

    dataset = dataset.reshape(-1, 1)
    # print(dataset.shape)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(dataset)
    series = pd.DataFrame(scaled)

    window_size = 24

    series_s = series.copy()
    for i in range(window_size):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)

    series.dropna(axis=0, inplace=True)
    # print(dataset)

    # print(series.shape)

    nrow = round(0.8 * series.shape[0])
    train = series.iloc[:nrow, :]
    test = series.iloc[nrow:, :]

    train = shuffle(train)

    train_X = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_X = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]

    train_X = train_X.values
    train_y = train_y.values
    test_X = test_X.values
    test_y = test_y.values

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    model = Sequential()
    model.add(LSTM(input_shape=(24, 1), output_dim=24, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    start = time.time()
    model.fit(train_X, train_y, batch_size=512, epochs=5, validation_split=0.1)
    print("> Compilation Time : ", time.time() - start)

    # preds_moving = moving_test_window_preds(24,test_X, model,scaler)
    # test_y = test_y.reshape(-1, 1)
    # actuals = scaler.inverse_transform(test_y)
    # print(preds_moving.shape)
    # print(actuals.shape)
    # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds_moving)))

    preds = model.predict(test_X)
    preds = scaler.inverse_transform(preds)
    test_y = test_y.reshape(-1, 1)
    actuals = scaler.inverse_transform(test_y)

    print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds)))

    preds = model.predict(train_X)

    preds = scaler.inverse_transform(preds)
    train_y = train_y.reshape(-1, 1)
    actuals = scaler.inverse_transform(train_y)

    print("RMSE je ", math.sqrt(mean_squared_error(actuals, preds)))
    # print(preds)
    pyplot.plot(actuals)
    pyplot.plot(preds)
    pyplot.show()
    # kf = KFold(n_splits=2)
    #
    #
    # trainRMSE = list()
    # testRMSE = list()
    #
    #
    # for train_indices, test_indices in kf.split(dataset):
    #     # reshape dataset
    #     look_back = 2
    #     # print('Train: %s | test: %s' % (train_indices, test_indices))
    #
    #     train = dataset[train_indices,:]
    #     test = dataset[test_indices, :]
    #
    #     # scaler = MinMaxScaler(feature_range=(-1, 1))
    #     # scaler.fit(train)
    #     # train = scaler.transform(train)
    #     # test = scaler.transform(test)
    #
    #     # print(train_indices,test_indices.shape)
    #     trainX, trainY = create_dataset(train, look_back)
    #     testX, testY = create_dataset(test, look_back)
    #
    #
    #     # create and fit Multilayer Perceptron model
    #     # model = Sequential()
    #     # model.add(Dense(48, input_dim=look_back, activation='relu'))
    #     # model.add(Dense(16, activation='relu'))
    #     # model.add(Dense(1))
    #     # model.compile(loss='mean_squared_error', optimizer='adam')
    #     # model.fit(trainX, trainY, epochs=600, batch_size=256, verbose=2)
    #     #
    #     #
    #     #
    #     #
    #     # trainScore = model.evaluate(trainX, trainY, verbose=0)
    #     # print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    #     # testScore = model.evaluate(testX, testY, verbose=0)
    #     # print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    #     # trainRMSE.append(math.sqrt(trainScore))
    #     # testRMSE.append(math.sqrt(testScore))
    #
    #
    #
    # # Estimate model performance
    #
    # # print("Trenovacie RMSE ", trainRMSE)
    # # print("Testovacie RMSE ", testRMSE)
    # # print("Priemerna hodnota RMSE na testovacich: ", sum(trainRMSE)/ float(len(trainRMSE)))
    # # print("Priemerna hodnota RMSE na testovacich: ", sum(testRMSE) / float(len(testRMSE)))
    # # print("Testovacia x", testX)
    # # print("Testovacie y ",testY)
    # # # generate predictions for training
    # # # trainPredict = model.predict(trainX)
    # # predikcia= model.predict(testY)
    # # print(predikcia)



if __name__ == '__main__':
    main()



