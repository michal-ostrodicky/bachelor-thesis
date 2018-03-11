import numpy
import pandas as pd
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import time
from sklearn.neural_network import MLPClassifier

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    print("X data", dataX)
    print("Y data", dataY)

    return numpy.array(dataX), numpy.array(dataY)

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
    print(train_X)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)


    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    mlp.fit(train_X, train_y)

   #  model = Sequential()
   #  model.add(Dense(128, input_shape=(24,1)))
   #  model.add(Activation('relu'))
   #  model.add(Dropout(0.15))
   #  model.add(Dense(128, input_shape=(24,1)))
   #  model.add(Activation('relu'))
   #  model.add(Dropout(0.15))
   #  model.add(Dense(24,input_shape=(24,1)))
   #  model.add(Activation('relu'))
   #  model.compile(loss='mse', optimizer='adam')
   #  start = time.time()
   # #  model.fit(train_X, train_y, epochs=10, batch_size=512, validation_split=0.1, verbose=2)
   #  print("> Compilation Time : ", time.time() - start)
    # preds = model.predict(test_X)
    #
    # preds = scaler.inverse_transform(preds)
    # test_y = test_y.reshape(-1, 1)
    # actuals = scaler.inverse_transform(test_y)
    #
    # print("RMSE je ", math.sqrt(mean_squared_error(actuals,preds)))




if __name__ == '__main__':
    main()



