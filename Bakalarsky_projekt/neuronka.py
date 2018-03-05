import numpy
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


market = 'Bergen'

# load the dataset
data_csv = pd.read_csv("prices.csv")
dataset = data_csv[market].values
dataset = dataset.astype('float32')
dataset = dataset[:,None]

kf = KFold(n_splits=8)

trainRMSE = list()
testRMSE = list()


for train_indices, test_indices in kf.split(dataset):
    # reshape dataset
    look_back = 2
    # print('Train: %s | test: %s' % (train_indices, test_indices))

    train = dataset[train_indices,:]
    test = dataset[test_indices, :]

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler.fit(train)
    # train = scaler.transform(train)
    # test = scaler.transform(test)

    # print(train_indices,test_indices.shape)
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(48, input_dim=look_back, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=600, batch_size=256, verbose=2)

    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    trainRMSE.append(math.sqrt(trainScore))
    testRMSE.append(math.sqrt(testScore))


# Estimate model performance
print("Trenovacie RMSE ", trainRMSE)
print("Testovacie RMSE ", testRMSE)
print("Priemerna hodnota RMSE na testovacich: ", sum(trainRMSE)/ float(len(trainRMSE)))
print("Priemerna hodnota RMSE na testovacich: ", sum(testRMSE) / float(len(testRMSE)))


# generate predictions for training
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

