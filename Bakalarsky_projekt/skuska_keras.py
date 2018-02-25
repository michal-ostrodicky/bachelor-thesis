import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
data_csv = pd.read_csv("prices.csv")
dataset = data_csv['Bergen'].values
dataset = dataset.astype('float32')
dataset = dataset[:,None]



# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size


train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


# reshape dataset
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(48, input_dim=look_back, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam' )
model.fit(trainX, trainY, epochs=600, batch_size=256, verbose=2)


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# generate predictions for training
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

