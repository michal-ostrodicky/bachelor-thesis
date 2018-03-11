import numpy
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

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
    dataset = data_csv['Bergen'].values
    dataset = dataset.astype('float32')
    dataset = dataset[:,None]


    kf = KFold(n_splits=2)


    trainRMSE = list()
    testRMSE = list()


    for train_indices, test_indices in kf.split(dataset):
        # reshape dataset
        look_back = 3
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
        # model = Sequential()
        # model.add(Dense(48, input_dim=look_back, activation='relu'))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(trainX, trainY, epochs=600, batch_size=256, verbose=2)
        #
        #
        #
        #
        # trainScore = model.evaluate(trainX, trainY, verbose=0)
        # print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        # testScore = model.evaluate(testX, testY, verbose=0)
        # print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
        # trainRMSE.append(math.sqrt(trainScore))
        # testRMSE.append(math.sqrt(testScore))



    # Estimate model performance

    # print("Trenovacie RMSE ", trainRMSE)
    # print("Testovacie RMSE ", testRMSE)
    # print("Priemerna hodnota RMSE na testovacich: ", sum(trainRMSE)/ float(len(trainRMSE)))
    # print("Priemerna hodnota RMSE na testovacich: ", sum(testRMSE) / float(len(testRMSE)))
    # print("Testovacia x", testX)
    # print("Testovacie y ",testY)
    # # generate predictions for training
    # # trainPredict = model.predict(trainX)
    # predikcia= model.predict(testY)
    # print(predikcia)



if __name__ == '__main__':
    main()



# sns.distplot(data[market]);

# Plots
# data.index = data_csv['Hours'].values
# plt.figure(figsize=(18,9))
# plt.plot(data.index,data[market])
# plt.legend(loc='upper right')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.title('Price of electricity');
# plt.legend(loc='upper right')
# plt.grid(which='major', axis='both', linestyle='--')
# plt.show()
#
#
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(data[market], lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(data[market], lags=40, ax=ax2)
# plt.show()




