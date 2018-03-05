import numpy as np
import pandas as pd
import pywt
import xlrd
import pyflux as pf
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.robust import mad
import statsmodels.api as sm
import seaborn as sns
from math import sqrt
sns.set(color_codes=True)


'''
    Funkcia na zjemnenie casoveho radu

'''


def wavelet_smooth(x, wavelet="db4", level=1, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold
    sigma = mad(coeff[-level])
    # changing this threshold also changes the behavior
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    X = pywt.waverec(coeff, wavelet, mode="per")

    return X


'''
    Predikcia cien pomocou ARIMA, 
    pouziva sa rolling prediction, v ktorom si predikovane hdnoty ulozim a postupne porovnavam s
    testovacimi hodnotami. 
'''
def prediction_arima_flask(X,market,size):
    # size = int(len(X[market].values) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    t = 0

    while t < len(test):
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        # print(predicted_value)
        q = t + 24
        observation = test[t:q]
        for i in range(len(observation)):
            predictions.append(predicted_value[i])
            history.append(observation[i])
            # print('predicted=%f, expected=%f' % (predicted_value[i], observation[i]))
        t = q

    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # # plotting result
    # plotting_frame = pd.DataFrame(predictions)
    # plotting_frame.index = datum
    #
    # plt.figure(figsize=(18, 9))
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.title('Price of electricity');
    # plt.plot(data.index, data[market].values, label=market)
    # plt.plot(plotting_frame.index, plotting_frame, color='red', label='Predicted')
    # plt.legend(loc='upper right')
    # plt.grid(which='major', axis='both', linestyle='--')
    # plt.show()
    return model_fit,rmse


def prediction_arima(X,market,size):
    # size = int(len(X[market].values) * 0.66)

    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    t = 0

    while t < len(test):
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        #print(predicted_value)
        q = t + 24
        observation = test[t:q]
        for i in range(len(observation)):
            predictions.append(predicted_value[i])
            history.append(observation[i])
            # print('predicted=%f, expected=%f' % (predicted_value[i], observation[i]))
        t = q


    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # # plotting result
    # plotting_frame = pd.DataFrame(predictions)
    # plotting_frame.index = datum
    #
    # plt.figure(figsize=(18, 9))
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.title('Price of electricity');
    # plt.plot(data.index, data[market].values, label=market)
    # plt.plot(plotting_frame.index, plotting_frame, color='red', label='Predicted')
    # plt.legend(loc='upper right')
    # plt.grid(which='major', axis='both', linestyle='--')
    # plt.show()
    return rmse


def main():
    # PREPARING DATA

    # data_xls = pd.read_excel("elspot-prices_2013_hourly_sek.xls", 'elspot-prices_2013_hourly_sek', index_col=None)
   # data_xls.to_csv('prices.csv', encoding='utf-8')

    data_csv = pd.read_csv("prices.csv")

    # Pridanie hodiny k datumu
    dates = data_csv[data_csv.columns[0:2]]
    dates.columns = ['Day', 'Hour']
    dates['Hour'] = dates['Hour'].map(lambda x: str(x)[:2])

    df = pd.DataFrame(dates)
    df['Period'] = df.Day.astype(str).str.cat(df.Hour.astype(str), sep=' ')
    df['Period'] = pd.to_datetime(df["Period"])

    data_csv['Hours'] = df['Period']
    data_csv = data_csv.drop(data_csv.columns[[0]], axis=1)

    # VYBER STLPCA, pre ktory chceme robit predikciu
    market = 'Bergen'
    data = data_csv
    fancy_data = data[market].describe()
    for i in range(len(fancy_data)):
        print(fancy_data[i])
    print(fancy_data)
    # sns.distplot(data[market]);

    # Plots
    data.index = data_csv['Hours'].values
    plt.figure(figsize=(18, 9))
    plt.plot(data.index, data[market])
    plt.legend(loc='upper right')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.title('Price of electricity');
    plt.legend(loc='upper right')
    plt.grid(which='major', axis='both', linestyle='--')
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data[market], lags=30, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data[market], lags=30, ax=ax2)
    plt.show()

    ## ADF statistic If the value is larger than the critical values, again,
    # meaning that we can accept the null hypothesis and in turn that the time series is non-stationary

    print("Fuller test stacionarity test: ", sm.tsa.stattools.adfuller(data[market]))

    size = int(len(data[market].values) * 0.66)
    datum = data['Hours'][size:len(data_csv[market].values)].values

    data[market] = wavelet_smooth(data[market].values)
    prediction_arima(data[market].values,market,size)


if __name__ == '__main__':
    main()

