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



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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
    w = pywt.Wavelet('db4')
    t = 0
    print(len(test))

    while t < len(test):
        cA, cD = pywt.dwt(history, w)
        # print(cA.shape)
        # print(cD.shape)
        model = ARIMA(cA, order=(1, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        # print(predicted_value)

        # model = arch_model(cD, vol='Garch', p=1, o=0, q=1, dist='Normal')
        # results = model.fit()
        #
        #forecasts = results.forecast(horizon=24, method='simulation', simulations=1000)
        #sims = forecasts.simulations.values

        model = pf.GARCH(cD, p=1, q=1)
        x = model.fit()
        # print(x.summary())
        radCD1 = model.predict(h=24)
        pred = np.asarray(radCD1.iloc[:,0])

        z = pywt.upcoef('a', predicted_value, 'db4', take=24)
        zz = pywt.upcoef('d', pred, 'db4',take=24)
        vysledok = z + zz
        #print(vysledok)

        q = t + 24
        observation = test[t:q]

        t = q
        for i in range(len(observation)):
            predictions.append(vysledok[i])
            history.append(observation[i])
            # print('predicted=%f, expected=%f' % (vysledok[i], observation[i]))

        if(t % 25):
            print(t)
        # prediction = pywt.idwt(predicted_value, pred, w)
        # print(prediction)
    #
    #
    #     # radcA = model.predict(h=1, intervals=False)
    #     #  print(radcA)
    #
    #     # novy_cA = output.as_matrix()
    #     # novy_cA = novy_cA[:, 0]
    #
    #     # print(novy_cA.shape)
    #     # print(type(novy_cA))
    #     # print(novy_cA)
    #     # print(output)
    #     # print(type(output))
    #     novy_cA = np.asarray(output[0])
    #
    #
    #
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    print("MAPE testovacie ", mean_absolute_percentage_error(test, predictions))

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
    # return rmse


def main():
    # PREPARING DATA
    data_xls = pd.read_excel("elspot-prices_2013_hourly_sek.xls", 'elspot-prices_2013_hourly_sek', index_col=None)
    data_xls.to_csv('prices.csv', encoding='utf-8')

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
    market = 'Oslo'
    data = data_csv
    data = data[market]
    data.fillna((data.mean()), inplace=True)
    # sns.distplot(data[market]);

    # Plots
    # data.index = data_csv['Hours'].values
    # plt.figure(figsize=(18, 9))
    # plt.plot(data.index, data[market])
    # plt.legend(loc='upper right')
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.title('Price of electricity');
    # plt.legend(loc='upper right')
    # plt.grid(which='major', axis='both', linestyle='--')
    # plt.show()

    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(data[market], lags=30, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(data[market], lags=30, ax=ax2)
    # plt.show()

    ## ADF statistic If the value is larger than the critical values, again,
    # meaning that we can accept the null hypothesis and in turn that the time series is non-stationary

    print("Fuller test stacionarity test: ", sm.tsa.stattools.adfuller(data))

    size = round(0.6 * data.shape[0]) - (round(0.6 * data.shape[0]) % 24)
    datum = data_csv['Hours'][size:len(data)]

    data = wavelet_smooth(data)
    prediction_arima(data,market,size)


if __name__ == '__main__':
    main()



